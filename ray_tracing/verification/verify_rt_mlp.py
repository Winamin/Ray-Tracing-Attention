"""Verify native RT transmission MLP against FP64 SwiGLU on original weights."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from ..compiler.spec import asset_spec
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import LOCAL, default_model


def original_mlp(directory, layer):
    """Decode the exported original payload, independently of the GLSL decoder."""
    spec = asset_spec(directory)
    header = np.fromfile(directory / 'optical_model.bin', '<u4', count=9, offset=8)
    params = np.fromfile(directory / 'optical_model.bin', '<u4', count=int(header[6]), offset=44)
    packed = np.memmap(directory / 'geometry_weights.bin', dtype='u1', mode='r')
    base = 32 + layer * 64
    weights = []
    for field, shape in ((15, (spec.middle_size, spec.hidden_size)),
                         (19, (spec.middle_size, spec.hidden_size)),
                         (17, (spec.hidden_size, spec.middle_size))):
        offset = int(params[base + 27] + params[base + field]) * 4
        code = int(params[base + field + 1])
        if code not in (0, 1, 30, 12, 14):
            raise ValueError(f'Unexpected MLP format {code}')
        count = int(np.prod(shape))
        size = count * {0: 4, 1: 2, 30: 2, 12: 144 / 256, 14: 210 / 256}[code]
        raw = packed[offset:offset + int(size)]
        if code == 0:
            value = raw.view('<f4')
        elif code == 1:
            value = raw.view('<f2').astype(np.float32)
        elif code == 30:
            value = (raw.view('<u2').astype(np.uint32) << 16).view(np.float32)
        else:
            from gguf import GGMLQuantizationType, dequantize
            value = dequantize(raw, GGMLQuantizationType(code))
        weights.append(value.reshape(shape).astype(np.float64))
    return weights


def sigmoid(value):
    light = np.exp(-np.abs(value))
    return np.where(value >= 0, 1 / (1 + light), light / (1 + light))


def check_close(actual, reference):
    assert np.isfinite(actual).all(), 'Nonfinite RT detector output'
    error = float(np.max(np.abs(actual - reference)))
    scale = max(1e-12, float(np.max(np.abs(reference))))
    assert error <= 3e-6 * max(1.0, scale), (error, scale)
    return {'max_abs_error': error, 'max_error_over_peak': error / scale}


def verify(directory, layers, tile=8, row_counts=(1, 7, 8, 17, 32)):
    directory = Path(directory).resolve()
    spec = asset_spec(directory)
    rng = np.random.default_rng(20261110)
    inputs = rng.normal(0, .25, (32, spec.hidden_size)).astype(np.float32)
    inputs[1] = 0
    inputs[2] = -inputs[0]
    inputs[3] = np.where(np.arange(spec.hidden_size) % 2, -.5, .5)
    report = {'model': str(directory), 'tile': tile, 'checks': []}
    with ResidentRuntime(directory / 'optical_model.bin', capacity=32, cache=False,
                         prefill_batch=32, matrix_rows=tile,mlp_backend='rt_transmission') as runtime:
        report['mlp_compute'] = runtime.metadata['mlp_compute']
        # Confirm the built MLP module contains real ray queries and no matrix
        # accelerator instructions, even when other projections enable them.
        validator = os.environ.get('RT_SPIRV_VAL')
        disassembler = (str(Path(validator).with_name('spirv-dis.exe')) if validator else
                        shutil.which('spirv-dis') or str(LOCAL / 'tools/glslang/mingw64/bin/spirv-dis.exe'))
        module = Path(runtime.metadata['worker']).with_name('resident_mlp.spv')
        disassembly = subprocess.run([disassembler, str(module)], capture_output=True, text=True, check=True).stdout
        assert 'OpRayQueryInitializeKHR' in disassembly and 'OpRayQueryProceedKHR' in disassembly
        assert 'OpCooperativeMatrix' not in disassembly
        report['spirv_ray_query'] = True
        report['spirv_cooperative_matrix'] = False
        for layer in layers:
            gate_weight, up_weight, down_weight = original_mlp(directory, layer)
            x = inputs.astype(np.float64)
            gate = x @ gate_weight.T
            up = x @ up_weight.T
            activation = gate * sigmoid(gate) * up
            output = activation @ down_weight.T
            previous = None
            for rows in row_counts:
                actual = runtime.mlp(inputs[:rows], layer)
                assert np.all(actual['feature_hits'] == spec.middle_size), actual['feature_hits']
                if rows > 1:
                    assert np.all(actual['activation'][1] == 0) and np.all(actual['output'][1] == 0)
                if previous is not None:
                    np.testing.assert_array_equal(actual['output'][:len(previous)], previous)
                previous = actual['output'].copy()
                report['checks'].append({'layer': layer, 'rows': rows,
                                         'activation': check_close(actual['activation'], activation[:rows]),
                                         'output': check_close(actual['output'], output[:rows]),
                                         'profile_gpu_ms': actual.get('profile_gpu_ms')})
        for bad in (np.full((1, spec.hidden_size), np.nan), np.full((1, spec.hidden_size), np.inf)):
            try:
                runtime.mlp(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Nonfinite input was accepted')
        report['resources'] = runtime.resources()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, nargs='?', default=default_model())
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--tile', type=int, choices=(4, 8, 16), default=8)
    parser.add_argument('--rows', type=int, nargs='+', default=(1, 7, 8, 17, 32))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    directory = args.model.parent if args.model.is_file() else args.model
    spec = asset_spec(directory)
    layers = args.layers if args.layers is not None else range(spec.depth)
    result = verify(directory, layers, args.tile, args.rows)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding='utf8')
    peak = max(check['output']['max_abs_error'] for check in result['checks'])
    print(json.dumps({'checks': len(result['checks']), 'max_abs_output_error': peak,
                      'spirv_ray_query': result['spirv_ray_query'],
                      'spirv_cooperative_matrix': result['spirv_cooperative_matrix']}))


if __name__ == '__main__':
    main()
