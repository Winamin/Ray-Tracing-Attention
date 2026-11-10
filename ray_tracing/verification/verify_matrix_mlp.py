"""Check original MLP routes against a capture from the reference Git snapshot."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np


def verify(source, asset, capture, reference=None):
    sys.path.insert(0,str(source.resolve()))
    from ray_tracing.runtime.resident_runtime import ResidentRuntime

    options = dict(capacity=32,cache=True,prefill_batch=32)
    if 'mlp_backend' in inspect.signature(ResidentRuntime).parameters:
        options.update(mlp_backend='standard',attention_backend='standard')
    saved = np.load(reference) if reference else None
    report = dict(checks=[],source=str(source.resolve()),asset=str(asset.resolve()))
    values = {}
    with ResidentRuntime(asset/'optical_model.bin',**options) as runtime:
        report['worker_sha256'] = hashlib.sha256(Path(runtime.metadata['worker']).read_bytes()).hexdigest()
        report['metadata'] = runtime.metadata
        rng = np.random.default_rng(20261111)
        x = rng.normal(0,.25,(32,runtime.hidden_size)).astype('f4')
        x[1] = 0
        x[2] = -x[0]
        for layer in sorted({0,runtime.metadata['depth']-1}):
            for rows in (1,7,8,17,32):
                actual = runtime.mlp(x[:rows],layer)
                record = dict(layer=layer,rows=rows,profile_gpu_ms=actual.get('profile_gpu_ms'))
                for field in ('activation','output'):
                    key = f'l{layer}_r{rows}_{field}'
                    value = actual[field]
                    assert np.isfinite(value).all()
                    if rows>1:
                        assert np.all(value[1]==0)
                    values[key] = value
                    if saved is not None:
                        expected = saved[key]
                        error = float(np.max(np.abs(value.astype('f8')-expected)))
                        peak = float(np.max(np.abs(expected)))
                        assert error<=3e-6*max(1.,peak),(key,error,peak)
                        record[field] = dict(bit_identical=bool(np.array_equal(value,expected)),max_abs_error=error)
                report['checks'].append(record)
        report['resources'] = runtime.resources()
        if 'mlp_backend' in options:
            assert report['resources']['mlp_transmission_scene_bytes']==0
            assert report['resources']['mlp_transmission_surfaces']==0
    capture.parent.mkdir(parents=True,exist_ok=True)
    np.savez(capture,**values)
    report['passed'] = True
    capture.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,cases=len(report['checks']),
                          bit_identical=all(c[f]['bit_identical'] for c in report['checks'] for f in ('activation','output')) if reference else None)),flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--asset',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a = p.parse_args()
    verify(a.source_root,a.asset,a.capture,a.reference)


if __name__=='__main__':
    main()
