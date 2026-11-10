"""Measure compact attention work and latency from supplied Q/K/V."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..compiler.spec import asset_spec
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import default_model
from .verify_attention_receipts import (decode_history, error_metrics,
                                       full_attention, pinned_worker)


def inputs(spec):
    rng = np.random.default_rng(20261110)
    for name, count, rows in [('random_512', 512, 1), ('random_4096', 4096, 1),
                              ('random_batch', 512, 8), ('repeated_packets', 4096, 1),
                              ('two_value_columns', 512, 8), ('equal_q_distinct_k', 512, 1),
                              ('one_key_column', 512, 8), ('uniform_complete', 4096, 1)]:
        q = rng.normal(0, .25, (rows, spec.attention_heads, spec.attention_head_dim)).astype('f4')
        k = rng.normal(0, .25, (count, spec.kv_heads, spec.attention_head_dim)).astype('f4')
        v = rng.normal(0, .5, k.shape).astype('f4')
        if name == 'repeated_packets':
            identity = (np.arange(count) * 7) % 5
            k[:] = k[:5].copy()[identity]
            v[:] = v[:5].copy()[identity]
        elif name == 'two_value_columns':
            v[:] = np.linspace(-1., 1., count, dtype='f4')[:, None, None]
            v *= np.where(np.arange(spec.attention_head_dim) % 2, -1., 1.).astype('f4')
        elif name == 'equal_q_distinct_k':
            q.fill(.01)
        elif name == 'one_key_column':
            k[:] = np.linspace(-.5, .5, count, dtype='f4')[:, None, None]
        elif name == 'uniform_complete':
            q.fill(0)
            k.fill(0)
            v.fill(.5)
        yield name, q, k, v


def benchmark(asset, worker, output, blocks=(128, 256, 512), samples=5):
    directory = asset.resolve()
    if directory.is_file():
        directory = directory.parent
    spec = asset_spec(directory)
    report = dict(status='running', worker=str(worker.resolve()),
                  worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                  shader_sha256={name: hashlib.sha256(worker.with_name(name+'.spv').read_bytes()).hexdigest()
                                 for name in ('resident_attention_receipt', 'resident_attention_dense')},
                  seed=20261110, samples=samples, warmups=1, all_gpu_work_sequential=True,
                  input_kind='Constructed Q/K/V; no claim about model Q distribution',
                  timing_scope='GPU count reset, RT, classification, QK, softmax, V weighting and merge; '
                               'excludes QKV, history packing/BVH build, output projection, MLP and CPU readback',
                  compact_arithmetic='FP32 shader; RT Core performs geometric intersection',
                  cases=[])

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf8')

    with pinned_worker(worker, spec):
        for backend in ('grouped_rt_dense', 'grouped_rt'):
            for block in blocks:
                with ResidentRuntime(directory / 'optical_model.bin', capacity=4096, cache=True,
                                     prefill_batch=32, attention_backend=backend,
                                     attention_block=block) as runtime:
                    for name, q, k, v in inputs(spec):
                        report['running'] = dict(backend=backend, block=block, case=name)
                        save()
                        print(json.dumps(report['running']), flush=True)
                        timings = []
                        for repetition in range(samples + 1):
                            result = runtime.attention(q, k, v, layer=spec.full_layers[0])
                            if repetition:
                                timings.append({key: float(result[key]) for key in
                                                ('gpu_ms', 'fused_gpu_ms', 'merge_gpu_ms', 'legacy_gpu_ms')})
                        key, value = decode_history(result['history_packets'], spec)
                        reference = full_attention(q, key, value, np.zeros_like(q), len(k)-len(q),
                                                   0, 4096, spec.attention_gate)
                        check = error_metrics(result['output'], reference['output'])
                        if not check['passed']:
                            raise AssertionError((backend, block, name, check))
                        receipts = result['receipts']
                        covered = int(receipts[:, 2].astype('u8').sum())
                        record = dict(backend=backend, block=block, case=name, history=len(k),
                                      rows=len(q), heads=spec.attention_heads, dimension=spec.attention_head_dim,
                                      covered_head_positions=covered, dropped_contributions=0,
                                      full_qk_products=covered*spec.attention_head_dim,
                                      full_value_products=covered*spec.attention_head_dim,
                                      records=len(receipts), samples=timings,
                                      median={key: float(np.median([sample[key] for sample in timings]))
                                              for key in timings[0]}, output_FP64=check,
                                      rt_initializations=int(receipts[:, 6].astype('u8').sum()),
                                      exp_evaluations=int(receipts[:, 3].astype('u8').sum()),
                                      value_runs=int(receipts[:, 4].astype('u8').sum()),
                                      receipt_written_bytes=result['receipt_written_bytes'],
                                      receipt_capacity_bytes=int(result.get('compact_receipts', receipts).nbytes),
                                      history_sha256=hashlib.sha256(result['history_packets'].tobytes()).hexdigest())
                        if backend == 'grouped_rt':
                            record['shader_work'] = result['shader_work']
                            work = record['shader_work']
                            record['qk_product_reduction'] = 1-work['qk_products']/record['full_qk_products']
                            record['value_product_reduction'] = 1-work['value_products']/record['full_value_products']
                            record['value_dimension_reduction'] = 1-work['returned_value_dimensions']/(len(receipts)*spec.attention_head_dim)
                        report['cases'].append(record)
                        save()
    # Verify that both algorithms consumed byte-identical production packets.
    packets = {}
    for record in report['cases']:
        previous = packets.setdefault(record['case'], record['history_sha256'])
        assert previous == record['history_sha256'], 'Benchmark inputs changed across algorithms'
    report.update(status='complete', passed=True)
    report.pop('running', None)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=default_model())
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', type=int, choices=(128, 256, 512), nargs='+', default=[128, 256, 512])
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    report = benchmark(args.asset, args.worker, args.output, args.blocks, args.samples)
    print(json.dumps(dict(passed=report['passed'], cases=len(report['cases']))))


if __name__ == '__main__':
    main()
