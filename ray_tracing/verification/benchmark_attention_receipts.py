"""Sequential, fixed-worker attention timings, full-model states, and cache checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..compiler.spec import asset_spec
from ..runtime.prompt_encoder import LocalPromptEncoder
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import default_model
from .verify_attention_receipts import pinned_worker


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def small(result):
    return {k: v for k, v in result.items() if k not in ('state', 'layers')}


def delta(before, after):
    return {k: after[k] - before[k] for k in after if k.startswith('attention_') and
            k not in ('attention_backend_code', 'attention_block', 'attention_receipt_bytes')}


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    difference = actual - expected
    return dict(max_abs=float(np.max(np.abs(difference))),
                rms=float(np.sqrt(np.mean(difference**2))),
                relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)),
                bitwise_equal=bool(np.array_equal(actual, expected)),
                finite=bool(np.all(np.isfinite(actual))))


def benchmark(asset, worker, output, backends, blocks, samples, direct=True, model=False):
    asset = asset.resolve()
    directory = asset.parent if asset.is_file() else asset
    spec = asset_spec(directory)
    if model:
        encoder = LocalPromptEncoder(directory)
        baseline = json.loads(Path('.local/diagnostics/attention_receipts/baseline.json').read_text())
        prompt = baseline['prompt_ids']
        teacher = prompt + baseline['ids']
    report = dict(status='running', worker=str(worker.resolve()),
                  worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                  all_gpu_work_sequential=True, cache_default_on=True, samples=samples,
                  timing_scope='GPU timestamps; direct diagnostic forces summary and merge',
                  direct=[], model=[], state_comparisons=[], cache_checks=[])
    states = {}

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=serial), encoding='utf8')

    def event(value):
        report['running'] = value
        save()
        print(json.dumps(value), flush=True)

    configs = [(backend, block) for backend in backends for block in
               (blocks if backend == 'grouped_rt' else [256])]
    with pinned_worker(worker, spec):
        for backend, block in configs:
            name = f'{backend}_b{block}'
            event(dict(config=name, phase='startup'))
            started = time.perf_counter()
            with ResidentRuntime(directory / 'optical_model.bin', capacity=4096, cache=True,
                                 prefill_batch=32, attention_backend=backend,
                                 attention_block=block) as runtime:
                startup = time.perf_counter() - started
                if direct and backend == 'grouped_rt':
                    rng = np.random.default_rng(20261110)
                    for count, rows in [(17, 1), (128, 1), (256, 1), (512, 1),
                                        (1024, 1), (4096, 1), (512, 8), (4096, 8), (4096, 32)]:
                        event(dict(config=name, phase='direct', history=count, rows=rows))
                        q = rng.normal(size=(rows, spec.attention_heads, spec.attention_head_dim)).astype('f4')
                        k = rng.normal(size=(count, spec.kv_heads, spec.attention_head_dim)).astype('f4')
                        v = rng.normal(size=k.shape).astype('f4')
                        timings = []
                        for repetition in range(samples + 1):
                            result = runtime.attention(q, k, v, layer=spec.full_layers[0])
                            if repetition:
                                timings.append({key: float(result[key]) for key in
                                                ('gpu_ms', 'fused_gpu_ms', 'merge_gpu_ms', 'legacy_gpu_ms')})
                        receipt = result['receipts']
                        measured = {key: float(np.median([sample[key] for sample in timings]))
                                    for key in timings[0]}
                        report['direct'].append(dict(config=name, history=count, rows=rows,
                                                     samples=timings, median=measured,
                                                     records=len(receipt), receipt_bytes=int(receipt.nbytes),
                                                     covered_terms=int(receipt[:, 2].astype('u8').sum()),
                                                     rt_initializations=int(receipt[:, 6].astype('u8').sum()),
                                                     exp_evaluations=int(receipt[:, 3].astype('u8').sum()),
                                                     value_runs=int(receipt[:, 4].astype('u8').sum())))
                        save()
                if not model:
                    continue
                event(dict(config=name, phase='model_25_plus_32'))
                # Layer capture recomputes the forward pass.
                prefill = runtime.prefill(prompt, debug=True)
                before = runtime.resources()
                generated = runtime.generate(32, profile=True)
                after = runtime.resources()
                if generated['cached_tokens']:
                    raise AssertionError('Fresh fixed-worker timing unexpectedly reused reply tokens')
                production = dict(config=name, startup_seconds=startup, metadata=runtime.metadata,
                                  prefill_capture=small(prefill), generation=small(generated),
                                  compute_resources_delta=delta(before, after))
                event(dict(config=name, phase='teacher_forced_state'))
                capture = runtime.prefill(teacher, debug=True)
                final = runtime.generate(1, debug=True)
                states[name] = dict(prompt_state=prefill['state'], prompt_layers=prefill['layers'],
                                    teacher_state=capture['state'], teacher_layers=capture['layers'],
                                    final_state=final['state'])
                state_file = output.parent / f'{output.stem}_{name}_states.npz'
                np.savez_compressed(state_file, **states[name])
                production['state_capture'] = str(state_file)
                production['teacher_forced_next_ids'] = final['ids']
                report['model'].append(production)
                event(dict(config=name, phase='cache'))
                # Normal prefill enables cache; profiling-prefill intentionally recomputes.
                runtime.prefill(prompt)
                first_reply = runtime.generate(32, telemetry=True)
                before = runtime.resources()
                hit = runtime.prefill(prompt)
                after_prefill = runtime.resources()
                replay = runtime.generate(32, telemetry=True)
                after_reply = runtime.resources()
                check = dict(config=name, prefill=small(hit), replay=small(replay),
                             prefill_delta=delta(before, after_prefill),
                             replay_delta=delta(after_prefill, after_reply),
                             same_output_ids=bool(np.array_equal(first_reply['ids'], replay['ids'])))
                check['passed'] = (hit['history_address_hit'] == 1 and
                                   hit['prefill_computed_tokens'] == 0 and
                                   replay['compute_tokens'] == 0 and check['same_output_ids'] and
                                   all(value == 0 for value in check['prefill_delta'].values()) and
                                   all(value == 0 for value in check['replay_delta'].values()))
                if not check['passed']:
                    raise AssertionError(f'Cache failed for {name}: {check}')
                report['cache_checks'].append(check)
                if backend != 'grouped_rt':
                    production['resources'] = runtime.resources()
                    save()
                    continue
                event(dict(config=name, phase='four_round_agent'))
                messages = []
                rounds = []
                instructions = [
                    '写Python函数 clamp(x,lo,hi)，输出代码。',
                    '测试失败：lo>hi没有报错。定位原因。',
                    '修复函数：lo>hi抛出ValueError。',
                    'review修复，指出边界测试。',
                ]
                for number, instruction in enumerate(instructions, 1):
                    messages.append(dict(role='user', content=instruction))
                    ids, boundaries = encoder.encode_with_boundaries(messages)
                    before = runtime.resources()
                    started = time.perf_counter()
                    p = runtime.prefill(ids, encoder.stop_ids, checkpoint_boundaries=boundaries)
                    g = runtime.generate(8, profile=True)
                    after = runtime.resources()
                    text = encoder.tokenizer.decode(g['ids'].tolist())
                    messages.append(dict(role='assistant', content=text))
                    rounds.append(dict(round=number, input_tokens=len(ids), text=text,
                                       wall_seconds=time.perf_counter()-started, prefill=small(p),
                                       generation=small(g), resources_delta=delta(before, after)))
                    event(dict(config=name, phase='four_round_agent', round=number,
                               token_s=g['gpu_tokens_per_second']))
                production['four_rounds'] = rounds
                production['resources'] = runtime.resources()
                save()
    if model:
        reference_name = 'standard_fp32_b256'
        if reference_name in states:
            for name, capture in states.items():
                if name == reference_name:
                    continue
                comparison = dict(config=name, reference=reference_name,
                                  fields={key: metrics(value, states[reference_name][key])
                                          for key, value in capture.items()})
                comparison['per_layer_teacher_max_abs'] = [
                    metrics(capture['teacher_layers'][layer], states[reference_name]['teacher_layers'][layer])['max_abs']
                    for layer in range(spec.depth)]
                report['state_comparisons'].append(comparison)
    report['status'] = 'complete'
    report.pop('running', None)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=default_model())
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=('standard', 'standard_fp32', 'grouped_rt'),
                        default=['standard', 'standard_fp32', 'grouped_rt'])
    parser.add_argument('--blocks', nargs='+', type=int, choices=(128, 256, 512), default=[128, 256, 512])
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--no-direct', action='store_true')
    parser.add_argument('--include-model', dest='model', action='store_true',
                        help='Explicitly opt into full-model state/cache/generation checks')
    parser.add_argument('--no-model', dest='model', action='store_false')
    parser.set_defaults(model=False)
    args = parser.parse_args()
    benchmark(args.asset, args.worker, args.output, args.backends, args.blocks, args.samples,
              not args.no_direct, args.model)


if __name__ == '__main__':
    main()
