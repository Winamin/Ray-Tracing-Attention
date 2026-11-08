"""Validate native per-token RT cache paths against uncached generation."""
import json
import time
import numpy as np
from .fixtures import DEFAULT_OUTPUT
from ..runtime.prompt_encoder import LocalPromptEncoder
from ..runtime.resident_runtime import ResidentRuntime


def flatten(records):
    return [int(token) for record in records for token in record['ids']]


def main():
    encoder = LocalPromptEncoder(DEFAULT_OUTPUT)
    ids = encoder.encode([{'role': 'user', 'content':
        '写一个完整的Python二分查找函数，并解释边界条件。'}])
    stops = [248046, 248044]
    with ResidentRuntime(DEFAULT_OUTPUT / 'optical_model.bin', 256, cache=False) as rt:
        rt.prefill(ids, stops)
        reference = flatten([rt.generate(64), rt.generate(64), rt.generate(16)])
    assert len(reference) == 144, 'The coding fixture must continue past the cache boundary'
    print('Uncached coding reference: 144 tokens', flush=True)
    report = {'checks': [], 'prompt_tokens': len(ids)}
    with ResidentRuntime(DEFAULT_OUTPUT / 'optical_model.bin', 32, cache=True) as rt:
        rt.prefill(ids, stops)
        warm = list(rt.stream_generate(64)) + list(rt.stream_generate(64))
        assert flatten(warm) == reference[:128]
        assert all(len(r['ids']) == 1 and r['cached_tokens'] == 0 for r in warm)
        assert rt.growth_events
        report['checks'] += ['per-token writes match uncached generation', 'paths survive history growth']
        report['cold_gpu_tps'] = 128000 / sum(r['gpu_ms'] for r in warm)
        allocation = rt.resources()

        rt.prefill(ids, stops)
        chunks = [rt.generate(n, telemetry=True) for n in (7, 1, 56, 64)]
        assert flatten(chunks) == reference[:128]
        assert all(r['cached_tokens'] == len(r['ids']) for r in chunks)
        assert sum(r['tree_cache_queries'] for r in chunks) == 128
        assert all(r['mlp_feature_hits'] == r['vocabulary_hits'] == 0 for r in chunks)
        tail = rt.generate(8, telemetry=True)
        assert tail['ids'].tolist() == reference[128:136] and tail['cached_tokens'] == 0
        assert tail['cache_checkpoint_restored']
        report['checks'] += ['batch-independent hits', 'one ray per cached token', 'checkpoint continuation equality']
        report['checkpoint_restore_ms'] = tail['cache_state_gpu_ms']

        start = time.perf_counter()
        prefill = rt.prefill(ids, stops)
        generation_start = time.perf_counter()
        replay = list(rt.stream_generate(64)) + list(rt.stream_generate(64))
        generation_ms = (time.perf_counter() - generation_start) * 1000
        request_ms = (time.perf_counter() - start) * 1000
        assert flatten(replay) == reference[:128]
        assert all(len(r['ids']) == 1 and r['cached_tokens'] == 1 and
                   r['tree_cache_queries'] == 1 and r['tree_cache_hits'] == 1 for r in replay)
        after = rt.resources()
        assert allocation['reply_tree_bytes'] == after['reply_tree_bytes']
        assert allocation['cache_snapshot_bytes'] == after['cache_snapshot_bytes']
        report['checks'] += ['128 continuous native stream hits', 'fixed cache state allocation']
        report['stream'] = {'tokens': 128, 'cached_tokens': sum(r['cached_tokens'] for r in replay),
            'tree_queries': sum(r['tree_cache_queries'] for r in replay),
            'gpu_ms': sum(r['gpu_ms'] for r in replay), 'generation_wall_ms': generation_ms,
            'request_wall_ms': request_ms, 'generation_wall_tps': 128000 / generation_ms,
            'request_wall_tps': 128000 / request_ms, 'prefill_gpu_ms': prefill['prefill_gpu_ms'],
            'median_gpu_us_per_token': float(np.median([r['gpu_ms'] for r in replay]) * 1000)}
        print(json.dumps({'stream': report['stream']}), flush=True)

        other = encoder.encode([{'role': 'user', 'content': '1+1等于多少？只输出数字。'}])
        rt.prefill(other, stops)
        answer = rt.generate(8)
        assert answer['cached_tokens'] == 0 and encoder.tokenizer.decode(answer['ids'].tolist()) == '2'
        rt.prefill(ids, stops)
        returned = [rt.generate(64), rt.generate(64), rt.generate(8)]
        assert flatten(returned) == reference[:136]
        assert all(r['cached_tokens'] == len(r['ids']) for r in returned)
        new_tail = rt.generate(8, telemetry=True)
        assert new_tail['cached_tokens'] == 0 and new_tail['ids'].tolist() == reference[136:144]
        assert not new_tail['cache_checkpoint_restored'] and new_tail['cache_state_tokens'] == 136
        report['checks'] += ['different input misses', 'old task paths remain reusable', 'continuation after switching tasks']
        report['state_replay_ms'] = new_tail['cache_state_gpu_ms']

        rt.prefill(ids + reference[:12], stops)
        interior = rt.generate(16)
        assert interior['cached_tokens'] == 16 and interior['ids'].tolist() == reference[12:28]
        rt.prefill(ids, [248044])
        changed_stop = rt.generate(1)
        assert changed_stop['cached_tokens'] == 0 and changed_stop['ids'].tolist() == reference[:1]
        report['checks'] += ['context can start inside a saved path', 'decoding settings participate in address']
        report['cache_nodes'] = changed_stop['cache_nodes']
        report['resources'] = rt.resources()
        report['text'] = encoder.tokenizer.decode(reference)
    report['passed'] = True
    destination = DEFAULT_OUTPUT / 'validation' / 'rt_cache_chain.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'passed': True, 'checks': report['checks'], 'cache_nodes': report['cache_nodes']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
