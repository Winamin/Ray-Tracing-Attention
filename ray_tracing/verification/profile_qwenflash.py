"""Profile native uncached generation with the same prompt as the GGUF reference."""
import argparse
import json
import numpy as np
from .fixtures import DEFAULT_OUTPUT
from ..runtime.prompt_encoder import LocalPromptEncoder
from ..runtime.resident_runtime import ResidentRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='block_decode')
    args = parser.parse_args()
    if not args.label.replace('_', '').isalnum():
        parser.error('label must contain letters, digits or underscores')
    prompt = '写一个完整的Python二分查找函数，并解释边界条件。'
    encoder = LocalPromptEncoder(DEFAULT_OUTPUT)
    ids = encoder.encode([{'role': 'user', 'content': prompt}])
    records, tokens, mlp = [], [], []
    with ResidentRuntime(DEFAULT_OUTPUT / 'optical_model.bin', 512, cache=False) as rt:
        x = np.random.default_rng(42).normal(0, .1, 4096).astype(np.float32)
        for layer in (0, 4):
            for _ in range(2):
                m = rt.mlp(x, layer)
            mlp.append({'layer': layer, **m['profile_gpu_ms']})
        prefill = rt.prefill(ids, [248046, 248044])
        for _ in range(3):
            g = rt.generate(16, profile=True)
            tokens.extend(g['ids'].tolist())
            record = {k: v for k, v in g.items() if k not in ('ids', 'control')}
            record['tokens'] = len(g['ids'])
            records.append(record)
            print(json.dumps({'batch': len(records), 'gpu_tps': record['gpu_tokens_per_second'],
                              'profile_ms_per_token': {k: v / record['tokens'] for k, v
                                   in record['profile_gpu_ms'].items()}}), flush=True)
        resources = rt.resources()
    result = {'prompt': prompt, 'prompt_ids': list(ids), 'ids': tokens,
              'text': encoder.tokenizer.decode(tokens), 'mlp': mlp,
              'prefill_gpu_ms': prefill['prefill_gpu_ms'], 'batches': records,
              'gpu_tps': len(tokens) * 1000 / sum(r['gpu_ms'] for r in records),
              'wall_tps': len(tokens) * 1000 / sum(r['submit_wait_ms'] for r in records),
              'resources': resources}
    dest = DEFAULT_OUTPUT / 'validation' / (args.label + '_profile.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'gpu_tps': result['gpu_tps'], 'mlp': mlp,
                      'text': result['text']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
