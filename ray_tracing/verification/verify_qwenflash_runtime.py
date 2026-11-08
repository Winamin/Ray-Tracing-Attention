"""Validate growing native history, cached continuation and per-token receipts."""
from pathlib import Path
import json
import time
import numpy as np
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.prompt_encoder import LocalPromptEncoder
from .fixtures import DEFAULT_OUTPUT

def main():
    encoder=LocalPromptEncoder(DEFAULT_OUTPUT)
    ids=encoder.encode([{'role':'user','content':'请写一个Python函数，判断整数是否为素数。'}])
    with ResidentRuntime(DEFAULT_OUTPUT/'optical_model.bin',32,cache=True) as rt:
        rt.prefill(ids,[248046,248044]);parts=[];times=[];gpu=[]
        for count in (8,8):
            for receipt in rt.stream_generate(count):
                assert len(receipt['ids'])==1
                parts.extend(receipt['ids'].tolist());times.append(time.perf_counter());gpu.append(receipt['gpu_ms'])
        assert rt.growth_events,'Expected history growth during generation'
        # Growth preserves the addressed token paths and their stream hits.
        rt.prefill(ids,[248046,248044]);warm=[]
        for receipt in rt.stream_generate(16):warm.extend(receipt['ids'].tolist())
        assert warm==parts
        rt.prefill(ids,[248046,248044]);first=rt.generate(1);assert first['reply_cache_hit']
        tail=rt.generate(15);assert tail['cached_tokens']==15
        assert first['ids'].tolist()+tail['ids'].tolist()==parts
        rt.prefill(ids,[248046,248044]);cold=rt.generate(16,profile=True)
        assert cold['ids'].tolist()==parts
        other=encoder.encode([{'role':'user','content':'1+1等于多少？只输出数字。'}])
        rt.prefill(other,[248046,248044]);answer=rt.generate(8)
        assert encoder.tokenizer.decode(answer['ids'].tolist())=='2'
        rt.prefill(ids,[248046,248044]);recomputed=rt.generate(16)
        assert recomputed['reply_cache_hit'] and recomputed['ids'].tolist()==parts
        report={'passed':True,'checks':['native per-token receipts','history growth during decode',
            'cached continuation equality','batch/stream token equality','separate context addresses','arithmetic'],
            'text':encoder.tokenizer.decode(parts),'growth':rt.growth_events,
            'median_token_interval_ms':float(np.median(np.diff(times))*1000),
            'stream_gpu_tps':len(gpu)*1000/sum(gpu),'resources':rt.resources()}
    output=DEFAULT_OUTPUT/'validation/runtime.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
