"""Check native packed-coefficient decoding, optical MLP and actual generation."""
from pathlib import Path
import json
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.prompt_encoder import LocalPromptEncoder
from .fixtures import DEFAULT_GGUF, DEFAULT_OUTPUT

def main():
    reader=GGUFReader(str(DEFAULT_GGUF),'r');tensors={t.name:t for t in reader.tensors}
    def matrix(name):
        t=tensors[name];return dequantize(t.data,t.tensor_type).reshape(tuple(map(int,t.shape[::-1])))
    result={}
    with ResidentRuntime(DEFAULT_OUTPUT/'optical_model.bin',512,cache=False) as rt:
        x=np.random.default_rng(42).normal(0,.1,4096).astype(np.float32)
        v=rt.vocabulary(x)
        tensor=tensors['output.weight']
        source=dequantize(tensor.data[:256],tensor.tensor_type).reshape(256,4096)
        result['head_decode']={'max_error':float(np.max(np.abs(v['geometric_words']-source))),
            'finite':bool(np.isfinite(v['geometric_words']).all()),'token':v['token'],'timings':v['timings'].tolist()}
        del source
        assert result['head_decode']['max_error']==0
        print(json.dumps({'head_decode':result['head_decode']}),flush=True)
        m=rt.mlp(x,0);gate=matrix('blk.0.ffn_gate.weight')@x;up=matrix('blk.0.ffn_up.weight')@x
        activation=gate/(1+np.exp(-gate))*up;expected=matrix('blk.0.ffn_down.weight')@activation
        result['mlp']={'activation_error':float(np.max(np.abs(activation-m['activation'][0]))),
            'output_error':float(np.max(np.abs(expected-m['output'][0]))),'finite':bool(np.isfinite(m['output']).all())}
        assert result['mlp']['finite'] and result['mlp']['output_error']<1e-5
        print(json.dumps({'mlp':result['mlp']}),flush=True)
        result['mlp_formats_and_batches']=[]
        for layer in (0,4):
            inputs=np.random.default_rng(42+layer).normal(0,.1,(4,4096)).astype(np.float32)
            gate=inputs@matrix(f'blk.{layer}.ffn_gate.weight').T
            up=inputs@matrix(f'blk.{layer}.ffn_up.weight').T
            activation=gate/(1+np.exp(-gate))*up
            expected=activation@matrix(f'blk.{layer}.ffn_down.weight').T
            for rows in (1,4):
                m=rt.mlp(inputs[:rows],layer)
                check={'layer':layer,'rows':rows,
                    'down_format':tensors[f'blk.{layer}.ffn_down.weight'].tensor_type.name,
                    'activation_error':float(np.max(np.abs(activation[:rows]-m['activation']))),
                    'output_error':float(np.max(np.abs(expected[:rows]-m['output']))),
                    'feature_hits':m['feature_hits'].tolist()}
                assert np.isfinite(m['output']).all() and check['output_error']<1e-5
                assert np.all(m['feature_hits']==12288)
                result['mlp_formats_and_batches'].append(check)
        print(json.dumps({'mlp_formats_and_batches':result['mlp_formats_and_batches']}),flush=True)
        encoder=LocalPromptEncoder(DEFAULT_OUTPUT);ids=encoder.encode([{'role':'user','content':'你好，请用一句话介绍自己。'}])
        p=rt.prefill(ids,stops=[248046,248044],debug=True)
        result['layers']=[{'layer':i,'finite':bool(np.isfinite(a).all()),'max':float(np.nanmax(np.abs(a)))} for i,a in enumerate(p['layers'])]
        print(json.dumps({'layers':result['layers']}),flush=True)
        g=rt.generate(16,profile=True);result['text']=encoder.tokenizer.decode(g['ids'].tolist());result['speed']=g['gpu_tokens_per_second'];result['resources']=rt.resources()
        print(json.dumps({'text':result['text'],'speed':result['speed']},ensure_ascii=False),flush=True)
    dest=DEFAULT_OUTPUT/'validation';dest.mkdir(exist_ok=True);(dest/'native_quantized.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
