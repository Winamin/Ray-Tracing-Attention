"""Verify 32 full attention trees and a 64-token telemetry batch on real RT."""
import json
import numpy as np
from .verify_portable import fixture, save_safetensors, reference
from ..compiler.sources import SafetensorsSource
from ..compiler.export import export
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import LOCAL

def main():
    root=LOCAL/'verification/deep';path=fixture('llama',root);src=SafetensorsSource(path)
    cfg=json.loads((path/'config.json').read_text());cfg['num_hidden_layers']=32;cfg['layer_types']=['full_attention']*32
    tensors={}
    for name in src.tensors:
        if '.layers.' in name:continue
        if name=='model.embed_tokens.weight':canonical='token_embd.weight'
        elif name=='lm_head.weight':canonical='output.weight'
        else:canonical='output_norm.weight'
        tensors[name]=(src.floating(canonical),src.tensors[name][2]['dtype'])
    for layer in range(32):
        for suffix in ('attn_norm.weight','post_attention_norm.weight','attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight','ffn_gate.weight','ffn_up.weight','ffn_down.weight'):
            source=f'blk.{layer%2}.{suffix}';actual=src.actual(source);destination=actual.replace(f'.layers.{layer%2}.',f'.layers.{layer}.')
            tensors[destination]=(src.floating(source),src.tensors[actual][2]['dtype'])
    save_safetensors(path/'model.safetensors',tensors);(path/'config.json').write_text(json.dumps(cfg),encoding='utf8')
    src=SafetensorsSource(path);asset,_=export(src,root/'model');expected,_=reference(src,[2,7,11,3,19])
    with ResidentRuntime(asset,128) as rt:
        p=rt.prefill([2,7,11,3,19],debug=True);error=float(abs(p['layers']-expected).max());assert error<2e-4,error
        g=rt.generate(64,telemetry=True);assert len(g['ids'])==64 and not g['control'][3]
        result={'passed':True,'full_layers':32,'layer_error':error,'batch_tokens':64,'rt_query_gpu_ms':g['rt_query_gpu_ms']}
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
