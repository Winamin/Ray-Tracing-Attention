"""Check GGUF Llama rotary layout against an independent safetensors reference."""
import json
import numpy as np
from gguf import GGUFWriter
from .verify_portable import fixture, reference
from ..compiler.sources import SafetensorsSource
from ..compiler.export import export
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import LOCAL

def main():
    root=LOCAL/'verification/gguf_adapter';source=fixture('llama',root);src=SafetensorsSource(source);spec=src.spec
    path=root/'llama.gguf';writer=GGUFWriter(str(path),'llama')
    for name,value in {'embedding_length':256,'block_count':2,'feed_forward_length':512,'attention.head_count':8,'attention.head_count_kv':2,'attention.key_length':32,'rope.dimension_count':32}.items():writer.add_uint32('llama.'+name,value)
    writer.add_float32('llama.rope.freq_base',10000);writer.add_float32('llama.attention.layer_norm_rms_epsilon',1e-6)
    writer.add_string('tokenizer.ggml.model','gpt2');writer.add_string('tokenizer.ggml.pre','llama-bpe')
    writer.add_array('tokenizer.ggml.tokens',['a','b','ab']+[f'<|t{i}|>' for i in range(34)])
    writer.add_array('tokenizer.ggml.token_type',[1]*3+[3]*34);writer.add_array('tokenizer.ggml.merges',['a b'])
    writer.add_string('tokenizer.chat_template',"{% for message in messages %}{{ message['content'] }}{% endfor %}")
    names=['token_embd.weight','output.weight','output_norm.weight']
    for layer in range(2):names += [f'blk.{layer}.'+n for n in ('attn_norm.weight','post_attention_norm.weight','attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight','ffn_gate.weight','ffn_up.weight','ffn_down.weight')]
    perm=np.stack((np.arange(16),np.arange(16)+16),-1).ravel()
    for name in names:
        value=src.floating(name)
        if name.endswith(('attn_q.weight','attn_k.weight')):value=value.reshape(-1,32,256)[:,perm,:].reshape(value.shape)
        writer.add_tensor(name,np.ascontiguousarray(value))
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    asset,_=export(path,root/'model');expected,_=reference(src,[2,7,11,3,19])
    with ResidentRuntime(asset,32) as rt:
        actual=rt.prefill([2,7,11,3,19],debug=True)['layers']
    error=float(np.max(abs(expected-actual)));assert error<2e-4,error
    result={'passed':True,'architecture':'llama','source':'GGUF F32 with interleaved Q/K rows','layer_error':error}
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
