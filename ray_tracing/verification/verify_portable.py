"""Validate independent model shapes and formats on real Vulkan RT hardware."""
import json
import struct
from pathlib import Path
import numpy as np
from ..compiler.export import export
from ..compiler.sources import SafetensorsSource
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import LOCAL


def save_safetensors(path, tensors):
    header={};payload=[];offset=0
    for name,(array,kind) in tensors.items():
        array=np.asarray(array,dtype='<f4')
        data=(array.view('<u4')>>16).astype('<u2').tobytes() if kind=='BF16' else array.astype('<f2' if kind=='F16' else '<f4').tobytes()
        header[name]={'dtype':kind,'shape':list(array.shape),'data_offsets':[offset,offset+len(data)]};offset+=len(data);payload.append(data)
    encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
    with path.open('wb') as f:f.write(struct.pack('<Q',len(encoded)));f.write(encoded);f.writelines(payload)


def fixture(arch,base=None,tied=False):
    path=(Path(base) if base else LOCAL/'verification'/arch)/'source';path.mkdir(parents=True,exist_ok=True)
    hybrid=arch=='qwen35';hd=32 if arch=='llama' else 128;heads=8 if arch=='llama' else 4
    cfg={'model_type':'qwen3_5_text' if hybrid else arch,'hidden_size':256,'intermediate_size':512,
        'vocab_size':37,'num_attention_heads':heads,'num_key_value_heads':2 if arch=='llama' else 1,
        'head_dim':hd,'num_hidden_layers':3 if hybrid else 2,'rms_norm_eps':1e-6,'rope_theta':10000.0,
        'attn_output_gate':hybrid,'layer_types':['full_attention','linear_attention','full_attention'] if hybrid else ['full_attention']*2,
        'linear_num_value_heads':4,'linear_num_key_heads':2,'linear_key_head_dim':128,'linear_value_head_dim':128}
    (path/'config.json').write_text(json.dumps(cfg),encoding='utf8')
    from tokenizers import Tokenizer, models
    Tokenizer(models.WordLevel({str(i):i for i in range(37)},unk_token='0')).save(str(path/'tokenizer.json'))
    tensors={};rng=np.random.default_rng(17)
    def add(name,shape,kind='F32',value=None):
        a=np.full(shape,value,dtype=np.float32) if value is not None else rng.normal(0,.02,shape).astype(np.float32)
        tensors[name]=(a,kind)
    add('model.embed_tokens.weight',(37,256),'BF16')
    if not tied:add('lm_head.weight',(37,256),'F16')
    add('model.norm.weight',(256,),value=0 if hybrid else 1)
    for l,kind in enumerate(cfg['layer_types']):
        p=f'model.layers.{l}.'
        for norm in ('input_layernorm.weight','post_attention_layernorm.weight'):add(p+norm,(256,),value=0 if hybrid else 1)
        add(p+'mlp.gate_proj.weight',(512,256),'F16');add(p+'mlp.up_proj.weight',(512,256),'BF16');add(p+'mlp.down_proj.weight',(256,512),'F32')
        if kind=='full_attention':
            q=heads*hd;k=cfg['num_key_value_heads']*hd
            for name,width in (('q_proj',q*(2 if hybrid else 1)),('k_proj',k),('v_proj',k)):
                add(p+'self_attn.'+name+'.weight',(width,256),'BF16')
            add(p+'self_attn.o_proj.weight',(256,q),'F16')
            if arch!='llama':
                for name in ('q_norm','k_norm'):add(p+'self_attn.'+name+'.weight',(hd,),value=0 if hybrid else 1)
        else:
            for name,rows in (('in_proj_qkv',1024),('in_proj_z',512),('in_proj_a',4),('in_proj_b',4)):
                add(p+'linear_attn.'+name+'.weight',(rows,256),'BF16')
            add(p+'linear_attn.out_proj.weight',(256,512),'F16');add(p+'linear_attn.conv1d.weight',(1024,1,4),'F32')
            add(p+'linear_attn.dt_bias',(4,),value=-2);add(p+'linear_attn.A_log',(4,),value=-1)
            add(p+'linear_attn.norm.weight',(128,),value=1)
    save_safetensors(path/'model.safetensors',tensors)
    return path


def sigmoid(x):return 1/(1+np.exp(-x))
def norm(x,weight,eps=1e-6):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*weight

def quantize_history(x,content):
    flat=np.pad(x.reshape(-1),(0,(-x.size)%256));result=np.zeros_like(flat)
    for block in range(len(flat)//256):
        v=flat[block*256:(block+1)*256]
        if content:
            v=v.reshape(16,16);ds=np.max(abs(v),axis=1)/31;d=np.float32(np.float16(max(ds)/127))
            scales=np.clip(np.rint(ds/max(d,1e-30)),0,127);q=np.clip(np.rint(v/np.maximum(d*scales[:,None],1e-30)),-32,31)
            out=d*scales[:,None]*q
        else:
            v=v.reshape(8,32);lo=np.minimum(np.min(v,axis=1),0);hi=np.maximum(np.max(v,axis=1),0);ds=(hi-lo)/15;mins=-lo
            d=np.float32(np.float16(max(ds)/63));dm=np.float32(np.float16(max(mins)/63))
            scales=np.clip(np.rint(ds/max(d,1e-30)),0,63);minimums=np.clip(np.rint(mins/max(dm,1e-30)),0,63)
            q=np.clip(np.rint((v+dm*minimums[:,None])/np.maximum(d*scales[:,None],1e-30)),0,15)
            out=d*scales[:,None]*q-dm*minimums[:,None]
        result[block*256:(block+1)*256]=out.reshape(-1)
    return result[:x.size].reshape(x.shape)

def reference(src,ids):
    spec=src.spec;state=src.floating('token_embd.weight')[ids].copy();layers=[]
    for l,kind in enumerate(spec.layer_types):
        p=f'blk.{l}.';x=norm(state,src.norm(p+'attn_norm.weight'))
        if kind=='full':
            h=spec.attention_heads;hd=spec.attention_head_dim
            projected=x@src.floating(p+'attn_q.weight').T
            if spec.attention_gate:
                projected=projected.reshape(len(ids),h,2,hd);q=projected[:,:,0];gate=projected[:,:,1]
            else:q=projected.reshape(len(ids),h,hd);gate=None
            k=(x@src.floating(p+'attn_k.weight').T).reshape(len(ids),spec.kv_heads,hd)
            v=(x@src.floating(p+'attn_v.weight').T).reshape(len(ids),spec.kv_heads,hd)
            if spec.qk_norm:q=norm(q,src.norm(p+'attn_q_norm.weight'));k=norm(k,src.norm(p+'attn_k_norm.weight'))
            perm=np.arange(hd);r=spec.rotary_dim;perm[:r]=np.stack([np.arange(r//2),np.arange(r//2)+r//2],-1).ravel()
            q=q[...,perm].copy();k=k[...,perm].copy();v=v[...,perm].copy()
            for t in range(len(ids)):
                phase=t*spec.rope_theta**(-np.arange(r//2,dtype=np.float32)/max(r//2,1))
                for tensor in (q,k):
                    a=tensor[t,:,0:r:2].copy();b=tensor[t,:,1:r:2].copy()
                    tensor[t,:,0:r:2]=a*np.cos(phase)-b*np.sin(phase);tensor[t,:,1:r:2]=a*np.sin(phase)+b*np.cos(phase)
                k[t]=quantize_history(k[t],False);v[t]=quantize_history(v[t],True)
            k=np.repeat(k,h//spec.kv_heads,axis=1);v=np.repeat(v,h//spec.kv_heads,axis=1)
            scores=np.einsum('thd,jhd->thj',q,k)/np.sqrt(hd)
            scores=np.where(np.arange(len(ids))[None,None,:]<=np.arange(len(ids))[:,None,None],scores,-np.inf)
            weights=np.exp(scores-np.max(scores,axis=-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
            out=np.einsum('thj,jhd->thd',weights,v)
            if gate is not None:out*=sigmoid(gate[...,perm])
            inverse=np.argsort(perm);out=out[...,inverse].reshape(len(ids),spec.query_width)
            state+=out@src.floating(p+'attn_output.weight').T
        else:
            key=spec.state_key_heads*128;s=spec.state_width;raw=x@src.floating(p+'attn_qkv.weight').T
            conv=src.floating(p+'ssm_conv1d.weight').reshape(2*key+s,4);z=x@src.floating(p+'attn_gate.weight').T
            frames=np.zeros_like(raw)
            for t in range(len(ids)):
                for j in range(4):
                    if t+j-3>=0:frames[t]+=raw[t+j-3]*conv[:,j]
            frames*=sigmoid(frames);q=np.tile(frames[:,:key],(1,s//key)).reshape(len(ids),spec.state_heads,128)
            k=np.tile(frames[:,key:2*key],(1,s//key)).reshape(len(ids),spec.state_heads,128);v=frames[:,2*key:].reshape(len(ids),spec.state_heads,128)
            q/=np.sqrt((q*q).sum(-1,keepdims=True)+1e-6);k/=np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
            a=x@src.floating(p+'ssm_alpha.weight').T+src.floating(p+'ssm_dt.bias')
            decay=np.exp(-np.exp(src.floating(p+'ssm_a'))*np.logaddexp(0,a));beta=sigmoid(x@src.floating(p+'ssm_beta.weight').T)
            memory=np.zeros((spec.state_heads,128,128),np.float32);out=np.zeros_like(v)
            for t in range(len(ids)):
                memory*=decay[t,:,None,None]
                read=np.einsum('hv,hkv->hk',k[t],memory);delta=(v[t]-read)*beta[t,:,None]
                memory+=delta[:,:,None]*k[t,:,None,:]
                out[t]=np.einsum('hv,hkv->hk',q[t]/np.sqrt(128),memory)
            out=norm(out,src.floating(p+'ssm_norm.weight')).reshape(len(ids),s);out*=z*sigmoid(z)
            state+=out@src.floating(p+'ssm_out.weight').T
        x=norm(state,src.norm(p+'post_attention_norm.weight'));g=x@src.floating(p+'ffn_gate.weight').T;up=x@src.floating(p+'ffn_up.weight').T
        state+=(g*sigmoid(g)*up)@src.floating(p+'ffn_down.weight').T;layers.append(state.copy())
    return np.asarray(layers),norm(state,src.norm('output_norm.weight'))


def main():
    report=[]
    for arch in ('qwen3','llama','qwen35'):
        source=fixture(arch);src=SafetensorsSource(source);output=source.parent/'model';asset,_=export(src,output)
        spec=src.spec;ids=[2,7,11,3,19];expected,final=reference(src,ids)
        with ResidentRuntime(asset,32,cache=True) as rt:
            p=rt.prefill(ids,debug=True)
            error=float(np.max(abs(p['layers']-expected)));assert error<2e-4,(arch,error)
            x=np.random.default_rng(42).normal(0,.1,(3,spec.hidden_size)).astype(np.float32)
            g=x@src.floating('blk.0.ffn_gate.weight').T;up=x@src.floating('blk.0.ffn_up.weight').T
            m=rt.mlp(x);mlp_error=float(np.max(abs(m['output']-(g*sigmoid(g)*up)@src.floating('blk.0.ffn_down.weight').T)))
            assert mlp_error<1e-6 and np.all(m['feature_hits']==spec.middle_size),(arch,mlp_error,m['feature_hits'])
            rt.prefill(ids);a=rt.generate(8,telemetry=True);rt.prefill(ids);b=rt.generate(8,telemetry=True)
            assert np.array_equal(a['ids'],b['ids']) and b['cached_tokens']==8
            rt.prefill(ids);word=rt.vocabulary(final[-1]);winner=int(np.argmax(final[-1]@src.floating('output.weight').T))
            assert word['token']==winner and word['hit_count']==37,(arch,word['token'],winner,word['hit_count'])
            item={'architecture':arch,'model_spec':spec.to_dict(),'layer_error':error,'mlp_error':mlp_error,'vocabulary_hits':word['hit_count'],'cache_hits':b['cached_tokens']}
            report.append(item);print(json.dumps(item),flush=True)
    path=LOCAL/'verification/portable.json';path.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Portable native model verification passed')

if __name__=='__main__':main()
