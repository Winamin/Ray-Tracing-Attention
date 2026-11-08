"""Memory-mapped GGUF and safetensors inputs, with architecture adapters."""
from pathlib import Path
import json
import struct
import numpy as np
from .spec import ModelSpec


class GGUFSource:
    def __init__(self, path):
        from gguf import GGUFReader
        self.path=Path(path).resolve();self.reader=GGUFReader(str(self.path),'r')
        self.tensors={t.name:t for t in self.reader.tensors}
        arch=self.field('general.architecture')
        if arch not in ('qwen35','qwen3','llama'): raise ValueError(f'Architecture {arch} needs an operator adapter')
        prefix=arch+'.';depth=int(self.field(prefix+'block_count'))
        scaling=self.field(prefix+'rope.scaling.type','none')
        if scaling not in ('none','default'):raise ValueError(f'RoPE scaling {scaling!r} needs its position adapter')
        if any(k.endswith(('attn_q.bias','attn_k.bias','attn_v.bias','attn_output.bias','ffn_gate.bias','ffn_up.bias','ffn_down.bias')) for k in self.tensors):
            raise ValueError('This native graph uses unbiased attention projections and optical SwiGLU')
        heads=int(self.field(prefix+'attention.head_count'));d=int(self.field(prefix+'embedding_length'))
        hd=int(self.field(prefix+'attention.key_length',d//heads))
        self.spec=ModelSpec(arch,d,int(self.field(prefix+'feed_forward_length')),len(self.field('tokenizer.ggml.tokens')),
            tuple('recurrent' if f'blk.{i}.attn_qkv.weight' in self.tensors else 'full' for i in range(depth)),
            heads,int(self.field(prefix+'attention.head_count_kv')),hd,int(self.field(prefix+'rope.dimension_count',hd)),
            float(self.field(prefix+'rope.freq_base',10000.0)),float(self.field(prefix+'attention.layer_norm_rms_epsilon',1e-6)),
            arch=='qwen35',f'blk.0.attn_q_norm.weight' in self.tensors or any(k.endswith('attn_q_norm.weight') for k in self.tensors),
            int(self.field(prefix+'ssm.time_step_rank',1)),int(self.field(prefix+'ssm.group_count',1)),int(self.field(prefix+'ssm.state_size',128))).validate()
        self.norm_offset=0.0
    def field(self, name, default=None):
        return self.reader.fields[name].contents() if name in self.reader.fields else default
    def actual(self,name): return 'token_embd.weight' if name=='output.weight' and name not in self.tensors else name
    def has(self,name): return self.actual(name) in self.tensors
    def shape(self,name): return tuple(map(int,self.tensors[self.actual(name)].shape[::-1]))
    def packed(self,name):
        t=self.tensors[self.actual(name)];return np.ascontiguousarray(t.data).view(np.uint8).reshape(-1),int(t.tensor_type),t.tensor_type.name
    def floating(self,name):
        from gguf.quants import dequantize
        t=self.tensors[self.actual(name)]
        return dequantize(t.data,t.tensor_type).reshape(self.shape(name)).astype(np.float32)
    def norm(self,name): return self.floating(name)


class SafetensorsSource:
    def __init__(self,path):
        self.path=Path(path).resolve();raw=json.loads((self.path/'config.json').read_text(encoding='utf8'))
        cfg=raw.get('text_config',raw);arch=cfg.get('model_type','')
        if arch in ('qwen3_5_text','qwen3_5'): arch='qwen35'
        if arch not in ('qwen35','qwen3','llama'): raise ValueError(f'Architecture {arch} needs an operator adapter')
        if cfg.get('hidden_act','silu')!='silu' or cfg.get('attention_bias',False): raise ValueError('This operator graph uses unbiased projections and optical SwiGLU')
        rope=cfg.get('rope_parameters',{})
        scaling=cfg.get('rope_scaling')
        if scaling and scaling.get('rope_type',scaling.get('type','default'))!='default': raise ValueError('Scaled RoPE needs its own position adapter')
        if rope.get('rope_type','default')!='default': raise ValueError('Scaled RoPE needs its own position adapter')
        heads=cfg['num_attention_heads'];hd=cfg.get('head_dim',cfg['hidden_size']//heads)
        layers=cfg.get('layer_types',['full_attention']*cfg['num_hidden_layers'])
        self.spec=ModelSpec(arch,cfg['hidden_size'],cfg['intermediate_size'],cfg['vocab_size'],
            tuple('full' if k=='full_attention' else 'recurrent' for k in layers),heads,cfg['num_key_value_heads'],hd,
            int(hd*rope.get('partial_rotary_factor',cfg.get('partial_rotary_factor',1.0))),
            rope.get('rope_theta',cfg.get('rope_theta',10000.0)),cfg.get('rms_norm_eps',1e-6),
            cfg.get('attn_output_gate',arch=='qwen35'),arch in ('qwen35','qwen3'),
            cfg.get('linear_num_value_heads',1),cfg.get('linear_num_key_heads',1),cfg.get('linear_key_head_dim',128)).validate()
        if self.spec.recurrent_layers and (cfg.get('linear_value_head_dim',128)!=128 or cfg.get('linear_conv_kernel_dim',4)!=4):
            raise ValueError('Recurrent operator requires 128-value heads and a four-tap optical convolution')
        self.tensors={};self.cfg=cfg;self.norm_offset=1.0 if arch=='qwen35' else 0.0
        for file in sorted(self.path.glob('*.safetensors')):
            with file.open('rb') as stream:
                size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
            for name,item in header.items():
                if name=='__metadata__':continue
                self.tensors[name]=(file,8+size,item)
        self.prefix='model.language_model.' if any(k.startswith('model.language_model.') for k in self.tensors) else 'model.'

    def actual(self,name):
        if name=='token_embd.weight': return self.prefix+'embed_tokens.weight'
        if name=='output_norm.weight': return self.prefix+'norm.weight'
        if name=='output.weight': return 'lm_head.weight' if 'lm_head.weight' in self.tensors else self.prefix+'embed_tokens.weight'
        _,layer,suffix=name.split('.',2);base=self.prefix+f'layers.{layer}.'
        mappings={'attn_norm.weight':'input_layernorm.weight','post_attention_norm.weight':'post_attention_layernorm.weight',
            'ffn_gate.weight':'mlp.gate_proj.weight','ffn_up.weight':'mlp.up_proj.weight','ffn_down.weight':'mlp.down_proj.weight',
            'attn_q.weight':'self_attn.q_proj.weight','attn_k.weight':'self_attn.k_proj.weight','attn_v.weight':'self_attn.v_proj.weight',
            'attn_output.weight':'self_attn.o_proj.weight','attn_q_norm.weight':'self_attn.q_norm.weight','attn_k_norm.weight':'self_attn.k_norm.weight',
            'attn_qkv.weight':'linear_attn.in_proj_qkv.weight','attn_gate.weight':'linear_attn.in_proj_z.weight',
            'ssm_out.weight':'linear_attn.out_proj.weight','ssm_conv1d.weight':'linear_attn.conv1d.weight',
            'ssm_beta.weight':'linear_attn.in_proj_b.weight','ssm_alpha.weight':'linear_attn.in_proj_a.weight',
            'ssm_dt.bias':'linear_attn.dt_bias','ssm_a':'linear_attn.A_log','ssm_norm.weight':'linear_attn.norm.weight'}
        return base+mappings[suffix]
    def has(self,name): return self.actual(name) in self.tensors
    def shape(self,name): return tuple(self.tensors[self.actual(name)][2]['shape'])
    def packed(self,name):
        file,offset,item=self.tensors[self.actual(name)];start,end=item['data_offsets']
        kind={'F32':(0,'F32'),'F16':(1,'F16'),'BF16':(30,'BF16')}
        if item['dtype'] not in kind: raise ValueError(f'{name}: coefficient format {item["dtype"]} needs a decoder')
        code,label=kind[item['dtype']]
        return np.memmap(file,dtype=np.uint8,mode='r',offset=offset+start,shape=(end-start,)),code,label
    def floating(self,name):
        raw,code,_=self.packed(name)
        value=(raw.view('<u2').astype('<u4')<<16).view('<f4') if code==30 else raw.view('<f2' if code==1 else '<f4').astype('<f4')
        return value.reshape(self.shape(name))
    def norm(self,name): return self.floating(name)+self.norm_offset


def open_source(path):
    path=Path(path)
    return SafetensorsSource(path) if path.is_dir() else GGUFSource(path)
