"""Translate original compressed coefficients into versioned RT geometry assets."""
from pathlib import Path
import hashlib
import json
import shutil
import struct
import numpy as np
from .sources import GGUFSource, open_source

STRIDE=64
FORMATS={0:4,1:2,30:2,12:144/256,14:210/256}

def tokenizer(source,destination):
    if not isinstance(source,GGUFSource):
        for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','generation_config.json'):
            p=source.path/name
            if p.exists():shutil.copy2(p,destination/name)
        if not (destination/'tokenizer.json').exists(): raise ValueError('A tokenizer.json is required in the source directory')
        if not (destination/'chat_template.jinja').exists():
            config_path=source.path/'tokenizer_config.json'
            config=json.loads(config_path.read_text(encoding='utf8')) if config_path.exists() else {}
            template=config.get('chat_template')
            if isinstance(template,list):template=next((v['template'] for v in template if v['name']=='default'),template[0]['template'])
            if template:(destination/'chat_template.jinja').write_text(template,encoding='utf8')
            else:(destination/'chat_template.jinja').write_text("{% for message in messages %}{{ message['content'] }}\n{% endfor %}",encoding='utf8')
        if not (destination/'generation_config.json').exists():
            (destination/'generation_config.json').write_text(json.dumps({'eos_token_id':source.cfg.get('eos_token_id'),'bos_token_id':source.cfg.get('bos_token_id')}),encoding='utf8')
        return
    from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers, AddedToken
    tokens=source.field('tokenizer.ggml.tokens');merges=source.field('tokenizer.ggml.merges')
    if source.field('tokenizer.ggml.model')!='gpt2' or merges is None:
        raise ValueError('This tokenizer adapter supports byte-level BPE GGUF tokenizers')
    pre=source.field('tokenizer.ggml.pre','')
    patterns={'qwen2':r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
              'qwen35':r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
              'llama-bpe':r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"}
    if pre not in patterns: raise ValueError(f'Tokenizer preprocessor {pre!r} needs its exact split adapter; use safetensors with tokenizer.json')
    t=Tokenizer(models.BPE(vocab={v:i for i,v in enumerate(tokens)},merges=[tuple(m.split(' ',1)) for m in merges],byte_fallback=False))
    t.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(patterns[pre]),behavior='isolated'),pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
    t.decoder=decoders.ByteLevel()
    kinds=source.field('tokenizer.ggml.token_type',[1]*len(tokens))
    t.add_special_tokens([AddedToken(v,normalized=False,special=True) for v,k in zip(tokens,kinds) if k in (3,4,5)])
    if any(t.token_to_id(v)!=i for i,v in enumerate(tokens)):raise ValueError('Tokenizer IDs changed during translation')
    t.save(str(destination/'tokenizer.json'))
    template=source.field('tokenizer.chat_template')
    if template:(destination/'chat_template.jinja').write_text(template,encoding='utf8')
    generation={'eos_token_id':source.field('tokenizer.ggml.eos_token_id'),'bos_token_id':source.field('tokenizer.ggml.bos_token_id')}
    (destination/'generation_config.json').write_text(json.dumps(generation),encoding='utf8')

def export(source,destination):
    src=open_source(source) if isinstance(source,(str,Path)) else source
    spec=src.spec.validate();d=spec.hidden_size;depth=spec.depth;middle=spec.middle_size
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    # Validate the complete graph before opening any output coefficient stream.
    expected={'token_embd.weight':(spec.vocabulary_size,d),'output.weight':(spec.vocabulary_size,d)}
    for l,kind in enumerate(spec.layer_types):
        p=f'blk.{l}.'
        expected.update({p+'ffn_gate.weight':(middle,d),p+'ffn_up.weight':(middle,d),p+'ffn_down.weight':(d,middle)})
        if kind=='full':
            expected.update({p+'attn_q.weight':(spec.query_width*(2 if spec.attention_gate else 1),d),p+'attn_k.weight':(spec.kv_width,d),p+'attn_v.weight':(spec.kv_width,d),p+'attn_output.weight':(d,spec.query_width)})
        else:
            expected.update({p+'attn_qkv.weight':(spec.state_key_heads*256+spec.state_width,d),p+'attn_gate.weight':(spec.state_width,d),p+'ssm_out.weight':(d,spec.state_width)})
    for name,shape in expected.items():
        if src.shape(name)!=shape: raise ValueError(f'{name}: expected {shape}, got {src.shape(name)}')
        if src.packed(name)[1] not in FORMATS: raise ValueError(f'{name}: coefficient decoder required for this quantization')
    tokenizer(src,destination)
    header=np.zeros(32+depth*STRIDE,np.uint32);header[:4]=[depth,depth,middle,4];header[11:13]=[1,1]
    parts=[];records=[];count=len(header)
    def add(value):
        nonlocal count
        v=np.ascontiguousarray(value,dtype='<f4').ravel().view('<u4');start=count;parts.append(v);count+=len(v);return start
    header[4]=add(src.norm('output_norm.weight'));header[7]=add(np.zeros(8))
    matrix_words=0
    with (destination/'geometry_weights.bin').open('wb') as stream:
        for layer,kind in enumerate(spec.layer_types):
            base=32+layer*STRIDE;prefix=f'blk.{layer}.';start=matrix_words;written={}
            def packed(name):
                nonlocal matrix_words
                if name in written:return written[name]
                raw,code,label=src.packed(prefix+name)
                at=matrix_words-start;raw.tofile(stream);pad=(-raw.nbytes)%4;stream.write(bytes(pad));matrix_words+=(raw.nbytes+pad)//4
                written[name]=(at,code)
                sample=np.concatenate((raw[:4096],raw[-4096:]))
                records.append({'name':prefix+name,'type':label,'shape':list(src.shape(prefix+name)),'bytes':raw.nbytes,'layer_offset_words':at,'sample_sha256':hashlib.sha256(sample.tobytes()).hexdigest()})
                return written[name]
            def descriptor(index,name,mode=0,offset_rows=0):
                at,code=packed(name);row_bytes=int(d*FORMATS[code])
                if offset_rows*row_bytes%4:raise ValueError('Projection offset needs word alignment')
                offset=base+32 if index==8 else base+9+index*2
                header[offset]=at+offset_rows*row_bytes//4;header[offset+1]=code|(mode<<8)
            header[base]=add(src.norm(prefix+'attn_norm.weight'));header[base+1]=add(src.norm(prefix+'post_attention_norm.weight'))
            header[base+5]=add(np.zeros(8))
            if kind=='full':
                interleaved=isinstance(src,GGUFSource) and spec.architecture=='llama'
                descriptor(0,'attn_q.weight',0 if interleaved else 1 if spec.attention_gate else 3)
                descriptor(1,'attn_k.weight',0 if interleaved else 3);descriptor(8,'attn_v.weight',3);descriptor(2,'attn_output.weight',4)
                if spec.attention_gate:descriptor(6,'attn_q.weight',2)
                perm=np.arange(spec.attention_head_dim);r=spec.rotary_dim
                perm[:r]=np.stack((np.arange(r//2),np.arange(r//2)+r//2),-1).ravel()
                header[base+2]=add(np.zeros(spec.query_width//2))
                for name,field in (('attn_q_norm.weight',24),('attn_k_norm.weight',25)):
                    norm=src.norm(prefix+name)[perm] if spec.qk_norm else np.ones(spec.attention_head_dim)
                    header[base+field]=add(np.tile(norm,spec.attention_heads))
                header[base+23]=spec.attention_heads;header[base+29]=spec.rotary_dim
            else:
                key=spec.state_key_heads*128;s=spec.state_width
                descriptor(0,'attn_qkv.weight',5);descriptor(1,'attn_qkv.weight',5,key);descriptor(8,'attn_qkv.weight',0,2*key)
                descriptor(2,'ssm_out.weight');descriptor(6,'attn_gate.weight')
                conv=src.floating(prefix+'ssm_conv1d.weight').reshape(2*key+s,4)
                repeat=spec.state_heads//spec.state_key_heads
                header[base+34]=add(np.concatenate((np.tile(conv[:key],(repeat,1)),np.tile(conv[key:2*key],(repeat,1)),conv[2*key:]),0))
                header[base+35],header[base+39]=packed('ssm_beta.weight');header[base+36],header[base+40]=packed('ssm_alpha.weight')
                header[base+37]=add(src.floating(prefix+'ssm_dt.bias'))
                a=src.floating(prefix+'ssm_a')
                if isinstance(src,GGUFSource):
                    if not np.all(a<0):raise ValueError('GGUF transition coefficients must be negative')
                    a=np.log(-a)
                header[base+38]=add(a);header[base+24]=add(np.ones(s));header[base+25]=add(np.ones(s))
                header[base+31]=add(np.tile(src.floating(prefix+'ssm_norm.weight'),spec.state_heads));header[base+23]=spec.state_heads
            descriptor(3,'ffn_gate.weight');descriptor(4,'ffn_down.weight');descriptor(5,'ffn_up.weight')
            header[base+26]=header[base+30]=1 if kind=='full' else 2
            header[base+27]=start;header[base+28]=matrix_words-start
            print(json.dumps({'exported_layer':layer+1,'layers':depth}),flush=True)
    basis_words=0
    basis_written={}
    with (destination/'vocabulary.bin').open('wb') as stream:
        stream.write(b'RTBASQ01');stream.write(struct.pack('<2I',spec.vocabulary_size,d))
        for name,kind,offset in (('token_embd.weight',13,14),('output.weight',15,16)):
            raw,code,label=src.packed(name);identity=src.actual(name)
            header[kind]=code;header[offset]=basis_written.get(identity,basis_words)
            if identity not in basis_written:
                basis_written[identity]=basis_words
                raw.tofile(stream);pad=(-raw.nbytes)%4;stream.write(bytes(pad));basis_words+=(raw.nbytes+pad)//4
            records.append({'name':name,'type':label,'bytes':raw.nbytes,'shape':list(src.shape(name)),'basis_offset_words':int(header[offset])})
    params=np.concatenate((header,*parts));asset=destination/'optical_model.bin'
    with asset.open('wb') as stream:
        stream.write(b'RTRES004');stream.write(struct.pack('<9I',d,depth,depth,middle,spec.vocabulary_size,4,len(params),matrix_words,0));params.tofile(stream)
    metadata={'source_model':str(src.path),'architecture':spec.architecture,'model_spec':spec.to_dict(),
        'depth':depth,'hidden_size':d,'middle':middle,'vocabulary':spec.vocabulary_size,
        'precision':'original coefficient formats','quantization':' + '.join(sorted({r['type'] for r in records})),
        'full_contribution_layers':len(spec.full_layers),'optical_recurrent_layers':len(spec.recurrent_layers),
        'state_heads':spec.state_heads,'step':0,'parameter_bytes':int(params.nbytes+(matrix_words+basis_words)*4),
        'history_quantization':{'address':'Q4_K','content':'Q6_K','bytes_per_token':len(spec.full_layers)*spec.history_bytes},
        'tensor_payloads':records,'tokenizer_ids_verified':spec.vocabulary_size}
    (destination/'model.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    return asset,metadata
