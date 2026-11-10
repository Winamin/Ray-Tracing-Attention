"""Audit small real GPU histories against independently decoded duplicate counts."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
from unittest.mock import patch

import numpy as np

from ..compiler.spec import asset_spec
from ..runtime.prompt_encoder import LocalPromptEncoder
from ..runtime.resident_runtime import ResidentRuntime
from ..runtime.settings import LOCAL, default_model
from .verify_attention_receipts import decode_history, full_attention, error_metrics

ROOT=Path(__file__).resolve().parents[2]
BUILDER=importlib.import_module('ray_tracing.compiler.build')

CAPTURE=r'''
    struct SmallCapture {ResidentConfig cfg;Buffer* saved;};
    std::vector<SmallCapture> smallCaptures;
    void smallCapture(VkCommandBuffer c,ResidentConfig cfg){
        size_t qBytes=size_t(cfg.rows)*MODEL_A*4u,slotBytes=capacity*8u,historyBytes=size_t(capacity)*HISTORY_BYTES;
        auto* saved=e.buffer(qBytes+slotBytes+historyBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);
        sync(c);
        VkBufferCopy copies[3]{{0u,0u,qBytes},{size_t(fullIndex(cfg.layer))*slotBytes,qBytes,slotBytes},
            {size_t(fullIndex(cfg.layer))*historyBytes,qBytes+slotBytes,historyBytes}};
        vkCmdCopyBuffer(c,query->handle,saved->handle,1u,copies);
        vkCmdCopyBuffer(c,slots->handle,saved->handle,1u,copies+1u);
        vkCmdCopyBuffer(c,keyGeometry->handle,saved->handle,1u,copies+2u);
        Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        sync(c);smallCaptures.push_back({cfg,saved});
    }
    void flushSmallCaptures(){
        if(smallCaptures.empty())return;
        std::ofstream out(std::getenv("RT_SMALL_CAPTURE"),std::ios::binary|std::ios::app);
        for(auto& v:smallCaptures){
            uint32_t meta[12]{v.cfg.rows,v.cfg.layer,v.cfg.position,v.cfg.mode,capacity,MODEL_A,MODEL_Q,
                RT_ATTENTION_HEADS,RT_ATTENTION_HEAD_DIM,MODEL_GQA,HISTORY_BYTES,fullIndex(v.cfg.layer)};
            out.write(reinterpret_cast<char*>(meta),sizeof(meta));
            out.write(static_cast<char*>(v.saved->map),v.saved->size);e.release(v.saved);
        }
        smallCaptures.clear();
    }
'''


def probe(base,target):
    target.mkdir(parents=True,exist_ok=True)
    for p in [*base.parent.glob('*.spv'),base.parent/'rt_model_config.h',base.parent/'rt_layer_map.hpp']:
        shutil.copyfile(p,target/p.name)
    source=(ROOT/'ray_tracing/native/src/optical_resident.cpp').read_text(encoding='utf8')
    anchor='    void layer(VkCommandBuffer c,ResidentConfig cfg) {'
    source=source.replace(anchor,CAPTURE+'\n'+anchor)
    anchor='mergeAttentionReceipts(c,cfg);mark(c,cfg,5);'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'smallCapture(c,cfg);')
    anchor='void reply(const std::vector<char>& bytes){uint32_t header'
    assert source.count(anchor)==1
    source=source.replace(anchor,'void reply(const std::vector<char>& bytes){flushSmallCaptures();uint32_t header')
    path=target/'optical_resident.cpp';path.write_text(source,encoding='utf8')
    worker=target/'rt_optical_resident.exe'
    command=[os.environ.get('CXX','g++'),'-std=c++17','-O3','-ffp-contract=off',
             '-I'+str(target),'-I'+str(ROOT/'ray_tracing/native/src'),'-I'+str(ROOT/'ray_tracing/native/include'),str(path),
             '-static-libgcc','-static-libstdc++','-Wl,-Bstatic,--whole-archive','-lwinpthread',
             '-Wl,--no-whole-archive,-Bdynamic','-lvulkan-1','-o',str(worker)]
    subprocess.run(command,check=True,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return worker


def capture_real(asset,base,target):
    encoder=LocalPromptEncoder(asset)
    ids=encoder.encode([dict(role='user',content='分析当前项目的 Attention 和因果缓存。')])[:16]
    path=target/'histories.bin';worker=probe(base,target);path.write_bytes(b'')
    with patch.object(BUILDER,'build',return_value=worker),patch.dict(os.environ,RT_SMALL_CAPTURE=str(path)):
        with ResidentRuntime(asset/'optical_model.bin',capacity=32,cache=False,prefill_batch=32,
                             attention_backend='grouped_rt',mlp_backend='rt_transmission') as runtime:
            runtime.prefill(ids,())
    records=[]
    with path.open('rb') as stream:
        while header:=stream.read(48):
            rows,layer,position,mode,capacity,a,qwidth,heads,dh,kv,hist,full=struct.unpack('<12I',header)
            q=np.frombuffer(stream.read(rows*a*4),'<f4').reshape(rows,a)[:,:qwidth].reshape(rows,heads,dh)
            slots=np.frombuffer(stream.read(capacity*8),'<u4').reshape(capacity,2)
            packets=np.frombuffer(stream.read(capacity*hist),'u1').reshape(capacity,hist)
            records.append(dict(layer=layer,q=q.copy(),packets=packets[:len(ids)].copy()))
    return records,ids


def column_map(values):
    values=np.asarray(values,dtype='f4').copy()
    values[values==0]=0.0
    bits=values.view('u4')
    groups={};mapping=[]
    for axis in range(values.shape[-1]):
        key=bits[:,axis].tobytes();groups.setdefault(key,axis);mapping.append(groups[key])
    return np.asarray(mapping)


def check(runtime,q,k,v,label):
    result=runtime.attention(q,k,v,layer=runtime.model_spec.full_layers[0])
    keys,values=decode_history(result['history_packets'],runtime.model_spec)
    raw=result['compact_receipts'];offset=16+result['block']//32
    dim=runtime.model_spec.attention_head_dim;axis=np.arange(dim,dtype=np.uint32)
    maps=(raw[:,offset+axis//4]>>((axis%4)*8))&255
    reports=[]
    per=runtime.model_spec.attention_heads//runtime.model_spec.kv_heads
    for h in range(runtime.model_spec.attention_heads):
        head=h//per;km=column_map(keys[:,head]);vm=column_map(values[:,head])
        gpu_k=int(raw[h,8+result['block']//32]);gpu_v=int(raw[h,9+result['block']//32])
        wanted_k=int(np.unique(km).size);wanted_v=int(np.unique(vm).size)
        # Q=0 suppresses every K decode, independently of column equivalence.
        active=bool(np.any(q[0,h]!=0))
        reports.append(dict(head=h,cpu_key_classes=wanted_k,cpu_value_classes=wanted_v,
                            gpu_key_classes=gpu_k,gpu_value_classes=gpu_v,
                            missed_key_classes=max(0,gpu_k-wanted_k) if active else 0,
                            missed_value_classes=max(0,gpu_v-wanted_v),
                            value_map_matches=bool(np.array_equal(maps[h],vm)),
                            equal_values_per_single_row=[dim-np.unique(a).size for a in values[:,head]]))
    reference=full_attention(q,keys,values,np.zeros_like(q),len(k)-1,0,runtime.capacity,gated=False)
    # Default diagnostic gate is zero, so gated models multiply the result by 0.5.
    expected=reference['output']*(.5 if runtime.model_spec.attention_gate else 1.)
    return dict(name=label,history=len(k),heads=reports,shader_work=result['shader_work'],
                full_packet_classes=int(np.unique(result['history_packets'],axis=0).shape[0]),
                output_check=error_metrics(result['output'],expected),gpu_ms=result['gpu_ms'])


def audit(asset,output):
    asset=asset.resolve();asset=asset.parent if asset.is_file() else asset
    spec=asset_spec(asset);base=BUILDER.build(spec)
    target=LOCAL/'diagnostics/attention_duplicate_audit'
    actual,ids=capture_real(asset,base,target)
    report=dict(status='running',actual_input_ids=ids,source='actual GPU post-RoPE Q and packed K/V',
                attention_backend='grouped_rt',mlp_backend='rt_transmission',
                worker_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),checks=[])
    def save():
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding='utf8')
    with ResidentRuntime(asset/'optical_model.bin',capacity=32,cache=False,prefill_batch=32,
                         attention_backend='grouped_rt',mlp_backend='rt_transmission') as runtime:
        for item in (actual[0],actual[-1]):
            keys,values=decode_history(item['packets'],spec)
            for n in (1,2,4,8,16):
                q=item['q'][n-1:n]
                r=check(runtime,q,keys[:n],values[:n],f"real_layer_{item['layer']}_n{n}")
                report['checks'].append(r);save()
                print(json.dumps(dict(name=r['name'],cpu_v=[h['cpu_value_classes'] for h in r['heads'][::4]],
                                      gpu_v=[h['gpu_value_classes'] for h in r['heads'][::4]])),flush=True)
        rng=np.random.default_rng(20261110)
        for n in (1,2,4):
            q=rng.normal(0,.25,(1,spec.attention_heads,spec.attention_head_dim)).astype('f4')
            k=rng.normal(0,.25,(n,spec.kv_heads,spec.attention_head_dim)).astype('f4')
            v=rng.normal(0,.25,k.shape).astype('f4')
            for name,kk,vv in [('random',k,v),('constant',np.full_like(k,.25),np.full_like(v,.5)),('zeros',np.zeros_like(k),np.zeros_like(v))]:
                report['checks'].append(check(runtime,q,kk,vv,f'{name}_n{n}'));save()
    report.update(status='complete',missed_key_classes=sum(h['missed_key_classes'] for r in report['checks'] for h in r['heads']),
                  missed_value_classes=sum(h['missed_value_classes'] for r in report['checks'] for h in r['heads']))
    report['passed']=(report['missed_key_classes']==0 and report['missed_value_classes']==0
                      and all(h['value_map_matches'] for r in report['checks'] for h in r['heads'])
                      and all(r['output_check']['passed'] for r in report['checks']))
    save()
    assert report['passed'], 'Column classification or full attention mismatch'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',type=Path,default=default_model())
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();audit(args.asset,args.output)


if __name__=='__main__':main()
