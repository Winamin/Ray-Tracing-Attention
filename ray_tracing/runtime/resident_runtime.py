"""One resident Vulkan worker with batch replies and native per-token streaming."""
from pathlib import Path
import os
import json
import struct
import subprocess
import tempfile
import numpy as np

class ResidentRuntime:
    def __init__(self,asset,capacity=300,cache=True,routing="rt_full",aperture=.08,word_tile=32,feature_tile=8,readout_unroll=1,readout_words=4,prefill_batch=512,matrix_rows=8,attention_backend="standard",attention_block=128,attention_stats=False,mlp_backend="standard",attention_v_members=False):
        asset=Path(asset).resolve()
        from ..compiler.partition import ensure_partition
        self.partition=ensure_partition(asset.parent)
        config=json.loads((asset.parent/'model.json').read_text(encoding='utf8'))
        from ..compiler.spec import asset_spec
        from ..compiler.build import build
        spec=asset_spec(asset.parent)
        self.model_spec=spec
        if attention_backend not in ('standard','standard_fp32','grouped_rt','grouped_rt_dense','grouped_rt_local','grouped_rt_tc'):
            raise ValueError('Attention backend must be standard, standard_fp32, grouped_rt, grouped_rt_dense, grouped_rt_local or grouped_rt_tc')
        if attention_backend=='grouped_rt_local' and spec.attention_head_dim!=256:
            raise ValueError('grouped_rt_local requires a 256-dimensional attention head')
        if mlp_backend not in ('standard','rt_transmission','rt_material','rt_tensor','rt_bfloat'):
            raise ValueError('MLP backend must be standard, rt_transmission, rt_material, rt_tensor or rt_bfloat')
        if attention_block not in (128,256,512):raise ValueError('Attention block must be 128, 256 or 512')
        self.hidden_size=spec.hidden_size
        self.vocabulary_size=spec.vocabulary_size
        self.growable=True
        if not 1<=capacity<=1048576:raise ValueError("Invalid initial tree capacity")
        capacity=max(32,capacity)
        if prefill_batch not in (32,64,128,256,512,1024):raise ValueError('Prefill batch must be 32, 64, 128, 256, 512 or 1024')
        self.prefill_batch=prefill_batch
        if matrix_rows not in (4,8,16):raise ValueError('Matrix row tile must be 4, 8 or 16')
        if routing!='rt_full':raise ValueError('The resident runtime uses full-contribution RT routing')
        self.capacity=capacity;self.stderr=tempfile.TemporaryFile()
        worker=build(spec)
        native=worker.parent
        options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        options['env']=dict(os.environ,RT_ATTENTION_BACKEND=attention_backend,RT_ATTENTION_BLOCK=str(attention_block),RT_ATTENTION_STATS=str(int(attention_stats)),RT_MLP_BACKEND=mlp_backend,RT_ATTENTION_V_MEMBERS=str(int(bool(attention_v_members))))
        self.process=subprocess.Popen([str(worker),str(Path(asset).resolve()),str(Path(asset).resolve().parent/'vocabulary.bin'),str(capacity),str(int(cache)),routing,str(aperture),str(word_tile),str(feature_tile),str(readout_unroll),str(readout_words),str(prefill_batch),str(matrix_rows)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,cwd=native,**options)
        try:
            magic,depth,blocks,middle,mode=struct.unpack('<5I',self.read(20))
            if magic not in (0x52545239,0x5254523a):raise RuntimeError('Resident optical protocol mismatch')
            if attention_backend in ('grouped_rt','grouped_rt_dense','grouped_rt_local','grouped_rt_tc') and not mode&512:raise RuntimeError('Worker did not activate grouped RT attention')
            if attention_backend=='standard_fp32' and not mode&1024:raise RuntimeError('Worker did not activate FP32 attention')
            if (mlp_backend=='standard')!=bool(mode&4096):raise RuntimeError('Worker did not activate the requested MLP backend')
        except BaseException:
            self.close()
            raise
        self.metadata={'depth':depth,'blocks':blocks,'middle':middle,'hidden_size':self.hidden_size,'quantization':config.get('precision','bf16 optical'),'context_window':capacity,'growable_tree':self.growable,'state_cache':bool(cache),'routing':routing,'route_aperture':2.0 if routing=='rt_full' else aperture,'word_tile':word_tile,'feature_tile':feature_tile,'readout_unroll':readout_unroll,'readout_words':readout_words}
        self.metadata.update(native_token_stream=magic==0x5254523a,rt_query_telemetry=magic==0x5254523a,worker=str(worker))
        self.metadata['prefill_batch']=prefill_batch
        self.metadata['matrix_rows']=matrix_rows
        self.metadata.update(attention_backend=attention_backend,attention_block=attention_block,
                             attention_compute='fused RT / FP32 grouped softmax' if mode&512 else 'original FP32' if mode&1024 else 'original',
                             attention_rt_timing='includes score and weighted aggregation' if mode&512 else 'RT traversal only')
        self.metadata['attention_receipt_format']='compact_columns' if mode&2048 else 'dense'
        self.metadata['attention_stats']=bool(attention_stats)
        if mode&2048:self.metadata['attention_compute']='RT hit classification / compact FP32 grouped softmax'
        if attention_backend=='grouped_rt_local':
            self.metadata['attention_compute']='RT local K/V classes / FP32 QK, cached weights, '+('full V members' if attention_v_members else 'representative V')
            self.metadata['attention_receipt_format']='score_weight_cache'
            self.metadata['attention_value_policy']='rt_member_distribution' if attention_v_members else 'rt_representatives_only'
            self.metadata['attention_v_members']=bool(attention_v_members)
            self.metadata['attention_full_contribution']=bool(attention_v_members)
        if attention_backend=='grouped_rt_tc':self.metadata['attention_compute']='P1 f16 QK MMA / FP32 accumulation, softmax and V'
        self.metadata['prefill_compute']='cooperative_matrix2' if mode&256 else 'subgroup'
        self.metadata['matrix_channel_tile']=min(64,spec.attention_head_dim) if mode&256 else 4
        self.metadata['matrix_token_tile']=256 if mode&256 else matrix_rows
        self.metadata['reply_cache_kind']='verified optical state path'
        self.metadata['cache_verification_bits']=256
        self.metadata['decode_route_cache']=False
        self.metadata['mlp_backend']=mlp_backend
        self.metadata['mlp_compute']='full_matrix' if mode&4096 else 'rt_core_balanced_transmission'
        self.metadata['mlp_activation']='swiglu' if mode&4096 else 'sigmoid_transmission'
        self.metadata['mlp_prefill_compute']=('cooperative_matrix2' if mode&256 else 'subgroup') if mode&4096 else 'rt_transmission'
        self.metadata['mlp_decode_compute']='subgroup' if mode&4096 else 'rt_transmission'
        self.metadata['mlp_token_tile']=256 if mode&4096 and mode&256 else matrix_rows
        if not mode&4096:self.metadata['mlp_material_receipts']='shared within token tile'
        self.metadata['mlp_backend_effective']=('rt_bfloat' if mode&32768 else 'rt_tensor' if mode&16384 else 'rt_material' if mode&8192 else
                                               'standard' if mode&4096 else 'rt_transmission')
        if mode&8192:
            self.metadata['mlp_compute']='RT weight-block receipts / complete FP32 material classes'
            self.metadata['mlp_prefill_compute']=self.metadata['mlp_decode_compute']='subgroup_partitioned_fp32'
        if mode&16384:
            self.metadata['mlp_compute']='RT weight-block addresses / '+('BF16' if mode&32768 else 'scaled FP16')+' operands with FP32 MMA accumulation'
            self.metadata['mlp_prefill_compute']='cooperative_matrix2 for rows >= 17; RT transmission for smaller batches'
            self.metadata['mlp_decode_compute']='rt_transmission'
            self.metadata['mlp_token_tile']=128
            self.metadata['mlp_tensor_min_rows']=17
            self.metadata['mlp_tensor_operands']='bf16' if mode&32768 else 'fp16'
            self.metadata['mlp_tensor_accumulator']='fp32'
            self.metadata['mlp_tensor_bfloat16']=bool(mode&32768)
            self.metadata['mlp_material_receipts']='direct block-address slots; no sorting pass'
        self.metadata['source_model']=config.get('source_model')
        self.metadata['coordinate_partition']=config['coordinate_partition']
        self.position=0;self._attention_diagnostic_dirty=False
        self.growth_events=[]

    def ensure_capacity(self,required):
        if not self.growable or required<=self.capacity:return
        if required>1048576:raise ValueError('History exceeds this worker’s address range')
        next_capacity=self.capacity
        while next_capacity<required:next_capacity=min(1048576,next_capacity*2)
        if next_capacity>1048576:raise ValueError('History exceeds this worker’s address range')
        data=np.frombuffer(self.request(10,next_capacity),'<f8')
        self.capacity=int(data[1]);self.metadata['context_window']=self.capacity
        self.growth_events.append({'previous':int(data[0]),'capacity':self.capacity,'wall_ms':float(data[2])})

    def read(self,count):
        parts=[]
        while count:
            part=self.process.stdout.read(count)
            if not part:
                self.stderr.seek(0);message=self.stderr.read().decode('utf8',errors='replace');raise RuntimeError(message or f'Resident worker exited: {self.process.wait()}')
            parts.append(part);count-=len(part)
        return b''.join(parts)

    def request(self,op,count,debug=False,body=b''):
        self.process.stdin.write(struct.pack('<3I',op,count,int(debug)));self.process.stdin.write(body);self.process.stdin.flush()
        status,size=struct.unpack('<2I',self.read(8));payload=self.read(size)
        if status:raise RuntimeError('Resident request failed')
        return payload

    def prefill(self,ids,stops=(),debug=False,profile=False,checkpoint_tokens=None,checkpoint_boundaries=None):
        if debug and profile:raise ValueError('Use either layer capture or prefill timing')
        if checkpoint_tokens is not None and not 0<checkpoint_tokens<=len(ids):
            raise ValueError('Checkpoint must end inside the input prefix')
        if checkpoint_boundaries is not None:
            if checkpoint_tokens is not None:
                raise ValueError('Use one checkpoint representation')
            if len(checkpoint_boundaries)>4096 or any(not 0<p<=len(ids) or not 0<kind<=15
                                                    for p,kind in checkpoint_boundaries):
                raise ValueError('Checkpoint boundaries must end inside the input prefix')
        self.ensure_capacity(len(ids))
        ids=np.asarray(ids,dtype='<u4')[-self.capacity:];stop=list(stops)[:2]+[0xffffffff]*2
        body=ids.tobytes()+struct.pack('<2I',*stop[:2])
        if checkpoint_tokens is not None:body+=struct.pack('<I',checkpoint_tokens)
        if checkpoint_boundaries is not None:
            body+=struct.pack('<I',len(checkpoint_boundaries))
            body+=b''.join(struct.pack('<2I',p,kind) for p,kind in checkpoint_boundaries)
        operation=12 if checkpoint_boundaries is not None else 11 if checkpoint_tokens is not None else 1
        payload=self.request(operation,len(ids),2 if profile else int(debug),body);time=np.frombuffer(payload,'<f8',count=6).copy();control=np.frombuffer(payload,'<u4',count=8,offset=48).copy()
        result={'setup_gpu_ms':float(time[0]),'prefill_gpu_ms':float(time[1]),'prefill_submit_wait_ms':float(time[2]),'prefill_wall_ms':float(time[3]),'prefill_cache_hit':bool(time[4]),'cached_prefix_tokens':int(time[5]),'control':control}
        if debug:
            result['state']=np.frombuffer(payload,'<f4',count=len(ids)*self.hidden_size,offset=80).copy().reshape(len(ids),self.hidden_size)
            result['layers']=np.frombuffer(payload,'<f4',count=self.metadata['depth']*len(ids)*self.hidden_size,offset=80+len(ids)*self.hidden_size*4).copy().reshape(self.metadata['depth'],len(ids),self.hidden_size)
        footer=80+(len(ids)*self.hidden_size*(self.metadata['depth']+1)*4 if debug else 0)
        if len(payload)>=footer+32:
            result.update(zip(('prefill_batch_size','prefill_batches','prefill_tree_updates','prefill_query_positions'),map(int,np.frombuffer(payload,'<u8',count=4,offset=footer))))
        if profile and len(payload)>=footer+80:
            result['profile_gpu_ms']=dict(zip(('projection_geometry','bvh_update','lens_history_trace','coordinate_restore','detector','output_mlp'),map(float,np.frombuffer(payload,'<f8',count=6,offset=footer+32))))
        optical=footer+(80 if profile else 32)
        if len(payload)>=optical+16:result.update(zip(('prefill_feature_rt_rows','prefill_feature_reused_rows'),map(int,np.frombuffer(payload,'<u8',count=2,offset=optical))))
        if len(payload)>=optical+32:result.update(zip(('stable_prefix_tokens','prefill_snapshot_kind'),map(int,np.frombuffer(payload,'<u8',count=2,offset=optical+16))))
        if len(payload)>=optical+80:
            result.update(zip(('history_shared_blocks','history_saved_boundaries','history_cache_boundaries',
                               'history_lookup_rays','prefill_computed_tokens','history_cache_evictions'),
                              map(int,np.frombuffer(payload,'<u8',count=6,offset=optical+32))))
        if len(payload)>=optical+88:result['history_lookup_wall_ms']=float(np.frombuffer(payload,'<f8',count=1,offset=optical+80)[0])
        if len(payload)>=optical+104:
            result.update(zip(('history_state_rebuilt_tokens','history_address_hit'),
                              map(int,np.frombuffer(payload,'<u8',count=2,offset=optical+88))))
        self.position=len(ids);self._attention_diagnostic_dirty=False
        return result

    def finish_history(self):
        payload=self.request(13,0)
        result=dict(zip(('history_output_saved','history_output_tokens','history_cache_boundaries'),
                        map(int,np.frombuffer(payload,'<u8',count=3))))
        result['history_output_capture_wall_ms']=float(np.frombuffer(payload,'<f8',count=1,offset=24)[0])
        return result

    def generate(self,count=32,debug=False,profile=False,telemetry=False):
        if self._attention_diagnostic_dirty:raise RuntimeError('Attention diagnostics replace history; prefill again before generation')
        self.ensure_capacity(self.position+count)
        mode=2 if profile else 1 if debug else 3 if telemetry and self.metadata['rt_query_telemetry'] else 0
        return self._generation_result(self.request(2,count,mode),debug,profile,telemetry)

    def stream_generate(self,count=32,profile=False,telemetry=True):
        if self._attention_diagnostic_dirty:raise RuntimeError('Attention diagnostics replace history; prefill again before generation')
        if not 1<=count<=64:raise ValueError('Stream count must be 1..64')
        self.ensure_capacity(self.position+count)
        if not self.metadata['native_token_stream']:
            for _ in range(count):
                result=self.generate(1,profile=profile,telemetry=telemetry);yield result
                if not result['control'][2]:break
            return
        self.process.stdin.write(struct.pack('<3I',9,count,2 if profile else 3 if telemetry else 0));self.process.stdin.flush()
        for _ in range(count):
            status,size=struct.unpack('<2I',self.read(8));payload=self.read(size)
            if status:raise RuntimeError('Resident stream failed')
            result=self._generation_result(payload,False,profile,telemetry);yield result
            if not result['control'][2]:break

    def _generation_result(self,payload,debug=False,profile=False,telemetry=False):
        time=np.frombuffer(payload,'<f8',count=18).copy();control=np.frombuffer(payload,'<u4',count=8,offset=144).copy();ids=np.frombuffer(payload,'<u4',count=64,offset=176).copy()[:int(control[1])]
        result={'ids':ids,'gpu_ms':float(time[0]),'submit_wait_ms':float(time[1]),'gpu_tokens_per_second':len(ids)*1000/max(float(time[0]),1e-20),
                'wall_tokens_per_second':len(ids)*1000/max(float(time[1]),1e-20),'generation_submissions':int(time[2]),'generation_waits':int(time[3]),
                'intermediate_host_bytes':int(time[4]),'reply_cache_hit':bool(time[5]),'absolute_position':int(time[6]),'context_coordinate_decoded_pairs':int(time[7]),'canonical_initial_decode_rays':65536,'control':control}
        if profile:result['profile_gpu_ms']=dict(zip(('projection_geometry','bvh_update','lens_history_trace','coordinate_restore','detector','output_mlp','vocabulary','vocabulary_rt','vocabulary_coefficients','vocabulary_reduce'),map(float,time[8:])))
        hits=np.frombuffer(payload,'<u4',count=5,offset=144+32+256)
        result.update(mlp_feature_hits=int(hits[0]),mlp_possible_features=int(hits[1]),vocabulary_hits=int(hits[2]),tree_cache_queries=int(result['reply_cache_hit']),tree_cache_hits=int(hits[3]),tree_cache_leaf=int(hits[4]) if result['reply_cache_hit'] else None)
        self.position=result['absolute_position']
        if debug:result['state']=np.frombuffer(payload,'<f4',count=self.hidden_size,offset=144+32+256+20).copy()
        offset=144+32+256+20+(self.hidden_size*4 if debug else 0)
        if len(payload)>=offset+32 and (telemetry or profile):
            result['rt_query_kind_gpu_ms']=dict(zip(('history','vocabulary','mlp','cache'),map(float,np.frombuffer(payload,'<f8',count=4,offset=offset))))
            result['rt_query_gpu_ms']=sum(result['rt_query_kind_gpu_ms'].values())
        if len(payload)>=offset+64:
            cache=np.frombuffer(payload,'<u4',count=4,offset=offset+32)
            timing=np.frombuffer(payload,'<f8',count=2,offset=offset+48)
            result.update(cached_tokens=int(cache[0]),tree_cache_queries=int(cache[1]),
                          cache_nodes=int(cache[2]),cache_capacity=int(cache[3]),
                          cache_gpu_ms=float(timing[0]),cache_submit_wait_ms=float(timing[1]))
        else:
            result['cached_tokens']=len(ids) if result['reply_cache_hit'] else 0
            result['cache_gpu_ms']=result['gpu_ms'] if result['reply_cache_hit'] else 0.
        result['cache_path_hit']=result['cached_tokens']>0
        result['compute_tokens']=len(ids)-result['cached_tokens']
        result['compute_gpu_ms']=result['gpu_ms']-result['cache_gpu_ms']
        if result['cache_path_hit']:result['tree_cache_leaf']=int(hits[4])
        if len(payload)>=offset+88:
            restore=np.frombuffer(payload,'<f8',count=3,offset=offset+64)
            result.update(cache_state_gpu_ms=float(restore[0]),cache_state_tokens=int(restore[1]),
                          cache_checkpoint_restored=bool(restore[2]))
        if len(payload)>=offset+96:
            result.update(zip(('cache_candidates','cache_rejected_candidates'),map(int,np.frombuffer(payload,'<u4',count=2,offset=offset+88))))
        if len(payload)>=offset+112:
            result.update(zip(('mlp_route_rt_rows','mlp_route_cached_rows'),map(int,np.frombuffer(payload,'<u8',count=2,offset=offset+96))))
        return result

    def codec(self,values):
        values=np.asarray(values,dtype='<f4').ravel()
        if len(values)%2 or len(values)>min(self.capacity,32)*self.hidden_size:raise ValueError('Expected an even number of coordinates within the 32-row codec workspace')
        payload=self.request(5,len(values),False,values.tobytes());return np.frombuffer(payload,'<f4').copy()

    def vocabulary(self,query):
        payload=self.request(4,1,False,np.asarray(query,dtype='<f4').reshape(self.hidden_size).tobytes());offset=0
        def take(dtype,shape):
            nonlocal offset
            value=np.frombuffer(payload,dtype,count=int(np.prod(shape)),offset=offset).copy().reshape(shape);offset+=value.nbytes;return value
        result={'timings':take('<f8',(5,)),'winner':take('<u4',(4,)),'hit_count':int(take('<u4',(1,))[0]),
                'ids':take('<u4',(self.vocabulary_size,)),'geometric_words':take('<f4',(256,self.hidden_size))}
        result['ids']=result['ids'][:result['hit_count']];result['geometric_words']=result['geometric_words'][:min(256,result['hit_count'])];result['sample_ids']=result['ids'][:256].copy();result['token']=int(result['winner'][0])
        result['profile_gpu_ms']=dict(zip(('vocabulary_rt','vocabulary_coefficients','vocabulary_reduce'),map(float,result['timings'][2:])))
        result['query_packets']=take('<u4',(self.partition.blocks,128,2))
        result['restored_query']=self.partition.from_packets(result['query_packets'])
        result['block_charts']=take('<f4',(self.partition.blocks,4))[:,:2]
        result['block_hits']=int(take('<u4',(1,))[0])
        result['blocks']=self.partition.blocks
        result['block_coverage']=result['block_hits']/(self.vocabulary_size*self.partition.blocks)
        return result

    def mlp(self,inputs,layer=0):
        inputs=np.asarray(inputs,dtype='<f4').reshape(-1,self.hidden_size)
        if not 1<=len(inputs)<=min(self.capacity,self.prefill_batch,32):raise ValueError('MLP check supports up to 32 rows')
        if not np.isfinite(inputs).all():raise ValueError('MLP input must be finite')
        if not 0<=layer<self.metadata['depth']:raise ValueError('Invalid MLP layer')
        payload=self.request(6,len(inputs),layer,inputs.tobytes());size=len(inputs)*(self.hidden_size+self.metadata['middle']);values=np.frombuffer(payload,'<f4',count=size).copy();count=len(inputs)*self.hidden_size
        result={'output':values[:count].reshape(len(inputs),self.hidden_size),'activation':values[count:].reshape(len(inputs),self.metadata['middle']),'feature_hits':np.frombuffer(payload,'<u4',count=len(inputs),offset=size*4).copy()}
        offset=size*4+len(inputs)*4
        if len(payload)>=offset+24:
            names=('setup','gate_up','down') if self.metadata['mlp_backend']=='standard' else ('setup','rt_gate_up','rt_down')
            result['profile_gpu_ms']=dict(zip(names,map(float,np.frombuffer(payload,'<f8',count=3,offset=offset))))
        return result

    def attention(self,query,keys,values,gate=None,*,layer=3,query_start=None,history_start=0):
        """Exercise full attention without QKV or MLP; prefill again before generation."""
        spec=self.model_spec
        q=np.asarray(query,dtype='<f4').reshape(-1,spec.attention_heads,spec.attention_head_dim)
        k=np.asarray(keys,dtype='<f4').reshape(-1,spec.kv_heads,spec.attention_head_dim)
        v=np.asarray(values,dtype='<f4').reshape(-1,spec.kv_heads,spec.attention_head_dim)
        if not 1<=len(q)<=min(self.prefill_batch,32) or not len(k) or k.shape!=v.shape:
            raise ValueError('Invalid query rows or compact K/V shapes')
        if layer not in spec.full_layers:raise ValueError('Attention diagnostic requires a full-attention layer')
        if query_start is None:query_start=history_start+len(k)-len(q)
        if history_start<0 or query_start<history_start or query_start+len(q)>history_start+len(k) or history_start+len(k)>1048576:
            raise ValueError('Query times must lie inside the supplied causal history')
        g=np.zeros_like(q) if gate is None else np.asarray(gate,dtype='<f4').reshape(q.shape)
        if not all(np.isfinite(a).all() for a in (q,k,v,g)):raise ValueError('Attention values must be finite')
        self.ensure_capacity(len(k))
        body=struct.pack('<4I',len(k),query_start,history_start,0)+q.tobytes()+k.tobytes()+v.tobytes()+g.tobytes()
        self._attention_diagnostic_dirty=True
        payload=self.request(14,len(q),layer,body)
        header=np.frombuffer(payload,'<u4',count=12)
        magic,rows,count,heads,kv_heads,dimension,packet_bytes,block,stride,records,legacy,error=map(int,header)
        expected_records=spec.attention_heads*sum((min(query_start-history_start+row+1,self.capacity)+self.metadata['attention_block']-1)//self.metadata['attention_block'] for row in range(len(q)))
        if records!=expected_records or legacy!=int(history_start==0):raise RuntimeError('Full-attention diagnostic coverage protocol mismatch')
        compact=self.metadata.get('attention_receipt_format')=='compact_columns'
        direct=self.metadata.get('attention_receipt_format')=='score_weight_cache'
        expected_stride=spec.attention_head_dim+8+self.metadata['attention_block']//32
        if compact:expected_stride+=8+(spec.attention_head_dim+3)//4+(spec.attention_head_dim+31)//32
        if direct:expected_stride=8+self.metadata['attention_block']//32+2*self.metadata['attention_block']
        if (magic,rows,count,heads,kv_heads,dimension,packet_bytes,block,stride)!=(0x41544431 if direct else 0x41545231,len(q),len(k),spec.attention_heads,spec.kv_heads,spec.attention_head_dim,spec.history_bytes,self.metadata['attention_block'],expected_stride):
            raise RuntimeError('Full-attention diagnostic protocol mismatch')
        if error:raise RuntimeError(f'Full-attention diagnostic error {error}')
        timing=np.frombuffer(payload,'<f8',count=4,offset=48);offset=80
        def take(dtype,shape):
            nonlocal offset
            a=np.frombuffer(payload,dtype,count=int(np.prod(shape)),offset=offset).copy().reshape(shape)
            offset+=a.nbytes;return a
        result=dict(output=take('<f4',(rows,heads,dimension)),physical=take('<f4',(rows,heads,2)),
                    legacy_output=take('<f4',(rows,heads,dimension)),legacy_physical=take('<f4',(rows,heads,2)),
                    history_packets=take('u1',(count,packet_bytes)),receipts=take('<u4',(records,stride)),
                    legacy_available=bool(legacy),block=block,stride=stride,query_start=query_start,history_start=history_start,
                    fused_gpu_ms=float(timing[0]),merge_gpu_ms=float(timing[1]),gpu_ms=float(timing[0]+timing[1]),
                    legacy_gpu_ms=float(timing[2]),submit_wait_ms=float(timing[3]))
        if self.metadata['attention_backend']=='grouped_rt_local':
            result['local_classes']=take('<u4',(count,2,kv_heads,dimension))
            result['classification_gpu_ms']=float(take('<f8',(1,))[0])
            result['gpu_ms_with_classification']=result['gpu_ms']+result['classification_gpu_ms']
            result['classification_rays']=count*2*kv_heads*dimension
        if offset!=len(payload):raise RuntimeError('Unexpected full-attention diagnostic payload size')
        if direct:
            raw=result['receipts'];score_offset=8+block//32
            result.update(receipt_format='score_weight_cache',receipt_written_bytes=int(raw.nbytes),v_member_distribution=self.metadata['attention_v_members'],
                          cached_scores=raw[:,score_offset:score_offset+block].copy().view('<f4'),
                          cached_weights=raw[:,score_offset+block:].copy().view('<f4'),
                          key_gpu_ms=float(timing[0]),value_gpu_ms=float(timing[1]),
                          shader_work={name:int(raw[:,index].astype(np.uint64).sum()) for name,index in
                                       (('qk_products',0),('key_decodes',1),('covered_head_positions',2),
                                        ('rt_initializations',3),('value_products',4),('value_decodes',5),('exp_evaluations',6))})
        elif compact:
            raw=result['receipts'];coverage=block//32;map_offset=16+coverage
            mask_offset=map_offset+(dimension+3)//4;u_offset=mask_offset+(dimension+31)//32
            axis=np.arange(dimension,dtype=np.uint32)
            mapping=(raw[:,map_offset+axis//4]>>((axis%4)*8))&255
            row_index=np.arange(records)[:,None]
            if np.any(mapping>axis) or np.any(mapping[row_index,mapping]!=mapping):raise RuntimeError('Invalid compact coordinate mapping')
            representatives=mapping==axis
            dimensions=representatives.sum(axis=1)
            if not np.array_equal(dimensions,raw[:,9+coverage]):raise RuntimeError('Invalid compact coordinate count')
            expected_masks=np.zeros((records,(dimension+31)//32),dtype=np.uint32)
            for bit in range(dimension):expected_masks[:,bit//32]|=representatives[:,bit].astype(np.uint32)<<(bit%32)
            if not np.array_equal(expected_masks,raw[:,mask_offset:u_offset]):raise RuntimeError('Invalid compact coordinate mask')
            ranks=np.cumsum(representatives,axis=1)-1
            dense=np.empty((records,dimension+8+coverage),dtype=np.uint32)
            dense[:,:8+coverage]=raw[:,:8+coverage];dense[:,7]&=15
            dense[:,8+coverage:]=raw[row_index,u_offset+ranks[row_index,mapping]]
            result.update(compact_receipts=raw,receipts=dense,receipt_format='compact_columns',
                          receipt_written_bytes=int((u_offset+dimensions).sum()*4),
                          shader_work=dict(key_dimensions=int(raw[:,8+coverage].sum()),value_dimensions=int(dimensions.sum()),
                                           unique_history_terms=int(raw[:,10+coverage].sum()),qk_products=int(raw[:,11+coverage].astype(np.uint64).sum()),
                                           key_decodes=int(raw[:,12+coverage].astype(np.uint64).sum()),value_decodes=int(raw[:,13+coverage].astype(np.uint64).sum()),
                                           value_products=int(raw[:,14+coverage].astype(np.uint64).sum()),returned_value_dimensions=int(raw[:,15+coverage].sum())))
        else:result.update(receipt_format='dense',receipt_written_bytes=int(result['receipts'].nbytes))
        return result

    def attention_statistics(self,reset=False):
        names=('covered_head_positions','unique_head_positions','qk_products','value_products',
               'full_return_dimensions','returned_dimensions','key_dimensions','value_groups',
               'exp_evaluations','rt_initializations','records','q_dimensions','q_zeros','q_below_1e3','q_below_1e6','merged_dimensions')
        payload=self.request(15,int(reset));magic,count,layers,block=struct.unpack_from('<4I',payload)
        if (magic,count,layers,block)!=(0x41545331,len(names),len(self.model_spec.full_layers),self.metadata['attention_block']) or len(payload)!=16+layers*count*8:
            raise RuntimeError('Attention statistics protocol mismatch')
        values=np.frombuffer(payload,'<u8',offset=16).reshape(layers,count)
        return dict(total={name:int(values[:,i].sum()) for i,name in enumerate(names)},
                    layers=[dict(layer=layer,**{name:int(value) for name,value in zip(names,row)}) for layer,row in zip(self.model_spec.full_layers,values)])

    def resources(self):
        names = (
            'vulkan_allocation_bytes', 'device_allocation_bytes', 'word_scene_bytes',
            'word_vertex_bytes', 'word_rays', 'word_tile', 'feature_tile', 'reply_tree_bytes',
            'cache_snapshot_bytes', 'packed_coefficient_bytes', 'recurrent_control_cache_bytes',
            'logical_device_buffer_bytes', 'device_memory_pools', 'history_path_packet_bytes',
            'history_decoded_copy_bytes', 'optical_field_bytes', 'debug_audit_bytes', 'mlp_partial_bytes',
            'upload_staging_bytes', 'host_readback_bytes', 'history_scene_bytes', 'history_blas_count',
            'history_tlas_count', 'static_scene_scratch_bytes', 'hit_receipt_bytes', 'feature_receipt_bytes',
            'workspace_rows', 'detector_key_storage_shared', 'inactive_auxiliary_bytes', 'stable_snapshot_bytes',
            'history_boundary_state_bytes', 'history_shared_block_bytes', 'history_directory_bytes',
            'history_cache_boundaries', 'history_cache_evictions', 'mlp_transmission_scene_bytes',
            'mlp_transmission_surfaces', 'attention_backend_code', 'attention_block', 'attention_receipt_bytes',
            'attention_fused_dispatches', 'attention_merge_dispatches', 'attention_planned_rays',
            'attention_covered_terms', 'attention_reported_records', 'attention_class_planned_rays',
            'attention_class_builds', 'attention_class_map_bytes', 'attention_class_scene_bytes',
            'attention_class_map_slots', 'attention_class_peak_scene_bytes', 'attention_class_release_count',
            'attention_class_split_submissions', 'attention_class_directory_bytes',
            'attention_class_descriptor_bytes', 'attention_class_value_bytes', 'attention_class_member_bytes',
            'attention_class_scratch_bytes', 'attention_class_count_readback_bytes', 'attention_class_representatives',
        )
        return dict(zip(names, map(int, np.frombuffer(self.request(8, 0), '<u8'))))

    def set_aperture(self,aperture):
        if self.metadata['routing']!='sparse' or not 0<aperture<=2:raise ValueError('Expected sparse routing and aperture in (0,2]')
        payload=self.request(7,0,False,struct.pack('<f',aperture));self.metadata['route_aperture']=aperture
        return dict(zip(('gpu_ms','submit_wait_ms'),map(float,np.frombuffer(payload,'<f8'))))

    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.write(struct.pack('<I',0));self.process.stdin.flush();self.process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
        for stream in (self.process.stdin,self.process.stdout,self.stderr):
            try:stream.close()
            except OSError:pass
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
