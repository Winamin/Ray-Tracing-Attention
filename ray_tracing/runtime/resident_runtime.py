"""One resident Vulkan worker with batch replies and native per-token streaming."""
from pathlib import Path
import os
import json
import struct
import subprocess
import tempfile
import numpy as np

class ResidentRuntime:
    def __init__(self,asset,capacity=300,cache=True,routing="rt_full",aperture=.08,word_tile=32,feature_tile=8,readout_unroll=1,readout_words=4,prefill_batch=512,matrix_rows=8):
        asset=Path(asset).resolve()
        from ..compiler.partition import ensure_partition
        self.partition=ensure_partition(asset.parent)
        config=json.loads((asset.parent/'model.json').read_text(encoding='utf8'))
        from ..compiler.spec import asset_spec
        from ..compiler.build import build
        spec=asset_spec(asset.parent)
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
        self.process=subprocess.Popen([str(worker),str(Path(asset).resolve()),str(Path(asset).resolve().parent/'vocabulary.bin'),str(capacity),str(int(cache)),routing,str(aperture),str(word_tile),str(feature_tile),str(readout_unroll),str(readout_words),str(prefill_batch),str(matrix_rows)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,cwd=native,**options)
        try:
            magic,depth,blocks,middle,mode=struct.unpack('<5I',self.read(20))
            if magic not in (0x52545239,0x5254523a):raise RuntimeError('Resident optical protocol mismatch')
        except BaseException:
            self.close()
            raise
        self.metadata={'depth':depth,'blocks':blocks,'middle':middle,'hidden_size':self.hidden_size,'quantization':config.get('precision','bf16 optical'),'context_window':capacity,'growable_tree':self.growable,'state_cache':bool(cache),'routing':routing,'route_aperture':2.0 if routing=='rt_full' else aperture,'word_tile':word_tile,'feature_tile':feature_tile,'readout_unroll':readout_unroll,'readout_words':readout_words}
        self.metadata.update(native_token_stream=magic==0x5254523a,rt_query_telemetry=magic==0x5254523a,worker=str(worker))
        self.metadata['prefill_batch']=prefill_batch
        self.metadata['matrix_rows']=matrix_rows
        self.metadata['prefill_compute']='cooperative_matrix2' if mode&256 else 'subgroup'
        self.metadata['matrix_channel_tile']=min(64,spec.attention_head_dim) if mode&256 else 4
        self.metadata['matrix_token_tile']=256 if mode&256 else matrix_rows
        self.metadata['reply_cache_kind']='verified optical state path'
        self.metadata['cache_verification_bits']=256
        self.metadata['decode_route_cache']=bool(cache)
        self.metadata['source_model']=config.get('source_model')
        self.metadata['coordinate_partition']=config['coordinate_partition']
        self.position=0
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
        self.position=len(ids)
        return result

    def finish_history(self):
        payload=self.request(13,0)
        result=dict(zip(('history_output_saved','history_output_tokens','history_cache_boundaries'),
                        map(int,np.frombuffer(payload,'<u8',count=3))))
        result['history_output_capture_wall_ms']=float(np.frombuffer(payload,'<f8',count=1,offset=24)[0])
        return result

    def generate(self,count=32,debug=False,profile=False,telemetry=False):
        self.ensure_capacity(self.position+count)
        mode=2 if profile else 1 if debug else 3 if telemetry and self.metadata['rt_query_telemetry'] else 0
        return self._generation_result(self.request(2,count,mode),debug,profile,telemetry)

    def stream_generate(self,count=32,profile=False,telemetry=True):
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
        if not 1<=len(inputs)<=min(self.capacity,32):raise ValueError('MLP check supports up to 32 rows')
        payload=self.request(6,len(inputs),layer,inputs.tobytes());size=len(inputs)*(self.hidden_size+self.metadata['middle']);values=np.frombuffer(payload,'<f4',count=size).copy();count=len(inputs)*self.hidden_size
        result={'output':values[:count].reshape(len(inputs),self.hidden_size),'activation':values[count:].reshape(len(inputs),self.metadata['middle']),'feature_hits':np.frombuffer(payload,'<u4',count=len(inputs),offset=size*4).copy()}
        offset=size*4+len(inputs)*4
        if len(payload)>=offset+24:result['profile_gpu_ms']=dict(zip(('rt_receipt','gate_up','down'),map(float,np.frombuffer(payload,'<f8',count=3,offset=offset))))
        return result

    def resources(self):
        return dict(zip(('vulkan_allocation_bytes','device_allocation_bytes','word_scene_bytes','word_vertex_bytes','word_rays','word_tile','feature_tile','reply_tree_bytes','cache_snapshot_bytes','packed_coefficient_bytes','recurrent_control_cache_bytes','logical_device_buffer_bytes','device_memory_pools','history_path_packet_bytes','history_decoded_copy_bytes','optical_field_bytes','debug_audit_bytes','mlp_partial_bytes','upload_staging_bytes','host_readback_bytes','history_scene_bytes','history_blas_count','history_tlas_count','static_scene_scratch_bytes','hit_receipt_bytes','feature_receipt_bytes','workspace_rows','detector_key_storage_shared','inactive_auxiliary_bytes','stable_snapshot_bytes','history_boundary_state_bytes','history_shared_block_bytes','history_directory_bytes','history_cache_boundaries','history_cache_evictions'),map(int,np.frombuffer(self.request(8,0),'<u8'))))

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
