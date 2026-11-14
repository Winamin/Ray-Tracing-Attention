// Decode uses the same weights and compact V records as the multi-row path.
// History is partitioned at its existing 64-term FP32 accumulation boundaries.
// Like llama.cpp's split-KV dispatch (MIT; see THIRD_PARTY.md), independent
// history intervals run concurrently. Their sums are merged in original order.
// https://github.com/ggml-org/llama.cpp/blob/7d7f02875/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L8414-L8455
#include "resident_common.glsl"
#define RT_ATTENTION_CLASSES_ACCESS readonly restrict
#include "resident_attention_classes.glsl"
#define headCount() parameters[blockBase()+23u]
layout(constant_id=1) const uint BLOCK=128u;
layout(constant_id=7) const uint DISTRIBUTE_V_MEMBERS=0u;
layout(std430,binding=42) buffer AttentionReceipts {uint attentionReceipts[];};
const uint DH=RT_ATTENTION_HEAD_DIM,Q_PER_KV=RT_QUERY_WIDTH/RT_GQA_WIDTH;
const uint COVERAGE_WORDS=BLOCK/32u,SCORE_OFFSET=8u+COVERAGE_WORDS;
const uint WEIGHT_OFFSET=SCORE_OFFSET+BLOCK,RECORD_WORDS=WEIGHT_OFFSET+BLOCK;
const uint ATTENTION_ERROR=65536u,CHUNK=64u,CHUNKS=BLOCK/CHUNK;
shared uint groupFailure;
#include "resident_attention_local_common.glsl"
bool splitValid(uint width){
    if(cfg.rows!=1u||DH!=256u||Q_PER_KV!=4u||BLOCK%CHUNK!=0u||
       gl_WorkGroupSize.x!=width||gl_SubgroupSize!=32u){
        if(gl_LocalInvocationID.x==0u)atomicOr(control[3],ATTENTION_ERROR);
        return false;
    }
    return !stopped()&&(control[3]&ATTENTION_ERROR)==0u;
}
bool splitDiagnostics(){return (cfg.debug&(65536u|131072u))!=0u;}
bool localAggregatePolicyMatches(){
    return pathAggregateEnabled()&&(pathScoreCache[7u]&65535u)==BLOCK&&
        ((pathScoreCache[7u]>>18u)&1u)==DISTRIBUTE_V_MEMBERS;
}
uint splitWeight(uint group,uint j,uint head){
    return recordIndex(0u,group,head)*RECORD_WORDS+WEIGHT_OFFSET+j;
}
uint splitChunkAddress(uint group,uint chunk,uint head,uint axis){
    uint records=groupCount(0u)*headCount();
    return records*RECORD_WORDS+(recordIndex(0u,group,head)*CHUNKS+chunk)*DH+axis;
}
void accurateAdd4(inout vec4 sum,inout vec4 correction,vec4 value){
    precise vec4 adjusted=value-correction,next=sum+adjusted;
    correction=(next-sum)-adjusted;sum=next;
}
