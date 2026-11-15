#ifndef RT_ATTENTION_CLASSES_ACCESS
#define RT_ATTENTION_CLASSES_ACCESS
#endif
layout(std430,binding=45) RT_ATTENTION_CLASSES_ACCESS buffer AttentionClasses {uint attentionClasses[];};

const uint CLASS_KV_HEADS=RT_GQA_WIDTH/RT_ATTENTION_HEAD_DIM;
const uint CLASS_DOMAINS=2u*CLASS_KV_HEADS;
// Per-layer slots, append cursor and arena capacity precede the directory.
const uint CLASS_DIRECTORY_HEADER_WORDS=3u;
const uint CLASS_DIRECTORY_ENTRY_WORDS=2u;
const uint CLASS_PACKED_MEMBER_WORDS=RT_ATTENTION_HEAD_DIM/4u;

uint classDomain(uint kv,bool content){return uint(content)*CLASS_KV_HEADS+kv;}
uint classDirectorySlots(){return attentionClasses[0];}
uint classDirectoryAddress(uint slot,uint kv,bool content){
    uint local=slot%cfg.capacity;
    return CLASS_DIRECTORY_HEADER_WORDS+
        (local*CLASS_DOMAINS+classDomain(kv,content))*CLASS_DIRECTORY_ENTRY_WORDS;
}
uvec2 classDirectory(uint slot,uint kv,bool content){
    uint at=classDirectoryAddress(slot,kv,content);
    return uvec2(attentionClasses[at],attentionClasses[at+1u]);
}
uint classPayloadBase(){
    return CLASS_DIRECTORY_HEADER_WORDS+
        classDirectorySlots()*CLASS_DOMAINS*CLASS_DIRECTORY_ENTRY_WORDS;
}
uint classDomainAddress(uint slot,uint kv,bool content){
    return classPayloadBase()+classDirectory(slot,kv,content).x;
}
uint compactCount(uint slot,uint kv,bool content){return classDirectory(slot,kv,content).y;}

// Each live domain is exactly [D descriptors][D FP32 representative values]
// [256 packed membership bytes]. K stores its original grouped axis list;
// V stores the inverse axis-to-rank map in the same bytes, so every consumer
// can address a representative without rebuilding or copying a dense map.
// Descriptors keep root8|memberOffset8|count9<<16 for both kinds.
uint compactDescriptor(uint slot,uint kv,uint rank,bool content){
    return attentionClasses[classDomainAddress(slot,kv,content)+rank];
}
uint compactAxis(uint descriptor){return descriptor&255u;}
uint compactMemberOffset(uint descriptor){return (descriptor>>8u)&255u;}
uint compactMemberCount(uint descriptor){return (descriptor>>16u)&511u;}
float classValue(uint slot,uint kv,uint rank,bool content){
    uint count=compactCount(slot,kv,content);
    return uintBitsToFloat(attentionClasses[classDomainAddress(slot,kv,content)+count+rank]);
}
uint compactValueRankAt(uint payload,uint count,uint axis){
    uint word=attentionClasses[payload+2u*count+axis/4u];
    return (word>>((axis&3u)*8u))&255u;
}
uint compactValueRank(uint slot,uint kv,uint axis){
    uvec2 directory=classDirectory(slot,kv,true);
    return compactValueRankAt(classPayloadBase()+directory.x,directory.y,axis);
}
uint compactMember(uint slot,uint kv,uint index,bool content){
    uint count=compactCount(slot,kv,content);
    uint payload=classDomainAddress(slot,kv,content);
    if(!content){
        uint word=attentionClasses[payload+2u*count+index/4u];
        return (word>>((index&3u)*8u))&255u;
    }
    // Preserve the grouped-list reader's meaning for legacy diagnostics.
    // Numerical V consumers use compactValueRankAt directly.
    for(uint rank=0u;rank<count;rank++){
        uint descriptor=attentionClasses[payload+rank];
        uint offset=compactMemberOffset(descriptor),members=compactMemberCount(descriptor);
        if(index<offset||index>=offset+members)continue;
        uint selected=index-offset;
        for(uint axis=0u;axis<RT_ATTENTION_HEAD_DIM;axis++)
            if(compactValueRankAt(payload,count,axis)==rank){if(selected==0u)return axis;selected--;}
    }
    return RT_ATTENTION_HEAD_DIM;
}
