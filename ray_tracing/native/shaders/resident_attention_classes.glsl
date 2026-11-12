layout(std430,binding=45) buffer AttentionClasses {uint attentionClasses[];};

const uint CLASS_KV_HEADS=RT_GQA_WIDTH/RT_ATTENTION_HEAD_DIM;
const uint CLASS_DOMAINS=2u*CLASS_KV_HEADS;
const uint CLASS_DIRECTORY_HEADER_WORDS=1u;
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
// [256 packed member bytes]. Descriptors keep root8|memberOffset8|count9<<16.
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
uint compactMember(uint slot,uint kv,uint index,bool content){
    uint count=compactCount(slot,kv,content);
    uint word=attentionClasses[classDomainAddress(slot,kv,content)+2u*count+index/4u];
    return (word>>((index&3u)*8u))&255u;
}
