// Operations 0--2 use one fixed scratch record per row/K-or-V/KV-head.
// Operation 3 copies only the D live descriptors and D live FP32 values into
// the host-allocated variable payload. Operation 4 reconstructs the legacy
// per-axis diagnostic map from the compact persistent record.
shared uint compactWords[RT_ATTENTION_HEAD_DIM];
shared uint compactMembers[RT_ATTENTION_HEAD_DIM];
shared uint compactEntries[RT_ATTENTION_HEAD_DIM];
shared uint compactRootTotals[4],compactMemberTotals[4];

void buildCompactClasses(){
    uint domain=gl_WorkGroupID.x,domains=CLASS_DOMAINS;
    uint row=domain/domains,kind=domain%domains;
    if(row>=cfg.rows)return;
    uint lane=gl_LocalInvocationID.x,c=gl_SubgroupInvocationID,sg=gl_SubgroupID;
    uint base=tempDomainAddress(row,kind),dim=RT_ATTENTION_HEAD_DIM;
    for(uint axis=lane;axis<dim;axis+=gl_WorkGroupSize.x)compactWords[axis]=attentionClassScratch[base+CLASS_TEMP_MAP_OFFSET+axis];
    barrier();
    uint rootCount=0u,memberCount=0u;
    for(uint axis=lane;axis<dim;axis+=gl_WorkGroupSize.x){
        uint word=compactWords[axis];
        if(tempClassParent(word)==axis){rootCount++;memberCount+=tempClassCount(word);}
    }
    uint rootOffset=subgroupExclusiveAdd(rootCount),memberOffset=subgroupExclusiveAdd(memberCount);
    uint rootTotal=subgroupAdd(rootCount),memberTotal=subgroupAdd(memberCount);
    if(c==0u){compactRootTotals[sg]=rootTotal;compactMemberTotals[sg]=memberTotal;}
    barrier();
    uint roots=0u,members=0u;
    for(uint s=0u;s<4u;s++){
        if(s<sg){rootOffset+=compactRootTotals[s];memberOffset+=compactMemberTotals[s];}
        roots+=compactRootTotals[s];members+=compactMemberTotals[s];
    }
    if(roots==0u||roots>dim||members!=dim){if(lane==0u)atomicOr(control[3],65536u);return;}
    uint at=rootOffset,nextMember=memberOffset;
    for(uint axis=lane;axis<dim;axis+=gl_WorkGroupSize.x){
        uint word=compactWords[axis];if(tempClassParent(word)!=axis)continue;
        uint count=tempClassCount(word),seen=0u,member=axis;
        while(member<dim&&seen<count){
            compactMembers[nextMember+seen]=member;
            member=tempClassNext(compactWords[member]);seen++;
        }
        if(seen!=count||member!=dim)atomicOr(control[3],65536u);
        compactEntries[at]=axis|(nextMember<<8u)|(count<<16u);
        at++;nextMember+=count;
    }
    barrier();
    for(uint i=lane;i<roots;i+=gl_WorkGroupSize.x){
        uint descriptor=compactEntries[i],axis=compactAxis(descriptor);
        attentionClassScratch[base+CLASS_TEMP_DESCRIPTOR_OFFSET+i]=descriptor;
        attentionClassScratch[base+CLASS_TEMP_VALUE_OFFSET+i]=attentionClassScratch[base+CLASS_TEMP_BITS_OFFSET+axis];
    }
    for(uint i=lane;i<CLASS_PACKED_MEMBER_WORDS;i+=gl_WorkGroupSize.x){
        uint j=4u*i;
        attentionClassScratch[base+CLASS_TEMP_MEMBER_OFFSET+i]=compactMembers[j]|(compactMembers[j+1u]<<8u)|(compactMembers[j+2u]<<16u)|(compactMembers[j+3u]<<24u);
    }
    if(lane==0u)attentionClassScratch[base+CLASS_TEMP_COUNT_OFFSET]=roots;
}

void storeCompactClasses(){
    uint domain=gl_WorkGroupID.x,domains=CLASS_DOMAINS;
    uint row=domain/domains,kind=domain%domains;
    if(row>=cfg.rows)return;
    bool content=kind>=CLASS_KV_HEADS;uint kv=kind%CLASS_KV_HEADS;
    uint lane=gl_LocalInvocationID.x,slot=parameters[blockBase()+41u]*cfg.capacity+(cfg.position+row)%cfg.capacity;
    uint source=tempDomainAddress(row,kind),count=compactCount(slot,kv,content),target=classDomainAddress(slot,kv,content);
    if(count==0u||count>RT_ATTENTION_HEAD_DIM){if(lane==0u)atomicOr(control[3],65536u);return;}
    for(uint i=lane;i<count;i+=gl_WorkGroupSize.x){
        attentionClasses[target+i]=attentionClassScratch[source+CLASS_TEMP_DESCRIPTOR_OFFSET+i];
        attentionClasses[target+count+i]=attentionClassScratch[source+CLASS_TEMP_VALUE_OFFSET+i];
    }
    for(uint i=lane;i<CLASS_PACKED_MEMBER_WORDS;i+=gl_WorkGroupSize.x)
        attentionClasses[target+2u*count+i]=attentionClassScratch[source+CLASS_TEMP_MEMBER_OFFSET+i];
}

void expandCompactClasses(){
    uint domain=gl_WorkGroupID.x,domains=CLASS_DOMAINS;
    uint row=domain/domains,kind=domain%domains;
    if(row>=cfg.rows)return;
    bool content=kind>=CLASS_KV_HEADS;uint kv=kind%CLASS_KV_HEADS;
    uint lane=gl_LocalInvocationID.x,slot=parameters[blockBase()+41u]*cfg.capacity+(cfg.position+row)%cfg.capacity;
    uint base=tempDomainAddress(row,kind),count=compactCount(slot,kv,content),dim=RT_ATTENTION_HEAD_DIM;
    if(count==0u||count>dim){if(lane==0u)atomicOr(control[3],65536u);return;}
    for(uint rank=lane;rank<count;rank+=gl_WorkGroupSize.x){
        uint descriptor=compactDescriptor(slot,kv,rank,content),root=compactAxis(descriptor);
        uint offset=compactMemberOffset(descriptor),members=compactMemberCount(descriptor);
        if(root>=dim||members==0u||offset+members>dim){atomicOr(control[3],65536u);continue;}
        for(uint m=0u;m<members;m++){
            uint axis=compactMember(slot,kv,offset+m,content);
            uint next=m+1u<members?compactMember(slot,kv,offset+m+1u,content):dim;
            if(axis>=dim){atomicOr(control[3],65536u);continue;}
            attentionClassScratch[base+CLASS_TEMP_MAP_OFFSET+axis]=root|(next<<8u)|(members<<17u);
        }
    }
}
