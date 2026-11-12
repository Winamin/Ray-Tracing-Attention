#ifdef RT_ATTENTION_TC
shared uint tokenHash[BLOCK],tokenWeight[BLOCK],tokenList[BLOCK];
#else
shared uint tokenHash[BLOCK],tokenParent[BLOCK],tokenWeight[BLOCK],tokenList[BLOCK];
#endif
shared uint tokenCount;
shared uint keyHash[KV_BATCH*DH],valueHash[KV_BATCH*DH];
shared uint keyMap[KV_BATCH*DH],valueMap[KV_BATCH*DH];
shared uint keyDimensions[KV_BATCH],valueDimensions[KV_BATCH];
shared uint keyNeeded[KV_BATCH];

uint hashWord(uint h,uint x){return (h^x)*16777619u;}
uvec2 packetWord(uint slot,uint pair){return keyGeometry[slot*(HISTORY_RECORD_BYTES/8u)+pair];}
bool samePacket(uint a,uint b){
    for(uint i=0u;i<HISTORY_RECORD_BYTES/8u;i++)if(any(notEqual(packetWord(a,i),packetWord(b,i))))return false;
    return true;
}
void compactTokens(uint count,uint lane){
    for(uint i=lane;i<count;i+=gl_WorkGroupSize.x){
        uint h=2166136261u;
        // Sample packet words to shortlist candidates for full comparison.
        const uint pairs=HISTORY_RECORD_BYTES/8u;
        for(uint t=0u;t<3u;t++){uvec2 w=packetWord(localSlots[i],t*(pairs-1u)/2u);h=hashWord(hashWord(h,w.x),w.y);}
        tokenHash[i]=h;tokenWeight[i]=0u;
    }
    barrier();
    for(uint i=lane;i<count;i+=gl_WorkGroupSize.x){
        uint parent=i;
        for(uint p=0u;p<i;p++)if(tokenHash[p]==tokenHash[i]&&samePacket(localSlots[p],localSlots[i])){parent=p;break;}
#ifndef RT_ATTENTION_TC
        tokenParent[i]=parent;
#endif
        atomicAdd(tokenWeight[parent],1u);
    }
    barrier();
    if(lane==0u){tokenCount=0u;for(uint i=0u;i<count;i++){
#ifdef RT_ATTENTION_TC
        // Only a representative receives multiplicity; its parent array is
        // redundant after the workgroup barrier.
        if(tokenWeight[i]!=0u)tokenList[tokenCount++]=i;
#else
        if(tokenParent[i]==i)tokenList[tokenCount++]=i;
#endif
    }}
    barrier();
}

// Signatures identify restored FP32 values, including equivalent quantized
// expressions. Sample hashes shortlist; full column checks establish equality.
uvec3 coordinateSignature(uint slot,uint channel,bool content){
    precise float value=pathCoordinate(slot,channel,content);
    return uvec3(value==0.0?0u:floatBitsToUint(value),0u,0u);
}
bool sameColumn(uint kv,uint a,uint b,bool content){
    uint base=kv*Q_PER_KV*DH;
    for(uint i=0u;i<tokenCount;i++){
        uint slot=localSlots[tokenList[i]];
#ifdef RT_LOCAL_CLASSES
        if(classParent(classWord(slot,kv,a,content))!=classParent(classWord(slot,kv,b,content)))return false;
#else
        if(any(notEqual(coordinateSignature(slot,base+a,content),coordinateSignature(slot,base+b,content))))return false;
#endif
    }
    return true;
}
void compactDimensions(uint row,uint kvBase,uint kvHeads,uint lane){
    if(lane<KV_BATCH){
        bool needed=false;
        if(kvBase+lane<kvHeads)for(uint h=0u;h<Q_PER_KV;h++)for(uint a=0u;a<DH;a++)
            needed=needed||queries[row*A+((kvBase+lane)*Q_PER_KV+h)*DH+a]!=0.0;
        keyNeeded[lane]=uint(needed);
    }
    barrier();
    for(uint i=lane;i<KV_BATCH*DH;i+=gl_WorkGroupSize.x){
        uint kv=i/DH,axis=i%DH,hk=2166136261u,hv=2166136261u;
        uint samples=min(3u,tokenCount);
        if(kvBase+kv<kvHeads)for(uint t=0u;t<samples;t++){
            uint position=samples==1u?0u:t*(tokenCount-1u)/(samples-1u);
            uint slot=localSlots[tokenList[position]],channel=(kvBase+kv)*Q_PER_KV*DH+axis;
            uvec3 v=coordinateSignature(slot,channel,true);
            if(keyNeeded[kv]!=0u){uvec3 k=coordinateSignature(slot,channel,false);hk=hashWord(hashWord(hashWord(hk,k.x),k.y),k.z);}
            hv=hashWord(hashWord(hashWord(hv,v.x),v.y),v.z);
        }
        keyHash[i]=hk;valueHash[i]=hv;keyMap[i]=axis;valueMap[i]=axis;
    }
    barrier();
    for(uint i=lane;i<KV_BATCH*DH;i+=gl_WorkGroupSize.x){
        uint kv=i/DH,axis=i%DH;
        if(kvBase+kv>=kvHeads)continue;
        if(keyNeeded[kv]==0u)keyMap[i]=0u;
        else for(uint p=0u;p<axis;p++){
            if(keyHash[kv*DH+p]==keyHash[i]&&sameColumn(kvBase+kv,axis,p,false)){keyMap[i]=p;break;}
        }
        for(uint p=0u;p<axis;p++){
            if(valueHash[kv*DH+p]==valueHash[i]&&sameColumn(kvBase+kv,axis,p,true)){valueMap[i]=p;break;}
        }
    }
    barrier();
    if(lane<KV_BATCH){
        uint nk=0u,nv=0u;
        for(uint a=0u;a<DH;a++){
            if(keyMap[lane*DH+a]==a)keyHash[lane*DH+nk++]=a;
            if(valueMap[lane*DH+a]==a)valueHash[lane*DH+nv++]=a;
        }
        keyDimensions[lane]=keyNeeded[lane]!=0u?nk:0u;valueDimensions[lane]=nv;
    }
    barrier();
}
uint valueRank(uint kv,uint axis){uint rank=0u;for(uint a=0u;a<axis;a++)rank+=uint(valueMap[kv*DH+a]==a);return rank;}
const uint MAP_WORDS=(DH+3u)/4u;
const uint MASK_WORDS=(DH+31u)/32u;
const uint MAP_OFFSET=16u+COVERAGE_WORDS;
const uint MASK_OFFSET=MAP_OFFSET+MAP_WORDS;
const uint U_OFFSET=MASK_OFFSET+MASK_WORDS;
uint receiptAxis(uint at,uint axis){return (attentionReceipts[at+MAP_OFFSET+axis/4u]>>((axis%4u)*8u))&255u;}
uint receiptRank(uint at,uint axis){
    uint rank=0u;
    for(uint w=0u;w<axis/32u;w++)rank+=bitCount(attentionReceipts[at+MASK_OFFSET+w]);
    uint bit=axis%32u;uint mask=bit==0u?0u:((1u<<bit)-1u);
    return rank+bitCount(attentionReceipts[at+MASK_OFFSET+axis/32u]&mask);
}
