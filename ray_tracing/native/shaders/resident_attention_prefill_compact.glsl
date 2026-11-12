#define GRAPH_BUCKETS 64
#define GRAPH_EARLY 1
#if GRAPH_BUCKETS
const uint HASH_BUCKETS=GRAPH_BUCKETS;
const uint HASH_WORDS=(COVERAGE_WORDS>2u*KV_BATCH*LANE_AXES?COVERAGE_WORDS:2u*KV_BATCH*LANE_AXES);
shared uint candidateBits[HASH_BUCKETS*HASH_WORDS];
uint candidateBucket(uint hash){return (hash^(hash>>16u))&(HASH_BUCKETS-1u);}
uint beforeMask(uint end,uint word){return word<end/32u?0xffffffffu:(end%32u==0u?0u:((1u<<(end%32u))-1u));}
#endif
shared uint tokenHash[BLOCK],tokenParent[BLOCK],tokenWeight[BLOCK],tokenList[BLOCK];
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
#if GRAPH_BUCKETS
    for(uint i=lane;i<HASH_BUCKETS*COVERAGE_WORDS;i+=gl_WorkGroupSize.x)candidateBits[i]=0u;
    barrier();
#endif
    for(uint i=lane;i<count;i+=gl_WorkGroupSize.x){
        uint h=2166136261u;
        // Sample packet words to shortlist candidates for full comparison.
        const uint pairs=HISTORY_RECORD_BYTES/8u;
        for(uint t=0u;t<3u;t++){uvec2 w=packetWord(localSlots[i],t*(pairs-1u)/2u);h=hashWord(hashWord(h,w.x),w.y);}
        tokenHash[i]=h;tokenWeight[i]=0u;
#if GRAPH_BUCKETS
        atomicOr(candidateBits[candidateBucket(h)*COVERAGE_WORDS+i/32u],1u<<(i%32u));
#endif
    }
    barrier();
    for(uint i=lane;i<count;i+=gl_WorkGroupSize.x){
        uint parent=i;
        #if GRAPH_BUCKETS
        for(uint w=0u;w<=i/32u&&parent==i;w++){
            uint bits=candidateBits[candidateBucket(tokenHash[i])*COVERAGE_WORDS+w]&beforeMask(i,w);
            while(bits!=0u){uint p=w*32u+uint(findLSB(bits));bits&=bits-1u;
                if(tokenHash[p]==tokenHash[i]&&samePacket(localSlots[p],localSlots[i])){parent=p;break;}
            }
        }
#else
        for(uint p=0u;p<i;p++)if(tokenHash[p]==tokenHash[i]&&samePacket(localSlots[p],localSlots[i])){parent=p;break;}
#endif
        tokenParent[i]=parent;atomicAdd(tokenWeight[parent],1u);
    }
    barrier();
    if(lane==0u){tokenCount=0u;for(uint i=0u;i<count;i++)if(tokenParent[i]==i)tokenList[tokenCount++]=i;}
    barrier();
}

// Within each occurrence, RT parent IDs are equal exactly when restored values
// are equal. Signatures only shortlist; sameColumn checks every occurrence.
uvec3 coordinateSignature(uint slot,uint channel,bool content){
    uint axis=sharedAxis(channel);
    return uvec3(classParent(classWord(slot,axis/DH,axis%DH,content)),0u,0u);
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
        #if GRAPH_EARLY
        if(kvBase+lane<kvHeads)for(uint h=0u;h<Q_PER_KV&&!needed;h++)for(uint a=0u;a<DH&&!needed;a++)
#else
        if(kvBase+lane<kvHeads)for(uint h=0u;h<Q_PER_KV;h++)for(uint a=0u;a<DH;a++)
#endif
            needed=needed||queries[row*A+((kvBase+lane)*Q_PER_KV+h)*DH+a]!=0.0;
        keyNeeded[lane]=uint(needed);
    }
#if GRAPH_BUCKETS
    for(uint i=lane;i<HASH_BUCKETS*2u*KV_BATCH*LANE_AXES;i+=gl_WorkGroupSize.x)candidateBits[i]=0u;
#endif
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
#if GRAPH_BUCKETS
        atomicOr(candidateBits[(2u*kv*HASH_BUCKETS+candidateBucket(hk))*LANE_AXES+axis/32u],1u<<(axis%32u));
        atomicOr(candidateBits[((2u*kv+1u)*HASH_BUCKETS+candidateBucket(hv))*LANE_AXES+axis/32u],1u<<(axis%32u));
#endif
    }
    barrier();
    for(uint i=lane;i<KV_BATCH*DH;i+=gl_WorkGroupSize.x){
        uint kv=i/DH,axis=i%DH;
        if(kvBase+kv>=kvHeads)continue;
#if GRAPH_BUCKETS
        if(keyNeeded[kv]==0u)keyMap[i]=0u;
        for(uint kind=0u;kind<2u;kind++){
            if(kind==0u&&keyNeeded[kv]==0u)continue;
            uint hash=kind==0u?keyHash[i]:valueHash[i],parent=axis;
            for(uint w=0u;w<=axis/32u&&parent==axis;w++){
                uint bits=candidateBits[((2u*kv+kind)*HASH_BUCKETS+candidateBucket(hash))*LANE_AXES+w]&beforeMask(axis,w);
                while(bits!=0u){uint p=w*32u+uint(findLSB(bits));bits&=bits-1u;
                    uint other=kind==0u?keyHash[kv*DH+p]:valueHash[kv*DH+p];
                    if(other==hash&&sameColumn(kvBase+kv,axis,p,kind==1u)){parent=p;break;}
                }
            }
            if(kind==0u)keyMap[i]=parent;else valueMap[i]=parent;
        }
#else
        if(keyNeeded[kv]==0u)keyMap[i]=0u;
        else for(uint p=0u;p<axis;p++){
            if(keyHash[kv*DH+p]==keyHash[i]&&sameColumn(kvBase+kv,axis,p,false)){keyMap[i]=p;break;}
        }
        for(uint p=0u;p<axis;p++){
            if(valueHash[kv*DH+p]==valueHash[i]&&sameColumn(kvBase+kv,axis,p,true)){valueMap[i]=p;break;}
        }
#endif
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
