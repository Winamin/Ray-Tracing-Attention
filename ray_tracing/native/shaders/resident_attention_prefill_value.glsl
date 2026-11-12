// Consume scores from the K dispatch, then overwrite U with the final summary.
// Two history banks let the next load overlap reads from the current bank.
shared uint classTile[KV_BATCH*MICRO*DH];
#ifdef RT_LOCAL_PREFILL
// One padded product/residual pair of rows owned by each subgroup.
const uint WORK_STRIDE=DH+1u;
const uint WORK_PLANE=KV_BATCH*Q_BATCH*WORK_STRIDE;
shared float localWork[2u*WORK_PLANE];
#endif
// Paired GGUF reads share scales and packed loads, as in llama.cpp dequant_funcs.
// Only representatives publish values and enter product/compensation arithmetic.
void loadClassTile(uint microFirst,uint n,uint kvBase,uint kvHeads,uint lane,bool content){
    for(uint i=lane;i<KV_BATCH*n*(DH/2u);i+=gl_WorkGroupSize.x){
        uint axis=(i%(DH/2u))*2u,j=(i/(DH/2u))%n,kv=i/(n*(DH/2u));
        if(kvBase+kv>=kvHeads)continue;
        uint slot=localSlots[tokenList[microFirst+j]],at=(kv*MICRO+((microFirst+uint(content)*tokenCount)&1u))*DH+axis;
        uvec2 word=uvec2(classWord(slot,kvBase+kv,axis,content),classWord(slot,kvBase+kv,axis+1u,content));
        classTile[at]=word.x;classTile[at+1u]=word.y;
        bvec2 representative=equal(word&255u,uvec2(axis,axis+1u));vec2 value=vec2(0.0);
        if(any(representative)){
            uint compact=(kvBase+kv)*DH+axis,offset=slot*HISTORY_RECORD_BYTES+(content?((RT_GQA_WIDTH+255u)/256u)*144u:0u)+(compact/256u)*(content?210u:144u);
            value=ggufPair(2u,offset,content?14u:12u,compact%256u);
        }
        addressTile[at]=representative.x?value.x:0.0;addressTile[at+1u]=representative.y?value.y:0.0;
        if(any(isnan(value))||any(isinf(value)))attentionFailure();
    }
    barrier();
}
void localAttention(uint row,uint group,uint lane){
    uint available=availablePositions(row),first=group*BLOCK;
    if(first>=available)return;
    uint count=min(BLOCK,available-first);
    traceGroup(row,group,count,lane);if(groupFailure!=0u)return;
    compactTokens(count,lane);
    uint c=gl_SubgroupInvocationID,subgroup=gl_SubgroupID,heads=headCount();
    uint kvHeads=(heads+Q_PER_KV-1u)/Q_PER_KV,kvLane=subgroup/Q_BATCH,qLane=subgroup%Q_BATCH;
    for(uint kvBase=0u;kvBase<kvHeads;kvBase+=KV_BATCH){
        compactDimensions(row,kvBase,kvHeads,lane);
        for(uint headOffset=0u;headOffset<Q_PER_KV;headOffset+=Q_BATCH){
            uint head=(kvBase+kvLane)*Q_PER_KV+headOffset+qLane;
            bool headActive=kvBase+kvLane<kvHeads&&head<heads&&headOffset+qLane<Q_PER_KV;
            precise float signal[LANE_AXES],correction[LANE_AXES];
            for(uint k=0u;k<LANE_AXES;k++){signal[k]=0.0;correction[k]=0.0;}
            float scores[LANE_SCORES],maximum=-1.0/0.0;
            uint at=recordIndex(row,group,head)*RECORD_WORDS;
            uint productsK=headActive?attentionReceipts[at+11u+COVERAGE_WORDS]:0u;
            uint decodesK=headActive?attentionReceipts[at+12u+COVERAGE_WORDS]:0u;
            uint productsV=0u,decodesV=0u,expCount=0u;
            for(uint k=0u;k<LANE_SCORES;k++){
                uint index=c+k*32u;
                scores[k]=headActive&&index<tokenCount?uintBitsToFloat(attentionReceipts[at+U_OFFSET+index]):-1.0/0.0;
                maximum=max(maximum,scores[k]);
            }
            if(headActive)maximum=subgroupMax(maximum);
            precise float reference=0.0,referenceError=0.0;
            float previousWeight=0.0;uint previousScore=0u;bool havePrevious=false,reused=false;
            for(uint microFirst=0u;microFirst<tokenCount;microFirst++){
                uint n=1u;
                loadClassTile(microFirst,n,kvBase,kvHeads,lane,true);
                if(groupFailure!=0u)return;
                for(uint j=0u;j<n;j++){
                    uint index=microFirst+j,base=(kvLane*MICRO+((microFirst+tokenCount)&1u))*DH;
                    float score=subgroupShuffle(scores[index/32u],index%32u),weight=0.0;
                    if(headActive&&c==0u){
                        uint bits=floatBitsToUint(score);
                        if(havePrevious&&bits==previousScore){weight=previousWeight;reused=true;}
                        else{weight=exp(score-maximum);expCount++;}
                        previousWeight=weight;previousScore=bits;havePrevious=true;
                    }
                    weight=subgroupBroadcastFirst(weight)*float(tokenWeight[tokenList[index]]);
                    if(headActive)accurateAdd(reference,referenceError,weight);
                    for(uint k=0u;k<LANE_AXES;k++){
                        uint axis=c+k*32u;precise float product=0.0,residual=0.0;
                        if(headActive&&axis<DH&&classParent(classTile[base+axis])==axis){
                            decodesV++;float value=contentTile[base+axis];
                            if(value!=0.0){product=weight*value;residual=fma(weight,value,-product);productsV++;}
                        }
                        if(axis<DH){localWork[subgroup*WORK_STRIDE+axis]=product;localWork[WORK_PLANE+subgroup*WORK_STRIDE+axis]=residual;}
                    }
#ifdef RT_LOCAL_PREFILL
                    subgroupMemoryBarrierShared();subgroupBarrier();
#endif
                    for(uint k=0u;k<LANE_AXES;k++){
                        uint rank=c+k*32u;
                        uint axis=rank<valueDimensions[kvLane]?valueHash[kvLane*DH+rank]:0u;
                        uint parent=classParent(classTile[base+axis]);float contribution=0.0,tail=0.0;
#ifdef RT_LOCAL_PREFILL
                        contribution=localWork[subgroup*WORK_STRIDE+parent];tail=localWork[WORK_PLANE+subgroup*WORK_STRIDE+parent];
#else
                        for(uint t=0u;t<LANE_AXES;t++){
                            float a=subgroupShuffle(product[t],parent%32u),b=subgroupShuffle(residual[t],parent%32u);
                            if(t==parent/32u){contribution=a;tail=b;}
                        }
#endif
                        if(headActive&&rank<valueDimensions[kvLane]){accurateAdd(signal[k],correction[k],contribution);accurateAdd(signal[k],correction[k],tail);}
                    }
#ifdef RT_LOCAL_PREFILL
                    // Finish every member's read before the next token overwrites
                    // this subgroup's products; no other subgroup owns this row.
                    subgroupMemoryBarrierShared();subgroupBarrier();
#endif
                }

            }
            productsV=subgroupAdd(productsV);
            decodesV=subgroupAdd(decodesV);
            if(headActive){
                reference-=referenceError;
                for(uint k=0u;k<LANE_AXES;k++)signal[k]-=correction[k];
                if(!finiteValue(reference)||reference<=0.0||!finiteValue(maximum))attentionFailure();
                storeHead(row,group,head,kvLane,count,expCount,tokenCount,productsK,reused,maximum,reference,signal,
                    qLane==0u?decodesK:0u,qLane==0u?decodesV:0u,productsV);
            }
            barrier();if(groupFailure!=0u)return;
        }
    }
}
