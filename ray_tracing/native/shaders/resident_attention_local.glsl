// Classes belong to each history occurrence, not to the whole age block.
// The same persisted RT membership serves all query rows and GQA heads.
shared uint classTile[KV_BATCH*MICRO*DH];
#ifdef RT_LOCAL_PREFILL
// One padded row per subgroup. Q is live only during the K pass; V products
// and their FMA residuals reuse this storage after that pass's final barrier.
const uint WORK_STRIDE=DH+1u;
const uint WORK_PLANE=KV_BATCH*Q_BATCH*WORK_STRIDE;
shared float localWork[2u*WORK_PLANE];
#endif
void loadClassTile(uint microFirst,uint n,uint kvBase,uint kvHeads,uint lane,bool content){
    for(uint i=lane;i<KV_BATCH*n*DH;i+=gl_WorkGroupSize.x){
        uint axis=i%DH,j=(i/DH)%n,kv=i/(n*DH);
        if(kvBase+kv>=kvHeads)continue;
        uint slot=localSlots[tokenList[microFirst+j]],at=(kv*MICRO+j)*DH+axis;
        uint word=classWord(slot,kvBase+kv,axis,content);classTile[at]=word;
        float value=classParent(word)==axis?pathCoordinate(slot,(kvBase+kv)*Q_PER_KV*DH+axis,content):0.0;
        addressTile[at]=value;if(!finiteValue(value))attentionFailure();
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
            uint productsK=0u,productsV=0u,decodesK=0u,decodesV=0u,expCount=0u;
#ifdef RT_LOCAL_PREFILL
            for(uint k=0u;k<LANE_AXES;k++){
                uint axis=c+k*32u;
                if(headActive&&axis<DH)localWork[subgroup*WORK_STRIDE+axis]=queries[row*A+head*DH+axis];
            }
            // loadClassTile's workgroup barrier publishes Q before member reads.
#endif
            for(uint microFirst=0u;microFirst<tokenCount;microFirst+=MICRO){
                uint n=min(MICRO,tokenCount-microFirst);
                loadClassTile(microFirst,n,kvBase,kvHeads,lane,false);
                if(groupFailure!=0u)return;
                for(uint j=0u;j<n;j++){
                    precise float sum=0.0,error=0.0;
                    if(headActive)for(uint k=0u;k<LANE_AXES;k++){
                        uint axis=c+k*32u,base=(kvLane*MICRO+j)*DH;
                        if(axis>=DH||classParent(classTile[base+axis])!=axis)continue;
                        decodesK++;
                        float value=addressTile[base+axis];if(value==0.0)continue;
                        precise float qsum=0.0,qerror=0.0;
                        uint member=axis,seen=0u,expected=classCount(classTile[base+axis]);
                        while(member<DH&&seen<DH){
#ifdef RT_LOCAL_PREFILL
                            accurateAdd(qsum,qerror,localWork[subgroup*WORK_STRIDE+member]);
#else
                            accurateAdd(qsum,qerror,queries[row*A+head*DH+member]);
#endif
                            member=classNext(classTile[base+member]);seen++;
                        }
                        if(seen!=expected||member!=DH){attentionFailure();continue;}
                        qsum-=qerror;
                        if(qsum!=0.0){
                            precise float product=qsum*value;
                            accurateAdd(sum,error,product);accurateAdd(sum,error,fma(qsum,value,-product));productsK++;
                        }
                    }
                    float total=subgroupAdd(sum-error),score=total/sqrt(float(DH));
                    if(headActive){
                        if(!finiteValue(score))attentionFailure();
                        uint index=microFirst+j;
                        if(c==index%32u){scores[index/32u]=score;maximum=max(maximum,score);}
                    }
                }
                barrier();if(groupFailure!=0u)return;
            }
            if(headActive)maximum=subgroupMax(maximum);
            precise float reference=0.0,referenceError=0.0;
            float previousWeight=0.0;uint previousScore=0u;bool havePrevious=false,reused=false;
            for(uint microFirst=0u;microFirst<tokenCount;microFirst+=MICRO){
                uint n=min(MICRO,tokenCount-microFirst);
                loadClassTile(microFirst,n,kvBase,kvHeads,lane,true);
                if(groupFailure!=0u)return;
                for(uint j=0u;j<n;j++){
                    uint index=microFirst+j,base=(kvLane*MICRO+j)*DH;
                    float score=subgroupShuffle(scores[index/32u],index%32u),weight=0.0;
                    if(headActive&&c==0u){
                        uint bits=floatBitsToUint(score);
                        if(havePrevious&&bits==previousScore){weight=previousWeight;reused=true;}
                        else{weight=exp(score-maximum);expCount++;}
                        previousWeight=weight;previousScore=bits;havePrevious=true;
                    }
                    weight=subgroupBroadcastFirst(weight)*float(tokenWeight[tokenList[index]]);
                    if(headActive)accurateAdd(reference,referenceError,weight);
                    precise float product[LANE_AXES],residual[LANE_AXES];
                    for(uint k=0u;k<LANE_AXES;k++){
                        uint axis=c+k*32u;product[k]=0.0;residual[k]=0.0;
                        if(headActive&&axis<DH&&classParent(classTile[base+axis])==axis){
                            decodesV++;float value=contentTile[base+axis];
                            if(value!=0.0){product[k]=weight*value;residual[k]=fma(weight,value,-product[k]);productsV++;}
                        }
                    }
#ifdef RT_LOCAL_PREFILL
                    for(uint k=0u;k<LANE_AXES;k++){
                        uint axis=c+k*32u;
                        if(axis<DH){localWork[subgroup*WORK_STRIDE+axis]=product[k];localWork[WORK_PLANE+subgroup*WORK_STRIDE+axis]=residual[k];}
                    }
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
                barrier();
            }
            productsK=subgroupAdd(productsK);productsV=subgroupAdd(productsV);
            decodesK=subgroupAdd(decodesK);decodesV=subgroupAdd(decodesV);
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
