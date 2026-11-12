// Four history positions share restored K while each Q keeps its original
// member-chain order and lane reduction order. Scores temporarily occupy U.
const uint KEY_SLICES=4u;
shared uint classTile[KV_BATCH*MICRO*DH];
const uint WORK_STRIDE=DH+1u,WORK_PLANE=Q_BATCH*WORK_STRIDE;
shared float localWork[WORK_PLANE];

shared uint keyWork[Q_BATCH*KEY_SLICES*2u];
void loadClassTile(uint microFirst,uint n,uint kvBase,uint kvHeads,uint lane,bool content){
    for(uint i=lane;i<n*DH;i+=gl_WorkGroupSize.x){
        uint axis=i%DH,j=i/DH,slot=localSlots[tokenList[microFirst+j]],at=j*DH+axis;
        uint word=classWord(slot,kvBase,axis,content);classTile[at]=word;
        float value=classParent(word)==axis?pathCoordinate(slot,kvBase*Q_PER_KV*DH+axis,content):0.0;
        addressTile[at]=value;if(!finiteValue(value))attentionFailure();
    }
    barrier();
}
void keyAttention(uint row,uint group,uint lane){
    uint available=availablePositions(row),first=group*BLOCK;
    if(first>=available)return;
    uint count=min(BLOCK,available-first);
    traceGroup(row,group,count,lane);if(groupFailure!=0u)return;
    if(BLOCK>DH||!recordFits(recordIndex(row,group,headCount()-1u))){if(lane==0u)attentionFailure();return;}
    for(uint i=lane;i<count;i+=gl_WorkGroupSize.x)hits[row*(cfg.capacity+1u)+group*BLOCK+i+1u]=localSlots[i];
    if(lane==0u)attentionReceipts[recordIndex(row,group,0u)*RECORD_WORDS+6u]=rayInitializations;
    compactTokens(count,lane);
    uint c=gl_SubgroupInvocationID,subgroup=gl_SubgroupID,heads=headCount();
    uint kvHeads=(heads+Q_PER_KV-1u)/Q_PER_KV,qLane=subgroup%Q_BATCH,slice=subgroup/Q_BATCH;
    for(uint kvBase=0u;kvBase<kvHeads;kvBase++){

        for(uint headOffset=0u;headOffset<Q_PER_KV;headOffset+=Q_BATCH){
            uint head=kvBase*Q_PER_KV+headOffset+qLane;
            bool headActive=head<heads&&headOffset+qLane<Q_PER_KV;
            uint productsK=0u,decodesK=0u;
            if(slice==0u)for(uint k=0u;k<LANE_AXES;k++){
                uint axis=c+k*32u;
                if(headActive&&axis<DH)localWork[qLane*WORK_STRIDE+axis]=queries[row*A+head*DH+axis];
            }
            for(uint microFirst=0u;microFirst<tokenCount;microFirst+=MICRO){
                uint n=min(MICRO,tokenCount-microFirst);
                loadClassTile(microFirst,n,kvBase,kvHeads,lane,false);
                if(groupFailure!=0u)return;
                precise float sum=0.0,error=0.0;
                if(headActive&&slice<n)for(uint k=0u;k<LANE_AXES;k++){
                    uint axis=c+k*32u,base=slice*DH;
                    if(axis>=DH||classParent(classTile[base+axis])!=axis)continue;
                    decodesK++;
                    float value=addressTile[base+axis];if(value==0.0)continue;
                    precise float qsum=0.0,qerror=0.0;
                    uint member=axis,seen=0u,expected=classCount(classTile[base+axis]);
                    while(member<DH&&seen<DH){
                        accurateAdd(qsum,qerror,localWork[qLane*WORK_STRIDE+member]);
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
                if(headActive&&slice<n){
                    if(!finiteValue(score))attentionFailure();
                    uint index=microFirst+slice;
                    if(c==index%32u)attentionReceipts[recordIndex(row,group,head)*RECORD_WORDS+U_OFFSET+index]=floatBitsToUint(score);
                }
                barrier();if(groupFailure!=0u)return;
            }
            productsK=subgroupAdd(productsK);decodesK=subgroupAdd(decodesK);
            if(c==0u){keyWork[subgroup*2u]=productsK;keyWork[subgroup*2u+1u]=decodesK;}
            barrier();
            if(slice==0u&&headActive&&c==0u){
                productsK=0u;decodesK=0u;
                for(uint p=0u;p<KEY_SLICES;p++){productsK+=keyWork[(p*Q_BATCH+qLane)*2u];decodesK+=keyWork[(p*Q_BATCH+qLane)*2u+1u];}
                uint at=recordIndex(row,group,head)*RECORD_WORDS;
                attentionReceipts[at+11u+COVERAGE_WORDS]=productsK;attentionReceipts[at+12u+COVERAGE_WORDS]=decodesK;
            }
            barrier();if(groupFailure!=0u)return;
        }
    }
}
void localAttention(uint row,uint group,uint lane){}
