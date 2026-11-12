bool finiteValue(float value){return !isnan(value)&&!isinf(value);}
void accurateAdd(inout float sum,inout float correction,float value){
    precise float adjusted=value-correction,next=sum+adjusted;
    correction=(next-sum)-adjusted;sum=next;
}
void attentionFailure(){atomicOr(groupFailure,1u);atomicOr(control[3],ATTENTION_ERROR);}
void statisticAdd(uint metric,uint value){
    if((cfg.debug&65536u)==0u)return;
    uint at=11u+cfg.capacity+65536u+parameters[blockBase()+41u]*32u+metric*2u;
    uint old=atomicAdd(control[at],value);if(old+value<old)atomicAdd(control[at+1u],1u);
}
float finalHeadSignal(uint row,uint channel,float signal,float reference){
    precise float result=signal/max(reference,1e-30);
    if(RT_ATTENTION_GATE==1){
        float gate=feedForward[row*W+channel];
        if(!finiteValue(gate)){attentionFailure();return 0.0;}
        result*=detectorSigmoid(gate);
    }
    if(!finiteValue(result)){attentionFailure();return 0.0;}
    return result;
}
uint availablePositions(uint row){
    uint time=absolute(row);
    return time<cfg.checkpoint?0u:min(time-cfg.checkpoint+1u,cfg.capacity);
}
uint groupCount(uint row){return (availablePositions(row)+BLOCK-1u)/BLOCK;}
uint partialGroupSum(uint n){
    uint q=n/BLOCK,r=n%BLOCK;
    return BLOCK*q*(q+1u)/2u+r*(q+1u);
}
uint rowGroupPrefix(uint row){
    if(cfg.mode==1u)return row*groupCount(0u);
    uint time0=absolute(0u),start=time0-cfg.checkpoint;
    uint uncapped=min(row,cfg.capacity>start?cfg.capacity-start:0u);
    uint offset=start%BLOCK;
    return uncapped*(start/BLOCK)+partialGroupSum(offset+uncapped)-partialGroupSum(offset)
        +(row-uncapped)*((cfg.capacity+BLOCK-1u)/BLOCK);
}
uint recordIndex(uint row,uint group,uint head){
    return (rowGroupPrefix(row)+group)*headCount()+head;
}
bool recordFits(uint record){return record<uint(attentionReceipts.length())/RECORD_WORDS;}
