shared uint groupMissSlots[BLOCK],groupMissCount;
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

// A K workgroup owns disjoint historical slots and finishes every KV/head
// before publishing their stamps. Local RT's shared gaps need no global queue
// or a later V-side scan; the native K-to-V barrier publishes these receipts.
void finishPathScoreGroup(uint row,uint count,uint lane){
    if(row!=0u||!pathTargetReady(0u))return;
    memoryBarrierBuffer();barrier();
    if(groupFailure!=0u)return;
    bool sameQuery=pathScoreWork[3u]==1u;
    uint filled=sameQuery?groupMissCount:count;
    for(uint i=lane;i<filled;i+=gl_WorkGroupSize.x)
        publishPathScoreStamp(sameQuery?groupMissSlots[i]:localSlots[i]);
}

void startReceiptLens(vec3 origin,vec3 direction){
    atomicAdd(rayInitializations,1u);
    rayQueryInitializeEXT(receiptRay,lensAS[0],gl_RayFlagsOpaqueEXT,1u,origin,.00001,direction,8.0);
}
vec3 receiptLensPoint(uint at){return vec3(lensVertices[at],lensVertices[at+1u],lensVertices[at+2u]);}
bool throughReceiptLens(uint row,out vec3 origin,out vec3 direction){
    vec4 emission=rays[row*rayStride()];
    if(any(isnan(emission))||any(isinf(emission)))return false;
    origin=vec3(emission.xy+vec2(float(cfg.layer)*4.0,0.0),0.0);
    direction=normalize(vec3(emission.zw,1.0));
    if(any(isnan(direction))||any(isinf(direction)))return false;
    uint parameterBase=parameters[blockBase()+5u],instance=lensIndex(cfg.layer,0u);
    float eta=1.0+.02/(1.0+exp(-pf(parameterBase+6u)));
    for(uint face=0u;face<2u;face++){
        startReceiptLens(origin,direction);
        while(rayQueryProceedEXT(receiptRay)){}
        if(rayQueryGetIntersectionTypeEXT(receiptRay,true)!=gl_RayQueryCommittedIntersectionTriangleEXT)return false;
        uint primitive=rayQueryGetIntersectionPrimitiveIndexEXT(receiptRay,true);
        uint owner=rayQueryGetIntersectionInstanceCustomIndexEXT(receiptRay,true);
        if(owner!=cfg.layer*B||primitive/2u!=face)return false;
        uint at=(instance*12u+primitive*3u)*3u;
        vec3 normal=normalize(cross(receiptLensPoint(at+3u)-receiptLensPoint(at),receiptLensPoint(at+6u)-receiptLensPoint(at)));
        if(dot(normal,direction)>0.0)normal=-normal;
        origin+=rayQueryGetIntersectionTEXT(receiptRay,true)*direction;
        vec3 next=refract(direction,normal,face==0u?1.0/eta:eta);
        if(dot(next,next)<.5)return false;
        direction=normalize(next);origin+=.00001*direction;
        if(any(isnan(direction))||any(isinf(direction))||any(isnan(origin))||any(isinf(origin)))return false;
    }
    return direction.z>0.0&&finiteValue(direction.z);
}

#ifndef RT_PREFILL_VALUE
void traceGroup(uint row,uint group,uint count,uint lane){
    uint first=group*BLOCK,time=absolute(row);
    for(uint i=lane;i<BLOCK;i+=gl_WorkGroupSize.x)localSlots[i]=0xffffffffu;
    for(uint i=lane;i<COVERAGE_WORDS;i+=gl_WorkGroupSize.x)coverage[i]=0u;
    if(lane==0u){groupFailure=0u;uniqueHits=0u;rayInitializations=0u;groupMissCount=0u;}
    barrier();
    if(lane==0u){
        vec3 origin,direction;
        if(!throughReceiptLens(row,origin,direction))attentionFailure();
        historyOrigin=origin;historyDirection=direction;
    }
    barrier();
    if(groupFailure!=0u)return;
    beginPathScoreTraversal(row);
    uint subrays=(count+15u)/16u;
    if(lane<subrays){
        uint rayFirst=first+lane*16u,rayCount=min(16u,count-lane*16u);
        vec3 origin=historyOrigin,direction=historyDirection;
        origin.z-=4.0*float(time)+2.0;
        origin+=direction*(4.0*float(rayFirst)/direction.z);
        atomicAdd(rayInitializations,1u);
        rayQueryInitializeEXT(receiptRay,historyAS[0],gl_RayFlagsNoOpaqueEXT|gl_RayFlagsCullBackFacingTrianglesEXT,
            1u,origin,.00001,direction,4.0*float(rayCount)/direction.z);
        while(rayQueryProceedEXT(receiptRay)){
            if(rayQueryGetIntersectionTypeEXT(receiptRay,false)!=gl_RayQueryCandidateIntersectionTriangleEXT)continue;
            uint primitive=rayQueryGetIntersectionPrimitiveIndexEXT(receiptRay,false);
            uint owner=rayQueryGetIntersectionInstanceCustomIndexEXT(receiptRay,false);
            uint slot=(owner&RT_SLOT_MASK)+primitive/2u;
            if(primitive>=32u||slot>=cfg.capacity){attentionFailure();continue;}
            // historyAS[0] is bound to this layer's fullIndex() TLAS. Only the
            // slot identity is carried in its instanceCustomIndex.
            uint global=parameters[blockBase()+41u]*cfg.capacity+slot;
            if(global>=uint(slots.length())){attentionFailure();continue;}
            uint age=slots[global].x;
            if(age>time||age<cfg.checkpoint||time-age>=cfg.capacity||slot!=age%cfg.capacity){attentionFailure();continue;}
            uint relative=time-age;
            if(relative<rayFirst||relative>=rayFirst+rayCount){attentionFailure();continue;}
            uint local=relative-first,word=local/32u,bit=1u<<(local%32u);
            if((atomicOr(coverage[word],bit)&bit)!=0u)continue;
            bool cached=publishPathScores(row,global);
            if(pathTraversalSame&&!cached){
                uint missing=atomicAdd(groupMissCount,1u);
                if(missing<BLOCK)groupMissSlots[missing]=global;
                else attentionFailure();
            }
            localSlots[local]=global;atomicAdd(uniqueHits,1u);
            uint old=atomicOr(slots[global].y,1u);
            if((old&3u)==0u){uint queue=atomicAdd(control[6],1u);control[8u+queue]=global;}
        }
    }
    memoryBarrierBuffer();
    barrier();
    for(uint word=lane;word<COVERAGE_WORDS;word+=gl_WorkGroupSize.x){
        uint firstBit=word*32u,n=firstBit<count?min(32u,count-firstBit):0u;
        uint expected=n==32u?0xffffffffu:n==0u?0u:(1u<<n)-1u;
        if(coverage[word]!=expected)attentionFailure();
    }
    if(lane==0u){
        if(uniqueHits!=count||groupMissCount>count)attentionFailure();
        atomicAdd(hits[row*(cfg.capacity+1u)],uniqueHits);
    }
    barrier();
}

#else
// The preceding K dispatch completed and validated these RT hit slots. The
// command-buffer barrier publishes hits and ray counts before this pass starts.
void traceGroup(uint row,uint group,uint count,uint lane){
    if(lane==0u){groupFailure=0u;uniqueHits=count;rayInitializations=attentionReceipts[recordIndex(row,group,0u)*RECORD_WORDS+6u];}
    for(uint i=lane;i<BLOCK;i+=gl_WorkGroupSize.x)localSlots[i]=i<count?hits[row*(cfg.capacity+1u)+group*BLOCK+i+1u]:0xffffffffu;
    for(uint word=lane;word<COVERAGE_WORDS;word+=gl_WorkGroupSize.x){
        uint firstBit=word*32u,n=firstBit<count?min(32u,count-firstBit):0u;
        coverage[word]=n==32u?0xffffffffu:n==0u?0u:(1u<<n)-1u;
    }
    barrier();
}
#endif
