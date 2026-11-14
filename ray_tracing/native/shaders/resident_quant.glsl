uint coefficientWord(uint kind,uint at);
uint coefficient16(uint kind,uint at);
// Original GGUF blocks stay packed; only the addressed coefficient is decoded.
uint coefficientByte(uint kind,uint at){uint word=kind==0u?matrices[cfg.layer].values[at/4u]:kind==1u?basis[at/4u]:keyGeometry[at/8u][(at/4u)%2u];return (word>>((at%4u)*8u))&255u;}
float coefficientHalf(uint kind,uint at){uint i=at/4u,word=kind==0u?matrices[cfg.layer].values[i]:kind==1u?basis[i]:keyGeometry[i/2u][i%2u];return unpackHalf2x16(word>>((at%4u)*8u)).x;}
float ggufValue(uint bufferKind,uint at,uint type,uint index){
    if(type==0u)return uintBitsToFloat(coefficientWord(bufferKind,at+index*4u));
    if(type==1u)return coefficientHalf(bufferKind,at+index*2u);
    if(type==30u)return uintBitsToFloat(coefficient16(bufferKind,at+index*2u)<<16u);
    if(type==12u){uint group=index/32u,sc,minval;
        if(group<4u){sc=coefficientByte(bufferKind,at+4u+group)&63u;minval=coefficientByte(bufferKind,at+8u+group)&63u;}
        else {uint p=group-4u,mixed=coefficientByte(bufferKind,at+12u+p);sc=(mixed&15u)|((coefficientByte(bufferKind,at+4u+p)>>6u)<<4u);minval=(mixed>>4u)|((coefficientByte(bufferKind,at+8u+p)>>6u)<<4u);}
        uint packed=coefficientByte(bufferKind,at+16u+(index/64u)*32u+index%32u),q=(packed>>((index%64u)/32u*4u))&15u;
        return (coefficientHalf(bufferKind,at)*float(sc))*float(q)-coefficientHalf(bufferKind,at+2u)*float(minval);
    }
    uint halfIndex=index/128u,local=index%128u,ql=(coefficientByte(bufferKind,at+halfIndex*64u+local%64u)>>((local/64u)*4u))&15u;
    uint qh=(coefficientByte(bufferKind,at+128u+halfIndex*32u+local%32u)>>((local/32u)*2u))&3u;
    int scale=int(coefficientByte(bufferKind,at+192u+index/16u));if(scale>=128)scale-=256;
    return (coefficientHalf(bufferKind,at+208u)*float(scale))*float(int(ql|(qh<<4u))-32);
}
uint originalRotaryAxis(uint axis){return axis<RT_ROTARY_DIM?(axis%2u)*(RT_ROTARY_DIM/2u)+axis/2u:axis;}
float quantizedWeight(uint bufferKind,uint offset,uint metadata,uint row,uint cols,uint axis){
    uint type=metadata&255u,mode=metadata>>8u;
    if(mode==1u||mode==2u)row=(row/RT_ATTENTION_HEAD_DIM)*(2u*RT_ATTENTION_HEAD_DIM)+(mode==2u?RT_ATTENTION_HEAD_DIM:0u)+originalRotaryAxis(row%RT_ATTENTION_HEAD_DIM);
    else if(mode==3u)row=(row/RT_ATTENTION_HEAD_DIM)*RT_ATTENTION_HEAD_DIM+originalRotaryAxis(row%RT_ATTENTION_HEAD_DIM);
    else if(mode==4u)axis=(axis/RT_ATTENTION_HEAD_DIM)*RT_ATTENTION_HEAD_DIM+originalRotaryAxis(axis%RT_ATTENTION_HEAD_DIM);
    else if(mode==5u)row=row%(RT_STATE_KEY_HEADS*128u);
    uint linear=row*cols+axis,bytes=type==12u?144u:210u;
    if(type!=12u&&type!=14u)return ggufValue(bufferKind,offset*4u,type,linear);
    return ggufValue(bufferKind,offset*4u+(linear/256u)*bytes,type,linear%256u);
}
uint coefficient16(uint kind,uint at){uint i=at/4u,word=kind==0u?matrices[cfg.layer].values[i]:kind==1u?basis[i]:keyGeometry[i/2u][i%2u];return (word>>((at%4u)*8u))&65535u;}
uint coefficientWord(uint kind,uint at){
    uint i=at/4u,word=kind==0u?matrices[cfg.layer].values[i]:kind==1u?basis[i]:keyGeometry[i/2u][i%2u];
    if(at%4u==0u)return word;
    uint next=kind==0u?matrices[cfg.layer].values[i+1u]:kind==1u?basis[i+1u]:keyGeometry[(i+1u)/2u][(i+1u)%2u];
    return (word>>16u)|(next<<16u);
}
uvec2 q4ScaleMin(uint kind,uint at,uint group){
    if(group<4u)return uvec2(coefficientByte(kind,at+4u+group),coefficientByte(kind,at+8u+group))&63u;
    uint p=group-4u,mixed=coefficientByte(kind,at+12u+p);
    return uvec2((mixed&15u)|((coefficientByte(kind,at+4u+p)>>6u)<<4u),(mixed>>4u)|((coefficientByte(kind,at+8u+p)>>6u)<<4u));
}
// One subgroup covers a 256-coefficient GGUF block. Each lane decodes
// four packed bytes into eight transfers, sharing their header and scales.
struct QuantizedOctet {uvec2 axis;vec4 low;vec4 high;};
QuantizedOctet quantizedOctetTyped(uint kind,uint offset,uint metadata,uint row,uint cols,uint block,uint thread){
    uint type=metadata&255u,mode=metadata>>8u;
    uint first=type==12u?(thread/8u)*64u+(thread%8u)*4u:(thread/16u)*128u+(thread%16u)*4u;
    uint second=first+(type==12u?32u:64u);QuantizedOctet result;
    result.axis=block*256u+uvec2(first,second);
    if(type!=12u&&type!=14u||mode==4u&&(result.axis.x%RT_ATTENTION_HEAD_DIM<RT_ROTARY_DIM||result.axis.y%RT_ATTENTION_HEAD_DIM<RT_ROTARY_DIM)){
        for(uint i=0u;i<4u;i++){result.low[i]=quantizedWeight(kind,offset,metadata,row,cols,result.axis.x+i);result.high[i]=quantizedWeight(kind,offset,metadata,row,cols,result.axis.y+i);}return result;
    }
    if(mode==1u||mode==2u)row=(row/RT_ATTENTION_HEAD_DIM)*(2u*RT_ATTENTION_HEAD_DIM)+(mode==2u?RT_ATTENTION_HEAD_DIM:0u)+originalRotaryAxis(row%RT_ATTENTION_HEAD_DIM);
    else if(mode==3u)row=(row/RT_ATTENTION_HEAD_DIM)*RT_ATTENTION_HEAD_DIM+originalRotaryAxis(row%RT_ATTENTION_HEAD_DIM);
    else if(mode==5u)row=row%(RT_STATE_KEY_HEADS*128u);
    uint at=offset*4u+(row*(cols/256u)+block)*(type==12u?144u:210u);
    if(type==12u){
        vec2 header=unpackHalf2x16(coefficientWord(kind,at));
        // Decode both scale/minimum pairs from the same three packed words.
        // Adjacent groups share their metadata, with no per-byte rereads.
        uint group=first/32u;uvec2 shift=uvec2(group&3u,(group&3u)+1u)*8u;
        uvec2 sc=(uvec2(coefficientWord(kind,at+4u))>>shift)&255u;
        uvec2 mn=(uvec2(coefficientWord(kind,at+8u))>>shift)&255u;
        uvec2 extra=(uvec2(coefficientWord(kind,at+12u))>>shift)&255u;
        sc=mix(sc&63u,(extra&15u)|((sc>>6u)<<4u),bvec2(group>=4u));
        mn=mix(mn&63u,(extra>>4u)|((mn>>6u)<<4u),bvec2(group>=4u));
        uint packed=coefficientWord(kind,at+16u+(first/64u)*32u+first%32u);
        vec4 qLow=vec4(bitfieldExtract(packed,0,4),bitfieldExtract(packed,8,4),bitfieldExtract(packed,16,4),bitfieldExtract(packed,24,4));
        vec4 qHigh=vec4(bitfieldExtract(packed,4,4),bitfieldExtract(packed,12,4),bitfieldExtract(packed,20,4),bitfieldExtract(packed,28,4));
        result.low=(header.x*float(sc.x))*qLow-(header.y*float(mn.x));
        result.high=(header.x*float(sc.y))*qHigh-(header.y*float(mn.y));
    }else{
        uint local=first%128u,part=first/128u;
        uint low=coefficientWord(kind,at+part*64u+local);
        uint high=coefficientWord(kind,at+128u+part*32u+local%32u);
        uint shift=(local/32u)*2u;
        int s0=bitfieldExtract(int(coefficientByte(kind,at+192u+first/16u)),0,8),s1=bitfieldExtract(int(coefficientByte(kind,at+192u+second/16u)),0,8);
        float scale=coefficientHalf(kind,at+208u);
        // Merge four Q6 coefficients while they are still packed, following the
        // word masks in llama.cpp's Vulkan Q6_K matvec. Decode each byte once.
        uint packedLow=(low&0x0f0f0f0fu)|(((high>>shift)&0x03030303u)<<4u);
        uint packedHigh=((low>>4u)&0x0f0f0f0fu)|(((high>>(shift+4u))&0x03030303u)<<4u);
        // Flipping the sign bit makes signed six-bit extraction return q-32.
        int centeredLow=int(packedLow^0x20202020u),centeredHigh=int(packedHigh^0x20202020u);
        vec4 qLow=vec4(bitfieldExtract(centeredLow,0,6),bitfieldExtract(centeredLow,8,6),bitfieldExtract(centeredLow,16,6),bitfieldExtract(centeredLow,24,6));
        vec4 qHigh=vec4(bitfieldExtract(centeredHigh,0,6),bitfieldExtract(centeredHigh,8,6),bitfieldExtract(centeredHigh,16,6),bitfieldExtract(centeredHigh,24,6));
        result.low=(scale*float(s0))*qLow;
        result.high=(scale*float(s1))*qHigh;
    }
    return result;
}
QuantizedOctet quantizedOctet(uint kind,uint offset,uint metadata,uint row,uint cols,uint block,uint thread){
    return quantizedOctetTyped(kind,offset,coefficientFormat!=0xffffffffu?coefficientFormat:metadata,row,cols,block,thread);
}
// Adjacent coefficients share block scales, header and packed word loads.
vec2 ggufPair(uint kind,uint at,uint type,uint index){
    if(type==12u){uint group=index/32u,sc,mn;
        if(group<4u){sc=coefficientByte(kind,at+4u+group)&63u;mn=coefficientByte(kind,at+8u+group)&63u;}
        else{uint p=group-4u,mixed=coefficientByte(kind,at+12u+p);sc=(mixed&15u)|((coefficientByte(kind,at+4u+p)>>6u)<<4u);mn=(mixed>>4u)|((coefficientByte(kind,at+8u+p)>>6u)<<4u);}
        vec2 header=vec2(coefficientHalf(kind,at),coefficientHalf(kind,at+2u));uint shift=(index%64u)/32u*4u,packed=coefficient16(kind,at+16u+(index/64u)*32u+index%32u);
        uvec2 q=uvec2((packed>>shift)&15u,(packed>>(shift+8u))&15u);
        return (header.x*float(sc))*vec2(q)-(header.y*float(mn));
    }
    uint part=index/128u,local=index%128u,ql=coefficient16(kind,at+part*64u+local%64u),qh=coefficient16(kind,at+128u+part*32u+local%32u);
    uint lowShift=local/64u*4u,highShift=local/32u*2u;
    uvec2 low=uvec2((ql>>lowShift)&15u,(ql>>(lowShift+8u))&15u),high=uvec2((qh>>highShift)&3u,(qh>>(highShift+8u))&3u);
    int scale=int(coefficientByte(kind,at+192u+index/16u));if(scale>=128)scale-=256;
    return (coefficientHalf(kind,at+208u)*float(scale))*vec2(ivec2(low|(high<<4u))-32);
}
vec2 packedTransfer(uint kind,uint offset,uint metadata,uint row,uint cols,uint pair){
    if(coefficientFormat!=0xffffffffu)metadata=coefficientFormat;
    uint type=metadata&255u,mode=metadata>>8u,axis=pair*2u;
    if(type!=12u&&type!=14u||mode==4u&&axis%RT_ATTENTION_HEAD_DIM<RT_ROTARY_DIM)return vec2(quantizedWeight(kind,offset,metadata,row,cols,axis),quantizedWeight(kind,offset,metadata,row,cols,axis+1u));
    if(mode==1u||mode==2u)row=(row/RT_ATTENTION_HEAD_DIM)*(2u*RT_ATTENTION_HEAD_DIM)+(mode==2u?RT_ATTENTION_HEAD_DIM:0u)+originalRotaryAxis(row%RT_ATTENTION_HEAD_DIM);
    else if(mode==3u)row=(row/RT_ATTENTION_HEAD_DIM)*RT_ATTENTION_HEAD_DIM+originalRotaryAxis(row%RT_ATTENTION_HEAD_DIM);
    else if(mode==5u)row=row%(RT_STATE_KEY_HEADS*128u);
    uint linear=row*cols+axis;return ggufPair(kind,offset*4u+(linear/256u)*(type==12u?144u:210u),type,linear%256u);
}
vec2 quantizedTransfer(uint offset,uint metadata,uint row,uint cols,uint pair){return packedTransfer(0u,offset,metadata,row,cols,pair);}
vec2 outputTransfer(uint word,uint pair){return packedTransfer(1u,parameters[16u],parameters[15u],word,D,pair);}
float vocabularyValue(uint word,uint axis){return quantizedWeight(1u,parameters[16u],parameters[15u],word,D,axis);}
float tokenEmbedding(uint word,uint axis){return quantizedWeight(1u,parameters[14u],parameters[13u],word,D,axis);}
const uint HISTORY_RECORD_BYTES=(((RT_GQA_WIDTH+255u)/256u)*(144u+210u)+7u)/8u*8u;
float quantizedPath(uint slot,uint axis,bool content){uint c=sharedAxis(axis),at=slot*HISTORY_RECORD_BYTES+(content?((RT_GQA_WIDTH+255u)/256u)*144u:0u),type=content?14u:12u;return ggufValue(2u,at+(c/256u)*(content?210u:144u),type,c%256u);}
