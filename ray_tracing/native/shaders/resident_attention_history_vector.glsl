// Four adjacent historical coefficients share packed words, block headers
// and scales, as in llama.cpp's Vulkan vector dequantization callbacks.
// Decode the original Q4_K/Q6_K packet in FP32 before the caller converts to
// its existing matrix operand format. No history copy or new cache is needed.
uint historyVectorWord(const in HistoryRecord record,uint byte){
    uint word=record.raw[byte/4u];
    return byte%4u==0u?word:(word>>16u)|(record.raw[byte/4u+1u]<<16u);
}
uint historyVectorByte(const in HistoryRecord record,uint byte){
    return (record.raw[byte/4u]>>((byte%4u)*8u))&255u;
}
float historyVectorHalf(const in HistoryRecord record,uint byte){
    return unpackHalf2x16(record.raw[byte/4u]>>((byte%4u)*8u)).x;
}
vec4 historyVector(const in HistoryRecord record,uint channel,bool content){
    uint compact=sharedAxis(channel),index=compact%256u;
    uint byte=(content?((RT_GQA_WIDTH+255u)/256u)*144u:0u)
        +(compact/256u)*(content?210u:144u);
    if(!content){
        uint group=index/32u,scale,minimum;
        if(group<4u){
            scale=historyVectorByte(record,byte+4u+group)&63u;
            minimum=historyVectorByte(record,byte+8u+group)&63u;
        }else{
            uint p=group-4u,mixed=historyVectorByte(record,byte+12u+p);
            scale=(mixed&15u)|((historyVectorByte(record,byte+4u+p)>>6u)<<4u);
            minimum=(mixed>>4u)|((historyVectorByte(record,byte+8u+p)>>6u)<<4u);
        }
        vec2 header=unpackHalf2x16(historyVectorWord(record,byte));
        uint packed=historyVectorWord(record,byte+16u+(index/64u)*32u+index%32u);
        uvec4 q=(uvec4(packed)>>uvec4(0u,8u,16u,24u)>>((index%64u)/32u*4u))&15u;
        return (header.x*float(scale))*vec4(q)-header.y*float(minimum);
    }
    uint part=index/128u,local=index%128u;
    uint low=historyVectorWord(record,byte+part*64u+local%64u);
    uint high=historyVectorWord(record,byte+128u+part*32u+local%32u);
    uvec4 ql=(uvec4(low)>>uvec4(0u,8u,16u,24u)>>((local/64u)*4u))&15u;
    uvec4 qh=(uvec4(high)>>uvec4(0u,8u,16u,24u)>>((local/32u)*2u))&3u;
    int scale=int(historyVectorByte(record,byte+192u+index/16u));
    if(scale>=128)scale-=256;
    float d=historyVectorHalf(record,byte+208u);
    return (d*float(scale))*vec4(ivec4(ql|(qh<<4u))-32);
}
