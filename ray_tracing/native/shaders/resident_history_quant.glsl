// One packet per token contains four Q4_K address and four Q6_K content blocks.
shared uint historyPacket[HISTORY_RECORD_BYTES/4u];
void packetByte(uint at,uint value){atomicOr(historyPacket[at/4u],(value&255u)<<((at%4u)*8u));}
void packetHalf(uint at,float value){uint halfIndex=packHalf2x16(vec2(value,0))&65535u;packetByte(at,halfIndex);packetByte(at+1u,halfIndex>>8u);}
float packetInput(uint row,uint compact,bool content){if(compact>=RT_GQA_WIDTH)return 0;uint axis=(compact/RT_ATTENTION_HEAD_DIM)*(RT_QUERY_WIDTH/RT_GQA_WIDTH)*RT_ATTENTION_HEAD_DIM+compact%RT_ATTENTION_HEAD_DIM;return content?feedForward[row*W+Q+axis]:keys[row*A+axis];}
void packHistory(uint row,uint lane){
    for(uint i=lane;i<HISTORY_RECORD_BYTES/4u;i+=128u)historyPacket[i]=0u;barrier();
    const uint BLOCKS=(RT_GQA_WIDTH+255u)/256u;
    if(lane<BLOCKS*2u){bool content=lane>=BLOCKS;uint block=lane%BLOCKS,first=block*256u;
        if(!content){
            float ds[8],mins[8],maxD=0,maxMin=0;uint scales[8],minimums[8];
            for(uint g=0u;g<8u;g++){float lo=0,hi=0;for(uint j=0u;j<32u;j++){float v=packetInput(row,first+g*32u+j,false);lo=min(lo,v);hi=max(hi,v);}ds[g]=(hi-lo)/15.0;mins[g]=-lo;maxD=max(maxD,ds[g]);maxMin=max(maxMin,mins[g]);}
            float d=unpackHalf2x16(packHalf2x16(vec2(maxD/63.0,0))).x,dm=unpackHalf2x16(packHalf2x16(vec2(maxMin/63.0,0))).x;uint at=block*144u;
            packetHalf(at,d);packetHalf(at+2u,dm);
            for(uint g=0u;g<8u;g++){scales[g]=uint(clamp(round(ds[g]/max(d,1e-30)),0.0,63.0));minimums[g]=uint(clamp(round(mins[g]/max(dm,1e-30)),0.0,63.0));}
            for(uint g=0u;g<4u;g++){packetByte(at+4u+g,scales[g]|((scales[g+4u]>>4u)<<6u));packetByte(at+8u+g,minimums[g]|((minimums[g+4u]>>4u)<<6u));packetByte(at+12u+g,(scales[g+4u]&15u)|((minimums[g+4u]&15u)<<4u));}
            for(uint i=0u;i<256u;i++){uint g=i/32u,q=uint(clamp(round((packetInput(row,first+i,false)+dm*float(minimums[g]))/max(d*float(scales[g]),1e-30)),0.0,15.0));packetByte(at+16u+(i/64u)*32u+i%32u,q<<((i%64u)/32u*4u));}
        }else{
            float ds[16],maxD=0;uint scales[16];
            for(uint g=0u;g<16u;g++){float mx=0;for(uint j=0u;j<16u;j++)mx=max(mx,abs(packetInput(row,first+g*16u+j,true)));ds[g]=mx/31.0;maxD=max(maxD,ds[g]);}
            float d=unpackHalf2x16(packHalf2x16(vec2(maxD/127.0,0))).x;uint at=BLOCKS*144u+block*210u;packetHalf(at+208u,d);
            for(uint g=0u;g<16u;g++){scales[g]=uint(clamp(round(ds[g]/max(d,1e-30)),0.0,127.0));packetByte(at+192u+g,scales[g]);}
            for(uint i=0u;i<256u;i++){uint q=uint(int(clamp(round(packetInput(row,first+i,true)/max(d*float(scales[i/16u]),1e-30)),-32.0,31.0))+32),halfIndex=i/128u,local=i%128u;
                packetByte(at+halfIndex*64u+local%64u,(q&15u)<<((local/64u)*4u));packetByte(at+128u+halfIndex*32u+local%32u,(q>>4u)<<((local/32u)*2u));}
        }
    }barrier();
    uint dest=keySlot(row)*(HISTORY_RECORD_BYTES/8u);for(uint i=lane;i<HISTORY_RECORD_BYTES/8u;i+=128u)keyGeometry[dest+i]=uvec2(historyPacket[i*2u],historyPacket[i*2u+1u]);
}
