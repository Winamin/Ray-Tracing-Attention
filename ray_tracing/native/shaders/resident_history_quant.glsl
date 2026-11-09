// One packet per token contains four Q4_K address and four Q6_K content blocks.
shared uint historyPacket[HISTORY_RECORD_BYTES/4u];
void packetByte(uint at,uint value){atomicOr(historyPacket[at/4u],(value&255u)<<((at%4u)*8u));}
void packetHalf(uint at,float value){uint halfIndex=packHalf2x16(vec2(value,0))&65535u;packetByte(at,halfIndex);packetByte(at+1u,halfIndex>>8u);}
float packetInput(uint row,uint compact,bool content){if(compact>=RT_GQA_WIDTH)return 0;uint axis=(compact/RT_ATTENTION_HEAD_DIM)*(RT_QUERY_WIDTH/RT_GQA_WIDTH)*RT_ATTENTION_HEAD_DIM+compact%RT_ATTENTION_HEAD_DIM;return content?feedForward[row*W+Q+axis]:keys[row*A+axis];}
const uint PACKET_BLOCKS=(RT_GQA_WIDTH+255u)/256u;
shared float groupScale[PACKET_BLOCKS*24u],groupMinimum[PACKET_BLOCKS*24u],blockScale[PACKET_BLOCKS*2u],blockMinimum[PACKET_BLOCKS];
shared uint scaleCode[PACKET_BLOCKS*24u],minimumCode[PACKET_BLOCKS*8u];
void packHistory(uint row,uint lane){
    for(uint i=lane;i<HISTORY_RECORD_BYTES/4u;i+=128u)historyPacket[i]=0u;barrier();
    for(uint job=lane;job<PACKET_BLOCKS*24u;job+=128u){
        bool content=job>=PACKET_BLOCKS*8u;uint local=content?job-PACKET_BLOCKS*8u:job;
        uint groups=content?16u:8u,block=local/groups,group=local%groups,width=content?16u:32u;
        float lo=0.0,hi=0.0;
        for(uint i=0u;i<width;i++){float v=packetInput(row,block*256u+group*width+i,content);if(content)hi=max(hi,abs(v));else{lo=min(lo,v);hi=max(hi,v);}}
        groupScale[job]=content?hi/31.0:(hi-lo)/15.0;groupMinimum[job]=content?0.0:-lo;
    }barrier();
    for(uint job=lane;job<PACKET_BLOCKS*2u;job+=128u){
        bool content=job>=PACKET_BLOCKS;uint block=job%PACKET_BLOCKS,groups=content?16u:8u,base=content?PACKET_BLOCKS*8u+block*16u:block*8u;
        float maxD=0.0,maxMin=0.0;for(uint g=0u;g<groups;g++){maxD=max(maxD,groupScale[base+g]);maxMin=max(maxMin,groupMinimum[base+g]);}
        float d=unpackHalf2x16(packHalf2x16(vec2(maxD/(content?127.0:63.0),0.0))).x;blockScale[job]=d;
        if(content)packetHalf(PACKET_BLOCKS*144u+block*210u+208u,d);
        else{float dm=unpackHalf2x16(packHalf2x16(vec2(maxMin/63.0,0.0))).x;blockMinimum[block]=dm;packetHalf(block*144u,d);packetHalf(block*144u+2u,dm);}
    }barrier();
    for(uint job=lane;job<PACKET_BLOCKS*24u;job+=128u){
        bool content=job>=PACKET_BLOCKS*8u;uint local=content?job-PACKET_BLOCKS*8u:job,groups=content?16u:8u,block=local/groups,group=local%groups;
        uint code=uint(clamp(round(groupScale[job]/max(blockScale[(content?PACKET_BLOCKS:0u)+block],1e-30)),0.0,content?127.0:63.0));scaleCode[job]=code;
        if(content)packetByte(PACKET_BLOCKS*144u+block*210u+192u+group,code);
        else minimumCode[job]=uint(clamp(round(groupMinimum[job]/max(blockMinimum[block],1e-30)),0.0,63.0));
    }barrier();
    for(uint block=lane;block<PACKET_BLOCKS;block+=128u){uint at=block*144u,base=block*8u;
        for(uint g=0u;g<4u;g++){packetByte(at+4u+g,scaleCode[base+g]|((scaleCode[base+g+4u]>>4u)<<6u));packetByte(at+8u+g,minimumCode[base+g]|((minimumCode[base+g+4u]>>4u)<<6u));packetByte(at+12u+g,(scaleCode[base+g+4u]&15u)|((minimumCode[base+g+4u]&15u)<<4u));}
    }
    for(uint job=lane;job<PACKET_BLOCKS*512u;job+=128u){
        bool content=job>=PACKET_BLOCKS*256u;uint local=content?job-PACKET_BLOCKS*256u:job,block=local/256u,i=local%256u;
        if(!content){uint group=block*8u+i/32u;float d=blockScale[block],dm=blockMinimum[block];uint q=uint(clamp(round((packetInput(row,block*256u+i,false)+dm*float(minimumCode[group]))/max(d*float(scaleCode[group]),1e-30)),0.0,15.0));packetByte(block*144u+16u+(i/64u)*32u+i%32u,q<<((i%64u)/32u*4u));}
        else{uint group=PACKET_BLOCKS*8u+block*16u+i/16u;float d=blockScale[PACKET_BLOCKS+block];uint q=uint(int(clamp(round(packetInput(row,block*256u+i,true)/max(d*float(scaleCode[group]),1e-30)),-32.0,31.0))+32),part=i/128u,within=i%128u,at=PACKET_BLOCKS*144u+block*210u;
            packetByte(at+part*64u+within%64u,(q&15u)<<((within/64u)*4u));packetByte(at+128u+part*32u+within%32u,(q>>4u)<<((within/32u)*2u));}
    }barrier();
    uint dest=keySlot(row)*(HISTORY_RECORD_BYTES/8u);for(uint i=lane;i<HISTORY_RECORD_BYTES/8u;i+=128u)keyGeometry[dest+i]=uvec2(historyPacket[i*2u],historyPacket[i*2u+1u]);
}
