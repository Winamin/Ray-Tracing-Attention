// Restore the eight coordinates owned by one lane of a 256D QK dot.
// Header and scale words are loaded once per original GGUF block. Each
// coefficient keeps ggufValue's FP32 multiplication/subtraction order.
void detectorKeyOctet(uint slot,uint channel,uint lane,out vec4 even,out vec4 odd){
    uint compact=sharedAxis(channel),at=slot*HISTORY_RECORD_BYTES+(compact/256u)*144u;
    vec2 header=unpackHalf2x16(coefficientWord(2u,at));
    uint scales=coefficientWord(2u,at+4u),minima=coefficientWord(2u,at+8u),extra=coefficientWord(2u,at+12u);
    for(uint part=0u;part<4u;part++){
        uint group=part*2u,shift=(group&3u)*8u;
        uvec2 sc=(uvec2(scales)>>uvec2(shift,shift+8u))&255u;
        uvec2 mn=(uvec2(minima)>>uvec2(shift,shift+8u))&255u;
        if(group<4u){sc&=63u;mn&=63u;}
        else{
            uvec2 mixed=(uvec2(extra)>>uvec2(shift,shift+8u))&255u;
            sc=(mixed&15u)|((sc>>6u)<<4u);mn=(mixed>>4u)|((mn>>6u)<<4u);
        }
        uint packed=coefficientByte(2u,at+16u+part*32u+lane);
        even[part]=(header.x*float(sc.x))*float(packed&15u)-header.y*float(mn.x);
        odd[part]=(header.x*float(sc.y))*float(packed>>4u)-header.y*float(mn.y);
    }
}
