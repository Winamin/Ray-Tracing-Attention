struct OrderedQ6Plan {
    uint qlWord,qlShift,qhWord,qhShift,scaleWord,scaleShift,dWord,dShift;
};
OrderedQ6Plan orderedQ6Plan(uint channel){
    uint compact=sharedAxis(channel),index=compact%256u;
    uint blockByte=((RT_GQA_WIDTH+255u)/256u)*144u+(compact/256u)*210u;
    uint part=index/128u,local=index%128u;
    uint qlByte=blockByte+part*64u+local%64u;
    uint qhByte=blockByte+128u+part*32u+local%32u;
    uint scaleByte=blockByte+192u+index/16u,dByte=blockByte+208u;
    OrderedQ6Plan plan;
    plan.qlWord=qlByte/4u;plan.qlShift=(qlByte%4u)*8u+(local/64u)*4u;
    plan.qhWord=qhByte/4u;plan.qhShift=(qhByte%4u)*8u+(local/32u)*2u;
    plan.scaleWord=scaleByte/4u;plan.scaleShift=(scaleByte%4u)*8u;
    plan.dWord=dByte/4u;plan.dShift=(dByte%4u)*8u;
    return plan;
}
float orderedQ6Coordinate(uint slot,OrderedQ6Plan plan){
    // HISTORY_RECORD_BYTES is eight-byte aligned. Flattening the existing
    // uvec2 view to uint preserves each packed word and its original bit fields.
    uint baseWord=(slot*HISTORY_RECORD_BYTES)/4u;
    uint ql=(orderedPackedHistory[baseWord+plan.qlWord]>>plan.qlShift)&15u;
    uint qh=(orderedPackedHistory[baseWord+plan.qhWord]>>plan.qhShift)&3u;
    int scale=int((orderedPackedHistory[baseWord+plan.scaleWord]>>plan.scaleShift)&255u);
    if(scale>=128)scale-=256;
    float d=unpackHalf2x16(orderedPackedHistory[baseWord+plan.dWord]>>plan.dShift).x;
    return (d*float(scale))*float(int(ql|(qh<<4u))-32);
}
