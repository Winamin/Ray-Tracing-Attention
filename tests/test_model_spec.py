from ray_tracing.compiler.spec import ModelSpec

def test_dimensions_and_layer_map():
    spec=ModelSpec('qwen35',256,512,37,('full','recurrent','full'),4,1,128,128,state_heads=4,state_key_heads=2).validate()
    assert spec.full_layers==(0,2) and spec.recurrent_layers==(1,)
    assert spec.query_width==512 and spec.working_width==512
    assert spec.feed_forward_stride==1024 and spec.history_bytes==360
    assert ModelSpec.from_dict(spec.to_dict()).fingerprint==spec.fingerprint

def test_distinct_native_configs():
    a=ModelSpec('llama',256,512,37,('full',),8,2,32,32,attention_gate=False,qk_norm=False).validate()
    b=ModelSpec.from_dict({**a.to_dict(),'vocabulary_size':41})
    assert a.fingerprint!=b.fingerprint and a.defines()['RT_GQA_WIDTH']==64
