import hashlib
import struct
from ray_tracing.compiler.export import export
from ray_tracing.compiler.sources import SafetensorsSource
from ray_tracing.verification.verify_portable import fixture


def test_original_coefficients_and_tied_vocabulary(tmp_path):
    source=fixture('llama',tmp_path,tied=True)
    src=SafetensorsSource(source);asset,metadata=export(src,tmp_path/'model')
    vocabulary=(asset.parent/'vocabulary.bin').read_bytes()
    raw,kind,_=src.packed('token_embd.weight')
    assert vocabulary[16:]==raw.tobytes()
    assert metadata['parameter_bytes']==asset.stat().st_size-44+(asset.parent/'geometry_weights.bin').stat().st_size+len(vocabulary)-16
    for t in metadata['tensor_payloads']:
        if 'sample_sha256' not in t:continue
        original=src.packed(t['name'])[0]
        assert hashlib.sha256(original[:4096].tobytes()+original[-4096:].tobytes()).hexdigest()==t['sample_sha256']
    params=asset.read_bytes()[44:]
    assert struct.unpack_from('<I',params,13*4)[0]==kind
    assert struct.unpack_from('<I',params,14*4)[0]==struct.unpack_from('<I',params,16*4)[0]
