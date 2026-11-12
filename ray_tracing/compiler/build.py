"""Build one shared shader implementation for a versioned model specification."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import subprocess
import re
from .spec import asset_spec
from ..runtime.settings import LOCAL

NATIVE=Path(__file__).resolve().parents[1]/'native'

def build(spec, output=None):
    spec.validate()
    files=sorted(p for p in NATIVE.rglob('*') if p.is_file())
    digest=hashlib.sha256(spec.fingerprint.encode())
    for p in files: digest.update(p.relative_to(NATIVE).as_posix().encode());digest.update(p.read_bytes())
    local_compiler=LOCAL/'tools/glslang/bin/glslangValidator.exe'
    if not local_compiler.exists():local_compiler=LOCAL/'tools/glslang/mingw64/bin/glslangValidator.exe'
    shader_compiler=os.environ.get('RT_GLSLANG',str(local_compiler) if local_compiler.exists() else 'glslangValidator')
    local_validator=local_compiler.with_name('spirv-val.exe')
    validator=os.environ.get('RT_SPIRV_VAL',str(local_validator) if local_validator.exists() else 'spirv-val')
    for tool in (shader_compiler,validator,os.environ.get('CXX','g++')):
        if not shutil.which(tool): raise RuntimeError(f'Build tool {tool} is required; install the Vulkan SDK and a C++17 compiler')
    version=subprocess.run([shader_compiler,'--version'],capture_output=True,text=True)
    if version.returncode:raise RuntimeError(version.stderr or version.stdout)
    match=re.search(r'Glslang Version: \d+:(\d+)\.(\d+)',version.stdout)
    vector_decode=bool(match and tuple(map(int,match.groups()))>=(16,6))
    digest.update(str(vector_decode).encode())
    target=Path(output) if output else LOCAL/'build'/digest.hexdigest()[:20]
    executable=target/('rt_optical_resident.exe' if os.name=='nt' else 'rt_optical_resident')
    if executable.exists() and (target/'complete').exists(): return executable
    target.mkdir(parents=True,exist_ok=True)
    lines=[f'#define {k} {v}' for k,v in spec.defines().items()]
    (target/'rt_model_config.h').write_text('\n'.join(lines)+'\n',encoding='utf8')
    full=list(spec.full_layers)
    mapping=[full.index(i) if i in full else 0 for i in range(spec.depth)]
    cpp='constexpr uint32_t MODEL_FULL_LAYERS[]={'+','.join(map(str,full or [0]))+'};\n'
    cpp+='constexpr uint32_t MODEL_FULL_INDEX[]={'+','.join(map(str,mapping))+'};\n'
    (target/'rt_layer_map.hpp').write_text(cpp,encoding='utf8')
    commands=[]
    shader_names=('resident_math','resident_mlp','resident_batch2','resident_pack','resident_attention','resident_attention_receipt','resident_attention_dense','resident_attention_local','resident_attention_classes','resident_auxiliary','resident_state','resident_ray','reply_cache','optical_path_key','history_cache')
    local_split_names=()
    if spec.full_layers and spec.attention_head_dim==256 and spec.query_width==4*spec.kv_width:
        local_split_names=('resident_attention_local_key','resident_attention_local_value',
                           'resident_attention_local_trace','resident_attention_local_key_fallback')
        shader_names+=local_split_names
    for name in shader_names:
        environment='vulkan1.3' if name in ('resident_batch2','resident_attention','resident_auxiliary') else 'vulkan1.2'
        defines=[]
        source='resident_attention_direct.comp' if name=='resident_attention_local' else name+'.comp'
        optimization=['-Os'] if name in local_split_names else []
        commands += [[shader_compiler,'-V',*optimization,'--target-env',environment,'-DRT_VECTOR_DECODE='+str(int(vector_decode)),'-I'+str(target),'-I'+str(NATIVE/'shaders'),*defines,str(NATIVE/'shaders'/source),'-o',str(target/(name+'.spv'))],
                     [validator,'--target-env',environment,str(target/(name+'.spv'))]]
        if name=='resident_attention_receipt':
            # Compile the TC variant from the same source; keep the portable
            # FP32 module free of cooperative-matrix capability requirements.
            commands += [[shader_compiler,'-V','--target-env','vulkan1.3','-DRT_ATTENTION_TC=1','-I'+str(target),'-I'+str(NATIVE/'shaders'),str(NATIVE/'shaders'/source),'-o',str(target/(name+'_tc.spv'))],
                         [validator,'--target-env','vulkan1.3',str(target/(name+'_tc.spv'))]]
    flags=['-static-libgcc','-static-libstdc++','-Wl,-Bstatic,--whole-archive','-lwinpthread','-Wl,--no-whole-archive,-Bdynamic','-lvulkan-1'] if os.name=='nt' else ['-lvulkan','-pthread']
    commands += [[os.environ.get('CXX','g++'),'-std=c++17','-O3','-ffp-contract=off','-I'+str(target),'-I'+str(NATIVE/'include'),str(NATIVE/'src/optical_resident.cpp'),*flags,'-o',str(executable)]]
    options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    for command in commands:
        r=subprocess.run(command,capture_output=True,text=True,encoding='utf8',errors='replace',**options)
        if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    (target/'complete').write_text(spec.fingerprint,encoding='utf8')
    return executable

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args(argv);print(build(asset_spec(args.model),args.output))

if __name__=='__main__':main()
