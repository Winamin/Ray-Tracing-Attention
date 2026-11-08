"""Build one shared shader implementation for a versioned model specification."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import subprocess
from .spec import asset_spec
from ..runtime.settings import LOCAL

NATIVE=Path(__file__).resolve().parents[1]/'native'

def build(spec, output=None):
    spec.validate()
    files=sorted(p for p in NATIVE.rglob('*') if p.is_file())
    digest=hashlib.sha256(spec.fingerprint.encode())
    for p in files: digest.update(p.relative_to(NATIVE).as_posix().encode());digest.update(p.read_bytes())
    target=Path(output) if output else LOCAL/'build'/digest.hexdigest()[:20]
    executable=target/('rt_optical_resident.exe' if os.name=='nt' else 'rt_optical_resident')
    if executable.exists() and (target/'complete').exists(): return executable
    for tool in ('glslangValidator','spirv-val',os.environ.get('CXX','g++')):
        if not shutil.which(tool): raise RuntimeError(f'Build tool {tool} is required; install the Vulkan SDK and a C++17 compiler')
    target.mkdir(parents=True,exist_ok=True)
    lines=[f'#define {k} {v}' for k,v in spec.defines().items()]
    (target/'rt_model_config.h').write_text('\n'.join(lines)+'\n',encoding='utf8')
    full=list(spec.full_layers)
    mapping=[full.index(i) if i in full else 0 for i in range(spec.depth)]
    cpp='constexpr uint32_t MODEL_FULL_LAYERS[]={'+','.join(map(str,full or [0]))+'};\n'
    cpp+='constexpr uint32_t MODEL_FULL_INDEX[]={'+','.join(map(str,mapping))+'};\n'
    (target/'rt_layer_map.hpp').write_text(cpp,encoding='utf8')
    commands=[]
    for name in ('resident_math','resident_ray','reply_cache'):
        commands += [['glslangValidator','-V','--target-env','vulkan1.2','-I'+str(target),'-I'+str(NATIVE/'shaders'),str(NATIVE/'shaders'/(name+'.comp')),'-o',str(target/(name+'.spv'))],
                     ['spirv-val','--target-env','vulkan1.2',str(target/(name+'.spv'))]]
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
