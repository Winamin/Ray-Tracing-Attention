"""Inspect, translate and build a model for the resident Vulkan RT runtime."""
import argparse
import json
from pathlib import Path
from .sources import open_source
from .export import export
from .build import build
from ..runtime.settings import LOCAL, select_model

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path,help='GGUF file or Hugging Face safetensors directory')
    p.add_argument('--output',type=Path);p.add_argument('--inspect',action='store_true');p.add_argument('--no-build',action='store_true');p.add_argument('--activate',action='store_true')
    args=p.parse_args(argv);src=open_source(args.source)
    if args.inspect:print(json.dumps(src.spec.to_dict(),ensure_ascii=False,indent=2));return
    destination=args.output or LOCAL/'models'/args.source.stem
    if (destination/'optical_model.bin').exists():p.error(f'{destination} already contains a compiled model; choose a new output directory')
    asset,metadata=export(src,destination)
    worker=build(src.spec) if not args.no_build else None
    if args.activate:select_model(destination)
    print(json.dumps({'asset':str(asset.resolve()),'worker':str(worker) if worker else None,'coefficient_bytes':metadata['parameter_bytes']},ensure_ascii=False))

if __name__=='__main__':main()
