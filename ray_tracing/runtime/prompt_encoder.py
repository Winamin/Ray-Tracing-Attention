"""Use the source tokenizer and chat template without a model framework."""
from pathlib import Path
import json
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
if __package__:
    from .settings import default_model
else:
    from settings import default_model


class LocalPromptEncoder:
    def __init__(self, model_dir=None):
        model_dir=model_dir or default_model()
        self.model_dir = Path(model_dir).resolve(strict=True)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / 'tokenizer.json'))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        generation=self.model_dir/'generation_config.json'
        config=json.loads(generation.read_text(encoding='utf8')) if generation.exists() else {}
        bos=config.get('bos_token_id')
        self.bos_token=self.tokenizer.id_to_token(bos) if isinstance(bos,int) else ''
        eos=config.get('eos_token_id',[])
        if isinstance(eos,int):eos=[eos]
        self.stop_ids=list(dict.fromkeys([*(eos or []),*[v for s in ('<|im_end|>','<|endoftext|>','<|eot_id|>') if (v:=self.tokenizer.token_to_id(s)) is not None]]))[:2]
        self.eos_token=self.tokenizer.id_to_token(self.stop_ids[0]) if self.stop_ids else ''
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise ValueError(message)

        environment.globals['raise_exception'] = raise_exception
        self.template = environment.from_string(
            (self.model_dir / 'chat_template.jinja').read_text(encoding='utf8'))

    def encode(self, messages):
        text = self.template.render(messages=messages, add_generation_prompt=True,
                                    enable_thinking=False, tools=None, add_vision_id=False,
                                    bos_token=self.bos_token or '',eos_token=self.eos_token or '')
        return self.tokenizer.encode(text, add_special_tokens=False).ids
