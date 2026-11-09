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

    def _render(self, messages, add_generation_prompt):
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                    enable_thinking=False, tools=None, add_vision_id=False,
                                    bos_token=self.bos_token or '',eos_token=self.eos_token or '')

    def encode(self, messages):
        return self.tokenizer.encode(self._render(messages, True), add_special_tokens=False).ids

    def encode_with_checkpoint(self, messages):
        """Keep the source prompt intact and checkpoint before its mutable suffix."""
        ids = self.encode(messages)
        history = self.tokenizer.encode(self._render(messages, False), add_special_tokens=False).ids
        boundary = 0
        for token, previous in zip(ids, history):
            if token != previous:
                break
            boundary += 1
        return ids, boundary or len(ids)

    def encode_with_boundaries(self, messages):
        """Mark canonical input/output ends without changing source rendering."""
        ids = self.encode(messages)
        boundaries = {}
        for end, message in enumerate(messages, 1):
            try:
                rendered = self._render(messages[:end], False)
            except ValueError:
                # Some source templates require a user turn even for rendering.
                # The complete prompt above has already passed their validation.
                continue
            prefix = self.tokenizer.encode(rendered, add_special_tokens=False).ids
            shared = 0
            for current, previous in zip(ids, prefix):
                if current != previous:
                    break
                shared += 1
            if shared:
                kind = 2 if message.get('role') == 'assistant' else 1
                boundaries[shared] = boundaries.get(shared, 0) | kind
        boundaries[len(ids)] = boundaries.get(len(ids), 0) | 1
        return ids, sorted(boundaries.items())
