import logging
import random
import sys
from pathlib import Path

import httpx
import llm
import torch

# Suppress noisy nanochat/torch/httpx logging
logging.getLogger("llm_mrchatterbox.nanochat").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
from tokenizers import Tokenizer

import llm_mrchatterbox.nanochat.gpt as gpt_module
from llm_mrchatterbox.nanochat.common import compute_init, autodetect_device_type
from llm_mrchatterbox.nanochat.engine import Engine
from llm_mrchatterbox.nanochat.checkpoint_manager import (
    load_checkpoint,
    _patch_missing_config_keys,
    _patch_missing_keys,
)
from llm_mrchatterbox.nanochat.gpt import GPT, GPTConfig

HF_REPO = "https://huggingface.co/tventurella/mr_chatterbox_model/resolve/main"
MODEL_FILES = [
    "model_000070.pt",
    "meta_000070.json",
]
TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"
STEP = 70


def _cache_dir():
    return llm.user_dir() / "mrchatterbox"


def _ensure_downloaded():
    """Download model files from HuggingFace if not already cached."""
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        dest = cache / filename
        if dest.exists():
            continue
        url = f"{HF_REPO}/{filename}"
        print(f"Downloading {filename}...", file=sys.stderr, flush=True)
        tmp = dest.with_suffix(".tmp")
        with httpx.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        print(
                            f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
            print(file=sys.stderr)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(
            f"  Saved {filename} ({size_mb:.1f} MB) to {cache}",
            file=sys.stderr,
            flush=True,
        )


SYSTEM_PREFIX = (
    "[You are a learned Victorian gentleman in conversation. "
    "Address the question or remark put to you directly.]\n\n"
)


class VictorianTokenizer:
    def __init__(self, tokenizer_path):
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        self._tok.no_padding()
        self._tok.no_truncation()

    def get_vocab_size(self):
        return self._tok.get_vocab_size()

    def get_bos_token_id(self):
        return self._tok.token_to_id("<|endoftext|>")

    def encode(self, text, prepend=None, **kwargs):
        if isinstance(text, str):
            ids = self._tok.encode(text).ids
            if prepend is not None:
                ids = [prepend] + ids
            return ids
        return [self._tok.encode(t).ids for t in text]

    def decode(self, ids):
        return self._tok.decode(ids)

    def encode_special(self, token):
        result = self._tok.token_to_id(token)
        if result is not None:
            return result
        _map = {
            "<|assistant_start|>": "<victorian>",
            "<|assistant_end|>": "<|endoftext|>",
            "<|user_start|>": "<human>",
            "<|user_end|>": "<|endoftext|>",
            "<|bos|>": "<|endoftext|>",
            "<|python_start|>": None,
            "<|python_end|>": None,
            "<|output_start|>": None,
            "<|output_end|>": None,
        }
        mapped = _map.get(token)
        if mapped:
            return self._tok.token_to_id(mapped)
        return None


def _load_model():
    """Load the Mr. Chatterbox model and tokenizer. Called once on first use."""
    _ensure_downloaded()
    checkpoint_dir = str(_cache_dir())

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    # Detect ve_gate_channels from checkpoint to patch model construction
    checkpoint_data = torch.load(
        f"{checkpoint_dir}/model_{STEP:06d}.pt", map_location="cpu"
    )
    ve_gate_channels = 12
    for key, val in checkpoint_data.items():
        if "ve_gate.weight" in key:
            ve_gate_channels = val.shape[1]
            break
    del checkpoint_data

    # Monkey-patch CausalSelfAttention to use correct ve_gate_channels
    _orig_init = gpt_module.CausalSelfAttention.__init__

    def _patched_init(self, config, layer_idx):
        _orig_init(self, config, layer_idx)
        if self.ve_gate is not None:
            self.ve_gate_channels = ve_gate_channels
            self.ve_gate = gpt_module.Linear(
                ve_gate_channels, self.n_kv_head, bias=False
            )

    gpt_module.CausalSelfAttention.__init__ = _patched_init

    # Load checkpoint and build model
    model_data, _, meta_data = load_checkpoint(
        checkpoint_dir, STEP, device, load_optimizer=False
    )
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    tokenizer = VictorianTokenizer(TOKENIZER_PATH)
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]

    return model, tokenizer


class MrChatterbox(llm.Model):
    model_id = "mrchatterbox"
    can_stream = True

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._engine = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tokenizer = _load_model()
            self._engine = Engine(self._model, self._tokenizer)

    def execute(self, prompt, stream, response, conversation):
        self._ensure_loaded()

        tokenizer = self._tokenizer
        bos = tokenizer.get_bos_token_id()
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")

        # Build token sequence from conversation history
        conversation_tokens = [bos]
        turn_count = 0

        # Replay previous turns if in conversation mode
        if conversation and conversation.responses:
            for prev_response in conversation.responses:
                # User turn
                content = prev_response.prompt.prompt
                if content and content[0].islower():
                    content = content[0].upper() + content[1:]
                if turn_count == 0:
                    content = SYSTEM_PREFIX + content
                conversation_tokens.append(user_start)
                conversation_tokens.extend(tokenizer.encode(content))
                conversation_tokens.append(user_end)
                turn_count += 1

                # Assistant turn
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(tokenizer.encode(prev_response.text()))
                conversation_tokens.append(assistant_end)

        # Current user turn
        text = prompt.prompt
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if turn_count == 0:
            text = SYSTEM_PREFIX + text
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(text))
        conversation_tokens.append(user_end)

        # Kick off assistant
        conversation_tokens.append(assistant_start)

        for token_column, token_masks in self._engine.generate(
            conversation_tokens,
            num_samples=1,
            max_tokens=256,
            temperature=0.7,
            top_k=50,
            seed=random.randint(0, 2**31 - 1),
            repetition_penalty=1.3,
            repetition_window=64,
        ):
            token = token_column[0]
            if token == assistant_end:
                break
            yield tokenizer.decode([token])


@llm.hookimpl
def register_models(register):
    register(MrChatterbox())


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="mrchatterbox")
    def mrchatterbox_():
        "Commands for the Mr Chatterbox model"

    @mrchatterbox_.command(name="path")
    def path():
        "Print the path to the cached model files"
        import click

        click.echo(_cache_dir())

    @mrchatterbox_.command(name="delete-model")
    def delete_model():
        "Delete cached model files"
        import click
        import shutil

        cache = _cache_dir()
        if not cache.exists():
            click.echo("No cached model files found.")
            return
        files = list(cache.iterdir())
        if not files:
            click.echo("No cached model files found.")
            cache.rmdir()
            return
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            click.echo(f"Deleting {f.name} ({size_mb:.1f} MB)")
            f.unlink()
        cache.rmdir()
        click.echo("Done.")
