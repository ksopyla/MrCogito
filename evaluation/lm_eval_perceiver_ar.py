"""lm-evaluation-harness adapter for the Perceiver AR v2 family (`nn/perceiver_ar_lm.py`).

Registers the model type ``perceiver_ar`` with lm-eval so any harness task runs on an E18/E21
checkpoint exactly as it runs on a Hugging Face causal LM:

    lm_eval --model perceiver_ar \
        --model_args pretrained=Cache/Training/<run>/final,max_length=2048,batch_size=16 \
        --tasks hellaswag,arc_easy --num_fewshot 0

or programmatically via :func:`evaluation.run_lm_eval_suite.main`.

Why a subclass of ``HFLM`` rather than a bare ``LM``: the harness' request batching, the
loglikelihood / rolling-loglikelihood machinery and the result schema are all inherited, so our
numbers are produced by the same code path as the SmolLM2 / Qwen reference numbers. Only the
model construction differs:

* the checkpoint is loaded with ``PerceiverARLM`` (not ``AutoModelForCausalLM`` — the family is
  not registered with transformers' auto classes);
* ``attn_backend`` defaults to ``sdpa`` for short-context scoring: the ``flex`` backend
  ``torch.compile``s one graph per sequence length and the harness batches hundreds of distinct
  lengths, so flex would recompile constantly; sdpa's dense sliding-window mask is cheap at
  ≤ 4k tokens;
* ``attn_pad_multiple`` is forced to 1 so a 120-token multiple-choice request is not padded to
  the training block size (2048) — a 10–20× waste on HellaSwag-sized inputs;
* the tokenizer is taken from the checkpoint directory when it was saved there, else from the
  E18 default (``HuggingFaceTB/SmolLM3-3B``, id-identical to the Llama-3 vocabulary).

``forward(input_ids)`` with no labels returns the tanh-soft-capped logits, so the harness scores
exactly the distribution the model was trained under (``chunked_softcap_ce``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402

from evaluation.long_context_probes import DEFAULT_TOKENIZER, resolve_tokenizer_name  # noqa: E402
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM  # noqa: E402


def load_perceiver_ar_for_eval(
    checkpoint: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_backend: str = "sdpa",
    attn_pad_multiple: int = 1,
) -> PerceiverARLM:
    """Load a checkpoint with eval-friendly attention settings (see module docstring)."""
    cfg = PerceiverARConfig.from_pretrained(checkpoint)
    cfg.attn_backend = attn_backend
    cfg.attn_pad_multiple = int(attn_pad_multiple)
    cfg.use_liger = False  # fused CE is a training-time path; eval reads logits
    model = PerceiverARLM.from_pretrained(checkpoint, config=cfg)
    model.eval()
    if device.startswith("cuda") and dtype is not None:
        model.to(dtype)
    model.to(device)
    return model


def _to_dtype(dtype) -> Optional[torch.dtype]:
    if dtype is None or dtype == "auto":
        return torch.bfloat16
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype).replace("torch.", ""))


@register_model("perceiver_ar")
class PerceiverARLMEval(HFLM):
    """``HFLM`` over a ``PerceiverARLM`` instance; causal backend; no generation path yet.

    Loglikelihood-scored tasks (HellaSwag, ARC, PIQA, WinoGrande, OpenBookQA, BoolQ, SIQA,
    CommonsenseQA, LAMBADA, WikiText word-perplexity, MMLU) are fully supported.
    ``generate_until`` tasks (GSM8K, RULER, ...) need the KV-cache ``generate`` of the E18 main
    run and raise ``NotImplementedError`` until then.
    """

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        attn_backend: str = "sdpa",
        attn_pad_multiple: int = 1,
        max_length: int = 2048,
        batch_size: int | str = 16,
        device: str = "cuda",
        dtype: str | torch.dtype | None = "bfloat16",
        add_bos_token: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            device = "cpu"
        torch_dtype = _to_dtype(dtype) if str(device).startswith("cuda") else torch.float32
        model = load_perceiver_ar_for_eval(
            pretrained,
            device=device,
            dtype=torch_dtype,
            attn_backend=attn_backend,
            attn_pad_multiple=int(attn_pad_multiple),
        )
        tok = resolve_tokenizer_name(pretrained, tokenizer)
        # HFLM ignores `device`/`dtype` for a pre-built model (it reads model.device); the
        # tokenizer, max_length and batch size are still ours to set.
        super().__init__(
            pretrained=model,
            backend="causal",
            tokenizer=tok,
            max_length=int(max_length),
            batch_size=batch_size,
            add_bos_token=add_bos_token,
            **kwargs,
        )
        self.checkpoint_path = pretrained

    def _model_call(self, inps: torch.Tensor, attn_mask=None, labels=None) -> torch.Tensor:
        # Causal-only family: the harness right-pads and never passes an attention mask here.
        assert attn_mask is None and labels is None, "perceiver_ar is a causal-only eval model"
        with torch.no_grad():
            return self.model(input_ids=inps).logits

    def _model_generate(self, context, max_length: int, stop, **generation_kwargs):
        raise NotImplementedError(
            "perceiver_ar has no HF-compatible generate() yet; use loglikelihood tasks "
            "(the E18 main run's KV-cache decode will enable generate_until tasks)."
        )


__all__ = [
    "DEFAULT_TOKENIZER",
    "PerceiverARLMEval",
    "load_perceiver_ar_for_eval",
    "resolve_tokenizer_name",
]

if __name__ == "__main__":  # pragma: no cover — convenience: `python -m evaluation.lm_eval_perceiver_ar`
    from lm_eval.__main__ import cli_evaluate

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cli_evaluate()
