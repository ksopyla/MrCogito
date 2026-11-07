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
import torch.nn.functional as F

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
    """Load a checkpoint with eval-friendly attention settings (see module docstring).
    Serves both from-scratch families: E18 `perceiver_ar` and E22 `perceiver_concept`."""
    from nn.perceiver_families import checkpoint_family

    if checkpoint_family(checkpoint) == "perceiver_concept":
        from nn.perceiver_concept_lm import PerceiverConceptConfig as Cfg, PerceiverConceptLM as Model
    else:
        Cfg, Model = PerceiverARConfig, PerceiverARLM
    cfg = Cfg.from_pretrained(checkpoint)
    cfg.attn_backend = attn_backend
    cfg.attn_pad_multiple = int(attn_pad_multiple)
    cfg.use_liger = False  # fused CE is a training-time path; eval reads logits
    model = Model.from_pretrained(checkpoint, config=cfg)
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

    # fp32 [rows, S, vocab] budget per forward chunk; the full-batch output is allocated once.
    LOGITS_CHUNK_BYTES = 2 * 1024**3
    # generation: the row is padded to a multiple of this once, so every decoding step has the same shape
    GEN_PAD = 256

    def _model_call(self, inps: torch.Tensor, attn_mask=None, labels=None) -> torch.Tensor:
        # Causal-only family: the harness right-pads and never passes an attention mask here.
        assert attn_mask is None and labels is None, "perceiver_ar is a causal-only eval model"
        # The harness applies `log_softmax` to what we return and never reads raw logits, so we
        # hand back log-probabilities (idempotent under its log_softmax). Computing them in row
        # chunks bounds the peak at one full fp32 [B, S, V] tensor instead of the ~4 copies the
        # softcap + log_softmax path would materialise for a 128k vocabulary at 2048 tokens.
        B, S = inps.shape
        V = int(self.model.config.vocab_size)
        rows = max(1, self.LOGITS_CHUNK_BYTES // (S * V * 4))
        with torch.no_grad():
            if rows >= B:
                return F.log_softmax(self.model(input_ids=inps).logits, dim=-1)
            out = torch.empty(B, S, V, dtype=torch.float32, device=inps.device)
            for i in range(0, B, rows):
                out[i:i + rows] = F.log_softmax(self.model(input_ids=inps[i:i + rows]).logits, dim=-1)
            return out

    def _model_generate(self, context, max_length: int, stop, **generation_kwargs):
        """Greedy decoding without a cache (E33n: RULER through `generate_until`).

        Every new token re-runs the whole row: O(new × forward), fine for RULER's short answers at ≤ 64k. Left padding
        from the harness becomes document id −1 (padding keys are never read; positions restart at the first real
        token). Only the current position is projected through the output layer (`hidden_states`), so a 64k row never
        materialises [S, V] logits. Stops when every row has produced one of `stop` or the length is reached.

        One row length for the whole answer (`GEN_PAD` multiple): flex attention compiles one graph per shape, and a
        row that grew by a token per step recompiled every step until dynamo's cache was full; it then fell back to
        eager flex attention, whose dense scores need 72 GB at 16k (E33p RULER 16k out of memory on every arm, and
        ~3 h for 4k, 2026-11-07). The slots after the current position hold document id −1; no position reads later
        content (tests/test_note_open.py::test_no_future_leak), so full attention and LSA's read write exactly the
        tokens the growing row wrote (tests/test_long_eval_fixes.py). Note models move by ~3 % with the row's
        length (the page grid's last page is placed at [S − W, S)); here every answer position sits inside a longer
        row, as in training."""
        ids = context
        attn = generation_kwargs.get("attention_mask")
        cfg = self.model.config
        W = self.model.lm_head.weight
        eos = self.tokenizer.eos_token_id
        B, L0 = ids.shape
        new = max(0, int(max_length) - L0)
        Lb = -(-(L0 + new) // self.GEN_PAD) * self.GEN_PAD
        buf = torch.full((B, Lb), eos, dtype=ids.dtype, device=ids.device)
        buf[:, :L0] = ids
        doc = torch.full((B, Lb), -1, dtype=torch.long, device=ids.device)
        doc[:, :L0] = 0 if attn is None else torch.where(attn.bool(), 0, -1)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        stops = [s for s in (stop or []) if s]
        t = L0
        with torch.no_grad():
            for _ in range(new):
                h = self.model.hidden_states(buf, doc_ids=doc)[:, t - 1]
                logits = F.linear(h, W).float()
                if cfg.logit_softcap:
                    logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
                nxt = logits.argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
                buf[:, t] = nxt
                doc[:, t] = 0
                t += 1
                done |= nxt == eos
                if stops:
                    for b in range(B):
                        if not done[b]:
                            text = self.tokenizer.decode(buf[b, L0:t].tolist())
                            done[b] = any(s in text for s in stops)
                if bool(done.all()):
                    break
        return buf[:, :t]


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
