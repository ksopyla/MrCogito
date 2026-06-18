"""L1 (generation faithfulness) + L3 (compression faithfulness) metrics for concept_ar.

These answer "can the input be recovered FROM the concepts, and is the recovery
specific to *this* input?" — the faithfulness axis of the evaluation protocol
(`docs/3_Evaluations_and_Baselines/evaluation_protocol.md`). They complement the
concept-ablation ΔCE (which asks "does the decoder USE the concepts?").

All functions reuse the model's own teacher-forcing convention via
`encode_concepts` / `_shift_right` / `decode_logits`; no shift is re-derived here
(re-deriving it is the class of bug behind the E01 double-shift).

`batches` is a list of `(input_ids, attention_mask)` CPU tensors, as produced for
`compute_ar_concept_ablation`. The encoder sees the full clean sequence, so this is
the reconstruction contract (matches E01; for prefix→suffix use the training eval).
"""

from collections import Counter
from typing import Dict, List, Optional

import torch


def token_f1(pred_ids: List[int], gold_ids: List[int]) -> float:
    """Multiset (order-independent) token overlap F1 between two id lists."""
    if not pred_ids or not gold_ids:
        return 0.0
    pred_c, gold_c = Counter(pred_ids), Counter(gold_ids)
    overlap = sum((pred_c & gold_c).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_ids)
    recall = overlap / len(gold_ids)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def _teacher_forced_logits(model, input_ids, attention_mask, concepts=None):
    if concepts is None:
        concepts = model.encode_concepts(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
    decoder_input_ids = model._shift_right(input_ids)
    logits = model.decode_logits(concepts, decoder_input_ids)
    return logits, concepts


@torch.no_grad()
def _greedy_decode(model, concepts, start_id, eos_id, max_new_tokens):
    cur = torch.tensor([[start_id]], device=concepts.device)
    for _ in range(max_new_tokens):
        logits = model.decode_logits(concepts, cur)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return cur[0, 1:].tolist()


@torch.no_grad()
def compute_roundtrip_recovery(
    model,
    batches,
    device,
    concept_num: int,
    free_running_examples: int = 8,
    eos_id: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> Dict[str, object]:
    """Round-trip recovery of the input FROM its concepts.

    Returns teacher-forced token accuracy (L1), free-running exact-match + token-F1
    on a small sample (L1), and a teacher-forced recovery-vs-compression-ratio curve
    bucketed by ceil(seq_len / C) (L3).
    """
    start_id = model.config.bos_token_id or model.config.eos_token_id or model.config.pad_token_id or 0
    if eos_id is None:
        eos_id = model.config.eos_token_id

    tf_correct = 0
    tf_total = 0
    # compression buckets: ratio -> [correct, total]
    buckets: Dict[int, List[int]] = {}
    fr_exact: List[float] = []
    fr_f1: List[float] = []
    fr_done = 0

    for input_ids, attention_mask in batches:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits, concepts = _teacher_forced_logits(model, input_ids, attention_mask)
        preds = logits.argmax(dim=-1)                      # [B, T]
        valid = attention_mask.bool()                      # non-pad targets
        correct = (preds == input_ids) & valid

        tf_correct += int(correct.sum().item())
        tf_total += int(valid.sum().item())

        # per-example compression bucket (L3)
        for i in range(input_ids.size(0)):
            n_valid = int(valid[i].sum().item())
            if n_valid == 0:
                continue
            ratio = max(1, -(-n_valid // max(1, concept_num)))  # ceil(seq_len / C)
            b = buckets.setdefault(ratio, [0, 0])
            b[0] += int(correct[i].sum().item())
            b[1] += n_valid

            # free-running recovery on a small sample (L1)
            if fr_done < free_running_examples:
                gold = input_ids[i][valid[i]].tolist()
                cap = max_new_tokens or (len(gold) + 1)
                single_concepts = concepts[i : i + 1]
                gen = _greedy_decode(model, single_concepts, start_id, eos_id, cap)
                m = min(len(gen), len(gold))
                exact = (
                    sum(1 for a, b_ in zip(gen[:m], gold[:m]) if a == b_) / len(gold)
                    if gold else 0.0
                )
                fr_exact.append(exact)
                fr_f1.append(token_f1(gen, gold))
                fr_done += 1

    out: Dict[str, object] = {
        "teacher_forced_token_acc": (tf_correct / tf_total) if tf_total else float("nan"),
        "free_running_exact_match": (sum(fr_exact) / len(fr_exact)) if fr_exact else float("nan"),
        "free_running_token_f1": (sum(fr_f1) / len(fr_f1)) if fr_f1 else float("nan"),
        "free_running_n": len(fr_exact),
        "compression_curve": {
            str(r): {
                "compression_ratio": r,
                "teacher_forced_token_acc": (c / t) if t else float("nan"),
                "n_tokens": t,
            }
            for r, (c, t) in sorted(buckets.items())
        },
    }
    return out


@torch.no_grad()
def compute_latent_specificity(model, batches, device) -> Dict[str, float]:
    """Is recovery specific to THIS input's concepts?

    Compares teacher-forced token accuracy with the matched concepts vs the same
    concepts row-shuffled across the batch. A positive `specificity_acc_drop` means
    the decoder's predictions depend on which input produced the concepts (not just
    on generic concept statistics). Also reports the mean symmetric-KL between the
    matched and shuffled next-token distributions.
    """
    matched_correct = matched_total = 0
    shuf_correct = 0
    kl_sum = 0.0
    kl_count = 0

    for input_ids, attention_mask in batches:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        valid = attention_mask.bool()
        if input_ids.size(0) < 2:
            continue

        logits_m, concepts = _teacher_forced_logits(model, input_ids, attention_mask)
        perm = torch.roll(torch.arange(input_ids.size(0), device=device), shifts=1)
        logits_s, _ = _teacher_forced_logits(model, input_ids, attention_mask, concepts=concepts[perm])

        matched_correct += int(((logits_m.argmax(-1) == input_ids) & valid).sum().item())
        shuf_correct += int(((logits_s.argmax(-1) == input_ids) & valid).sum().item())
        matched_total += int(valid.sum().item())

        # symmetric KL on valid positions
        lpm = torch.log_softmax(logits_m, dim=-1)
        lps = torch.log_softmax(logits_s, dim=-1)
        pm, ps = lpm.exp(), lps.exp()
        sym_kl = ((pm * (lpm - lps)).sum(-1) + (ps * (lps - lpm)).sum(-1))  # [B, T]
        kl_sum += float(sym_kl[valid].sum().item())
        kl_count += int(valid.sum().item())

    if matched_total == 0:
        return {
            "specificity_acc_matched": float("nan"),
            "specificity_acc_shuffled": float("nan"),
            "specificity_acc_drop": float("nan"),
            "specificity_symmetric_kl": float("nan"),
        }
    acc_m = matched_correct / matched_total
    acc_s = shuf_correct / matched_total
    return {
        "specificity_acc_matched": acc_m,
        "specificity_acc_shuffled": acc_s,
        "specificity_acc_drop": acc_m - acc_s,
        "specificity_symmetric_kl": (kl_sum / kl_count) if kl_count else float("nan"),
    }
