"""Text capability checks — round 1 (2026-10-06): calibrate the draft protocol on Polonez, then compare.

Spec: docs/engineering_specs/text_capability_checks.md · skill `text-checks` · board
docs/3_Evaluations_and_Baselines/text_capability_board.html (analysis/text_board.py).

The protocol is a first draft, so round 1 starts by measuring what the draft guessed: throughput, which
tasks the dense model can learn at 30M / 0.6B tokens, and whether the GPU kernels hold. Every later step
waits for the gate of the step before it; a failed gate changes the protocol (a new data or protocol
version, recorded in the spec), not the model.

Each experiment names the run folder(s) and jobs it consists of, so the board can show its state from
the pulled ledgers (`scripts/pull_text_checks_results.sh`).
"""
from __future__ import annotations

DATA = "text_checks_v0"            # under the server's datasets_tok folder
SMOKE_DATA = "text_checks_smoke_v0"
HOST = "polonez"
REFS = ("dense", "local")
CANDIDATES = ("e31c", "e31c_loop")

EXPERIMENTS = [
    {
        "id": "X0", "title": "Build the data",
        "question": "Tokenizer, the ~2.1B-token training mix and the frozen exams (1k → 128k).",
        "runs": {"data_v0": ["data"]}, "gpu_hours": 0, "estimate": "~1 h CPU",
        "gate": "builds; lengths within a few tokens of target; hashes recorded",
    },
    {
        "id": "X1", "title": "Server smoke (2M models, 1 GPU)",
        "question": "Do the real launcher, the GPU attention kernels (incl. the notebook's wider query) and the "
                    "scorer run on CUDA for all four models?",
        "runs": {"r1_smoke": [f"{k}_{a}" for a in (*REFS, *CANDIDATES) for k in ("train", "eval")]},
        "gpu_hours": 0.5, "estimate": "minutes",
        "gate": "every job exits 0; throughput recorded",
    },
    {
        "id": "X2", "title": "Dense recipe (tuning)",
        "question": "Step size for the dense model at 30M: 4 short runs, picked by held-out loss.",
        "runs": {"r1_screen": [f"tune_dense_lr{lr:.2e}" for lr in (5e-4, 1e-3, 2e-3, 4e-3)]},
        "gpu_hours": 4, "estimate": "~1 h on 4 GPUs",
        "gate": "a clear minimum inside the grid (else widen the grid)",
    },
    {
        "id": "X3", "title": "Dense calibration run (30M, 0.6B tokens)",
        "question": "Which question types can a 30M dense model learn at 4k with this budget? "
                    "Its passes decide which levels gate; its speed fixes the caps.",
        "runs": {"r1_screen": ["train_dense", "eval_dense"]}, "gpu_hours": 12, "estimate": "~2–4 h + eval",
        "gate": "dense passes at least T1–T2 at 4k; if not, change the mix / budget (new data version)",
    },
    {
        "id": "X4", "title": "Recipes for the other models",
        "question": "Same tuning for the no-memory control, E31c and E31c + reread loop.",
        "runs": {"r1_screen": [f"tune_{a}_lr{lr:.2e}" for a in ("local", *CANDIDATES) for lr in (5e-4, 1e-3, 2e-3, 4e-3)]},
        "gpu_hours": 12, "estimate": "~3 h on 4 GPUs",
        "gate": "a clear minimum per model",
    },
    {
        "id": "X5", "title": "Screen comparison (30M)",
        "question": "Does the notebook beat the no-memory control on reach and tokens-to-pass without "
                    "hurting language, and does the reread loop add anything?",
        "runs": {"r1_screen": [f"{k}_{a}" for a in ("local", *CANDIDATES) for k in ("train", "eval")]},
        "gpu_hours": 40, "estimate": "~10–14 h in two bursts",
        "gate": "a candidate passes T2 at 4k with story loss within 5 % of dense → it goes to the main tier",
    },
    {
        "id": "X6", "title": "Seed spread of the references",
        "question": "How much do dense and the no-memory control move between seeds? Sets the tie margin.",
        "runs": {"r1_screen_s1": ["train_dense", "eval_dense", "train_local", "eval_local"],
                 "r1_screen_s2": ["train_dense", "eval_dense", "train_local", "eval_local"]},
        "gpu_hours": 40, "estimate": "~10 h",
        "gate": "spread measured (no pass/fail)",
    },
    {
        "id": "X7", "title": "Main comparison (100M, 2B tokens)",
        "question": "The claim run: the models that passed the screen gate, at 100M, read up to 128k.",
        "runs": {"r1_main": [f"{k}_{a}" for a in (*REFS, *CANDIDATES) for k in ("train", "eval")]},
        "gpu_hours": 480, "estimate": "~4–5 days of Polonez in bursts",
        "gate": "needs the author's go-ahead after X3–X6",
    },
]
