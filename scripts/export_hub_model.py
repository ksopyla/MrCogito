"""Export a `perceiver_ar` checkpoint as a 🤗 Hub folder that loads with `from_pretrained` (2026-11-18).

Engineering spec: docs/engineering_specs/hub_from_pretrained_loading.md. The folder this writes loads with

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True, dtype=torch.bfloat16)

and scores with `labels=` or samples with 🤗 `generate` over the cached decoder (nn/perceiver_ar_hub.py). It holds:

* `modeling_perceiver_ar.py` — the source modules the checkpoint's config uses, pasted in order into one flat file
  with their `nn.` imports resolved inside it (nn/exact_read.py, nn/latent_memory.py, nn/perceiver_ar_lm.py,
  nn/incremental.py, nn/perceiver_ar_hub.py, plus nn/concept_memory.py for the concept read and nn/lsa_buckets.py
  for LSA's read). Generated, never edited: the source stays the one place to change. Imports of training-only
  packages (Liger, flash-attn, the repo's data modules) sit inside `try` blocks, so 🤗's remote-code check asks only
  for `torch` and `transformers`.
* `configuration_perceiver_ar.py` — `PerceiverARConfig` and the constants it reads.
* `config.json` — the checkpoint's, with `auto_map` (AutoConfig, AutoModelForCausalLM), `architectures` →
  `PerceiverARForCausalLM`, `use_liger: false`, `tokenizer_name` → the Hub tokenizer id, `attn_backend: sdpa` and
  `attn_pad_multiple: 1` (the eval loader's settings: flex attention compiles one graph per shape, so short scoring
  and sampling loops run on sdpa; pass `attn_backend="flex"` to `from_pretrained` for long inputs on CUDA), and
  `tie_word_embeddings: false` (the factorised head reuses the input table itself; 🤗's tying must stay off).
* `generation_config.json` (bos / eos / pad ids), the weights (copied, or hard-linked with --link) and the tokenizer
  files saved with the checkpoint.
* `export_manifest.json` — the source commit, the vendored modules, every config field changed (old → new) and the
  sha256 of each file.

Usage:
    uv run python scripts/export_hub_model.py <checkpoint_dir> <out_dir> [--card README.md] [--check]

`--check` loads the folder with `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)` on the CPU and
runs one forward and a 4-token `generate`. The full equivalence tests are tests/test_hub_export.py.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
NN_DIR = REPO / "nn"
REPO_URL = "https://github.com/ksopyla/MrCogito"

CONFIG_FILE = "configuration_perceiver_ar.py"
MODELING_FILE = "modeling_perceiver_ar.py"
CONFIG_CLASS = "PerceiverARConfig"
MODEL_CLASS = "PerceiverARForCausalLM"
AUTO_MAP = {
    "AutoConfig": f"{CONFIG_FILE[:-3]}.{CONFIG_CLASS}",
    "AutoModelForCausalLM": f"{MODELING_FILE[:-3]}.{MODEL_CLASS}",
}
HUB_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"

# Order matters: a module comes after every module whose names it uses at import time.
SOURCE_ORDER = ("exact_read", "lsa_buckets", "latent_memory", "concept_memory", "perceiver_ar_lm", "incremental",
                "perceiver_ar_hub")
ALLOWED_PACKAGES = set(sys.stdlib_module_names) | {"torch", "transformers"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "chat_template.jinja", "tokenizer.model")
WEIGHT_PATTERNS = ("model.safetensors", "model-*.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


# ---------------------------------------------------------------------------------------------- which modules
def vendored_modules(cfg: dict) -> list[str]:
    """The source modules a checkpoint's config needs, in import order. The reading layer picks the optional ones:
    nn/concept_memory.py for the concept read (E37/E38), nn/lsa_buckets.py for LSA's read (E33p)."""
    family = cfg.get("checkpoint_family") or cfg.get("model_type")
    if family != "perceiver_ar" or cfg.get("model_type") != "perceiver_ar":
        raise SystemExit(f"only perceiver_ar checkpoints export (checkpoint_family {family!r}, "
                         f"model_type {cfg.get('model_type')!r})")
    read = str(cfg.get("par_global_read") or "dense")
    want = {"exact_read", "latent_memory", "perceiver_ar_lm", "incremental", "perceiver_ar_hub"}
    if read == "concept":
        want.add("concept_memory")
    elif read == "lsa":
        want.add("lsa_buckets")
    return [m for m in SOURCE_ORDER if m in want]


# ---------------------------------------------------------------------------------------------- source handling
class _Source:
    def __init__(self, name: str):
        self.name = name
        self.path = NN_DIR / f"{name}.py"
        self.text = self.path.read_text(encoding="utf-8")
        self.lines = self.text.splitlines(keepends=True)
        self.tree = ast.parse(self.text, filename=str(self.path))
        self.parents: dict = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node

    def segment(self, node: ast.AST, with_comments: bool = True) -> str:
        """The node's source lines (decorators included), with the comment lines right above it."""
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        if with_comments:
            while start > 1 and self.lines[start - 2].lstrip().startswith("#"):
                start -= 1
        return "".join(self.lines[start - 1:node.end_lineno])

    def in_try(self, node: ast.AST) -> bool:
        child, parent = node, self.parents.get(node)
        while parent is not None:
            if isinstance(parent, ast.Try) and child in parent.body:
                return True
            child, parent = parent, self.parents.get(parent)
        return False


def _bound_names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    if isinstance(node, ast.Import):
        return [a.asname or a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names]
    return []


def _used_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _root(module: Optional[str]) -> str:
    return (module or "").split(".")[0]


def _is_docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_main_guard(node: ast.AST) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__")


def _comment_block(text: str) -> str:
    return "".join(f"# {ln}".rstrip() + "\n" for ln in text.strip("\n").splitlines())


def _dropped_stub(name: str, module: str, src: "_Source") -> str:
    node = next((n for n in src.tree.body if name in _bound_names(n)), None)
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        raise SystemExit(f"nn/{src.name}.py imports {name!r} from nn/{module}.py at import time, and it is not a "
                         f"function or class: vendor nn/{module}.py")
    return (f"def {name}(*args, **kwargs):  # nn/{module}.py is not part of this export\n"
            f"    raise NotImplementedError(\"{name}: this checkpoint's config does not use nn/{module}.py\")\n")


def _flatten(src: _Source, vendored: set[str], skip_classes: Iterable[str] = ()) -> tuple[str, list[str]]:
    """One module's text for the flat file, and the top-level import statements it needs (stdlib / torch /
    transformers, for the shared header). Top-level `nn.` imports are dropped (the names live in the flat file), the
    nested ones become `pass` (or a NotImplementedError for a module left out), other packages' nested imports are
    wrapped in `try` (🤗's remote-code check skips `try` blocks), `__all__` and `__main__` blocks go."""
    skip_classes = set(skip_classes)
    edits: dict[int, tuple[int, str]] = {}  # first line → (last line, replacement)
    header: list[str] = []
    for i, node in enumerate(src.tree.body):
        a, b = node.lineno, node.end_lineno
        if i == 0 and _is_docstring(node):
            edits[a] = (b, _comment_block(node.value.value))
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            edits[a] = (b, "")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            if getattr(node, "level", 0):
                raise SystemExit(f"nn/{src.name}.py:{a}: relative import")
            if _root(mod) == "nn":
                dep = mod.split(".", 1)[1]
                stubs = "" if dep in vendored else "".join(_dropped_stub(al.name, dep, _Source(dep))
                                                           for al in node.names)
                edits[a] = (b, stubs)
            elif _root(mod) in ALLOWED_PACKAGES:
                header.append(ast.unparse(node))
                edits[a] = (b, "")
            else:
                raise SystemExit(f"nn/{src.name}.py:{a}: top-level import of {mod!r} (only stdlib, torch, "
                                 f"transformers may be imported at import time in a Hub export)")
        elif isinstance(node, ast.Assign) and _bound_names(node) == ["__all__"]:
            edits[a] = (b, "")
        elif _is_main_guard(node):
            edits[a] = (b, "")
        elif isinstance(node, ast.ClassDef) and node.name in skip_classes:
            start = min([a] + [d.lineno for d in node.decorator_list])
            edits[start] = (b, f"# {node.name}: see {CONFIG_FILE} (imported at the top of this file)\n")
    top = set(src.tree.body)
    for node in ast.walk(src.tree):
        if node in top or not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        a, b = node.lineno, node.end_lineno
        ind = " " * node.col_offset
        if not src.lines[a - 1][:node.col_offset].isspace() and node.col_offset:
            raise SystemExit(f"nn/{src.name}.py:{a}: an import that shares its line with other code")
        mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
        if getattr(node, "level", 0):
            raise SystemExit(f"nn/{src.name}.py:{a}: relative import")
        if _root(mod) == "nn":
            dep = mod.split(".", 1)[1]
            if dep in vendored:
                alias = "".join(f"{ind}{al.asname} = {al.name}\n" for al in node.names if al.asname)
                edits[a] = (b, alias or f"{ind}pass  # {mod}: in this file\n")
            else:
                edits[a] = (b, f"{ind}raise NotImplementedError(\"this checkpoint's config does not use "
                               f"nn/{dep}.py (left out of the export)\")\n")
        elif _root(mod) not in ALLOWED_PACKAGES and not src.in_try(node):
            stmt = "".join(src.lines[a - 1:b]).strip()
            edits[a] = (b, f"{ind}try:\n{ind}    {stmt}\n{ind}except ImportError as e:  # not part of the Hub export\n"
                           f"{ind}    raise ImportError(\"{mod} belongs to the MrCogito training code ({REPO_URL})\""
                           f") from e\n")
    out, ln = [], 1
    for a in sorted(edits):
        if a < ln:
            raise SystemExit(f"nn/{src.name}.py:{a}: overlapping export edits")
        b, rep = edits[a]
        out.extend(src.lines[ln - 1:a - 1])
        out.append(rep)
        ln = b + 1
    out.extend(src.lines[ln - 1:])
    return "".join(out), header


def _merge_imports(stmts: Iterable[str]) -> str:
    """Top-level import statements → one block: `import x` lines, then one `from m import a, b` line per module."""
    plain, names = set(), {}
    for stmt in stmts:
        node = ast.parse(stmt).body[0]
        if isinstance(node, ast.Import):
            plain.add(stmt)
        else:
            names.setdefault(node.module, set()).update(
                al.name if not al.asname else f"{al.name} as {al.asname}" for al in node.names)
    froms = [f"from {m} import {', '.join(sorted(n))}" for m, n in sorted(names.items())]
    return "\n".join(sorted(plain) + froms)


# ---------------------------------------------------------------------------------------------- the two files
def _provenance() -> dict:
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return ""

    sha = git("rev-parse", "HEAD").strip() or "unknown"
    dirty = bool(git("status", "--porcelain", "--", "nn", "scripts/export_hub_model.py").strip())
    return {"repo": REPO_URL, "commit": sha, "dirty": dirty}


def _banner(title: str, prov: dict, modules: list[str]) -> str:
    commit = prov["commit"][:12] + (" + uncommitted changes" if prov["dirty"] else "")
    files = ", ".join(f"nn/{m}.py" for m in modules)
    text = (f"Generated by scripts/export_hub_model.py from {prov['repo']} at commit {commit}, from {files}. Do not "
            f"edit: change the source and export again. Engineering spec: "
            f"docs/engineering_specs/hub_from_pretrained_loading.md.")
    paras = [textwrap.fill(p, 116) for p in title.split("\n\n")] + [textwrap.fill(text, 116)]
    return '"""' + "\n\n".join(paras) + '\n"""\n'


def build_configuration(prov: dict) -> str:
    """configuration_perceiver_ar.py: `PerceiverARConfig` and every top-level name it reads (the closure over the
    source modules), with the imports those use."""
    srcs = {m: _Source(m) for m in SOURCE_ORDER}
    defs: dict[str, tuple[str, ast.AST]] = {}
    imports: dict[str, str] = {}
    for m, s in srcs.items():
        for node in s.tree.body:
            names = _bound_names(node)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                if _root(mod) in ALLOWED_PACKAGES and mod != "__future__":
                    for al in node.names:
                        imports.setdefault(al.asname or (al.name if isinstance(node, ast.ImportFrom)
                                                         else al.name.split(".")[0]),
                                           ast.unparse(ast.ImportFrom(module=mod, names=[al], level=0))
                                           if isinstance(node, ast.ImportFrom) else ast.unparse(
                                               ast.Import(names=[al])))
                continue
            for n in names:
                defs.setdefault(n, (m, node))
    need: dict[int, tuple[str, ast.AST]] = {}
    todo, seen = [CONFIG_CLASS], set()
    used_imports: list[str] = []
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in defs:
            m, node = defs[name]
            need[id(node)] = (m, node)
            todo.extend(_used_names(node) - seen)
        elif name in imports:
            used_imports.append(imports[name])
    parts = sorted(need.values(), key=lambda mn: (SOURCE_ORDER.index(mn[0]), mn[1].lineno))
    body = ""
    for m, node in parts:
        gap = "\n\n\n" if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else "\n\n"
        body += (gap if body else "") + srcs[m].segment(node).rstrip("\n")
    head = _banner("Perceiver AR (MrCogito) configuration for 🤗 Transformers: the configuration of a Hub release.",
                   prov, sorted({m for m, _ in parts}, key=SOURCE_ORDER.index))
    imps = _merge_imports(used_imports)
    return f"{head}from __future__ import annotations\n\n{imps}\n\n\n{body}\n\n\n__all__ = [\"{CONFIG_CLASS}\"]\n"


def build_modeling(modules: list[str], prov: dict) -> str:
    vendored = set(modules)
    header: list[str] = []
    bodies: list[str] = []
    for m in modules:
        text, imps = _flatten(_Source(m), vendored, skip_classes=(CONFIG_CLASS,) if m == "perceiver_ar_lm" else ())
        header.extend(imps)
        rule = "# " + "=" * 110
        bodies.append(f"{rule}\n# nn/{m}.py\n{rule}\n{text.strip(chr(10))}\n")
    head = _banner("Perceiver AR (MrCogito) for 🤗 Transformers: the modeling code of a Hub release.\n\nEntry point: "
                   f"`{MODEL_CLASS}` (AutoModelForCausalLM; nn/perceiver_ar_hub.py at the end of this file). The "
                   "source modules follow in import order, with their `nn.` imports resolved inside this file.",
                   prov, modules)
    imps = _merge_imports(header)
    return (f"{head}from __future__ import annotations\n\n{imps}\n\nfrom .{CONFIG_FILE[:-3]} import {CONFIG_CLASS}\n\n\n"
            + "\n\n".join(bodies) + f"\n\n__all__ = [\"{CONFIG_CLASS}\", \"{MODEL_CLASS}\", \"PerceiverARCache\"]\n")


def check_code(path: Path) -> list[str]:
    """Compiles, has no `nn.` import left, and asks 🤗's remote-code check only for allowed packages. Returns the
    packages 🤗 will require."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    compile(tree, str(path), "exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _root(node.module) == "nn":
            raise SystemExit(f"{path.name}:{node.lineno}: an nn. import survived the export")
    from transformers.dynamic_module_utils import get_imports

    req = get_imports(str(path))
    bad = sorted(set(req) - ALLOWED_PACKAGES)
    if bad:
        raise SystemExit(f"{path.name}: 🤗 trust_remote_code would require {bad}")
    return req


# ---------------------------------------------------------------------------------------------- config + files
def export_config(cfg: dict, tokenizer_name: str, attn_backend: str) -> tuple[dict, dict]:
    cfg = dict(cfg)
    changes: dict = {}

    def put(key, value):
        if cfg.get(key, "<missing>") != value:
            changes[key] = [cfg.get(key, None), value]
        cfg[key] = value

    put("architectures", [MODEL_CLASS])
    put("auto_map", dict(AUTO_MAP))
    put("use_liger", False)            # Liger's fused loss is a training kernel (Linux / CUDA only)
    put("tokenizer_name", tokenizer_name)
    put("attn_backend", attn_backend)
    put("attn_pad_multiple", 1)
    put("tie_word_embeddings", False)  # TiedFactorizedHead reuses embed.tok itself; 🤗's tying would break it
    return cfg, changes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src: Path, dst: Path, link: bool) -> None:
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def export(checkpoint: str | Path, out_dir: str | Path, *, tokenizer_name: str = HUB_TOKENIZER,
           attn_backend: str = "sdpa", link: bool = False, card: Optional[str | Path] = None,
           overwrite: bool = False) -> dict:
    """Write the Hub folder; returns the manifest."""
    ckpt, out = Path(checkpoint), Path(out_dir)
    cfg = json.loads((ckpt / "config.json").read_text())
    modules = vendored_modules(cfg)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty (pass --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    prov = _provenance()

    (out / CONFIG_FILE).write_text(build_configuration(prov), encoding="utf-8")
    (out / MODELING_FILE).write_text(build_modeling(modules, prov), encoding="utf-8")
    requires = sorted(set(check_code(out / CONFIG_FILE)) | set(check_code(out / MODELING_FILE)))

    new_cfg, changes = export_config(cfg, tokenizer_name, attn_backend)
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2, sort_keys=True) + "\n")
    import transformers

    gen = {"_from_model_config": False, "bos_token_id": cfg.get("bos_token_id"),
           "eos_token_id": cfg.get("eos_token_id"), "pad_token_id": cfg.get("pad_token_id"),
           "transformers_version": transformers.__version__}
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")

    weights = sorted({p for pat in WEIGHT_PATTERNS for p in ckpt.glob(pat)})
    if not any(p.suffix == ".safetensors" or p.name == "pytorch_model.bin" for p in weights):
        raise SystemExit(f"{ckpt}: no weights (model.safetensors / pytorch_model.bin)")
    for p in weights:
        _copy(p, out / p.name, link)
    tok_files = [ckpt / f for f in TOKENIZER_FILES if (ckpt / f).exists()]
    for p in tok_files:
        shutil.copy2(p, out / p.name)
    warnings = []
    if not tok_files:
        warnings.append(f"no tokenizer files in the checkpoint: save {tokenizer_name}'s tokenizer into the folder")
    for p in tok_files:
        if p.suffix == ".json" and ("/opt/" in p.read_text() or "/home/" in p.read_text()):
            warnings.append(f"{p.name} holds a local path")
    if card is not None:
        shutil.copy2(card, out / "README.md")

    files = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "export_manifest.json":
            files[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
    import torch

    manifest = {
        "exported_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": prov,
        "checkpoint": str(ckpt.resolve()),
        "vendored_modules": [f"nn/{m}.py" for m in modules],
        "left_out": [f"nn/{m}.py" for m in SOURCE_ORDER if m not in modules],
        "remote_code_requires": requires,
        "config_changes": changes,
        "auto_map": AUTO_MAP,
        "exported_with": {"transformers": transformers.__version__, "torch": torch.__version__},
        "warnings": warnings,
        "files": files,
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def quick_check(out_dir: str | Path) -> dict:
    """Load the folder as a reader would (local files only, CPU, fp32): one forward with labels, 4 greedy tokens."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(out_dir), trust_remote_code=True, local_files_only=True,
                                                 dtype=torch.float32).eval()
    ids = torch.randint(5, int(model.config.vocab_size), (1, 48), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        loss = float(model(input_ids=ids, labels=ids).loss)
    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False)
    return {"class": type(model).__name__, "module": type(model).__module__, "loss_random_ids": round(loss, 4),
            "generated": gen[0, ids.shape[1]:].tolist()}


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", help="perceiver_ar checkpoint directory (config.json + weights)")
    ap.add_argument("out_dir", help="the Hub folder to write")
    ap.add_argument("--tokenizer_name", default=HUB_TOKENIZER, help="config.tokenizer_name in the export")
    ap.add_argument("--attn_backend", default="sdpa", choices=("sdpa", "flex"),
                    help="default attention backend of the export (from_pretrained(attn_backend=...) overrides)")
    ap.add_argument("--card", default=None, help="model card to copy as README.md")
    ap.add_argument("--link", action="store_true", help="hard-link the weights instead of copying them")
    ap.add_argument("--overwrite", action="store_true", help="write into a non-empty out_dir")
    ap.add_argument("--check", action="store_true", help="load the export on the CPU and generate 4 tokens")
    a = ap.parse_args(argv)
    man = export(a.checkpoint, a.out_dir, tokenizer_name=a.tokenizer_name, attn_backend=a.attn_backend,
                 link=a.link, card=a.card, overwrite=a.overwrite)
    print(json.dumps({k: man[k] for k in ("vendored_modules", "left_out", "remote_code_requires", "config_changes",
                                          "warnings")}, indent=2))
    for name, f in man["files"].items():
        print(f"  {name:32s} {f['bytes']:>12,d} B  {f['sha256'][:12]}")
    if a.check:
        print(json.dumps(quick_check(a.out_dir), indent=2))


if __name__ == "__main__":
    main()
