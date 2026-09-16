"""Verified 1-token atom tables for CogitoProbe rows.

Rows are *composed* from these ids. We never BPE-encode a glued arithmetic string:
the tokenization probe showed SmolLM3 merges 96% of glued ``+ - * ( ) [ ] { }``
expressions (0.76 tokens/atom) and splits space-separated digits (1.28 tokens/atom).
Bare digits and operators are atomic; injecting those ids is the only 1:1 layout.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from data.concept_probes.schema import ARITH_ATOMS, DEFAULT_TOKENIZER

# Distinctive marker words. Tried in order; first 1-token encoding wins.
MARKER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "key": ("key", "Key"),
    "val": ("val", "Val"),
    "query": ("Q", "query"),
    "answer": ("A", "answer"),
    "end": ("END", "end"),
    "dot": (".", ";"),
    "who": ("who", "Who"),
    "hop": ("hop", "Hop"),
    "eval": ("eval", "Eval"),
    "sub": ("sub", "Sub"),
    "match": ("match", "Match"),
    "fact": ("fact", "Fact"),
    "friend": ("friend", "Friend"),
    "color": ("color", "Color"),
    "place": ("place", "Place"),
    "job": ("job", "Job"),
    "is": ("is",),
    "in": ("in",),
    "the": ("the",),
    "a": ("a",),
    "dropped": ("dropped",),
    "found": ("found",),
    "gave": ("gave",),
    "near": ("near",),
    "to": ("to",),
}

# Closed vocab seeds. Leading-space single-token filter keeps only those the tokenizer
# actually represents as one id (Llama-3 Ġword). Order is preference, not identity.
PREFERRED_WORDS: tuple[str, ...] = (
    "red", "blue", "green", "black", "white", "yellow", "purple", "brown",
    "pink", "gold", "silver", "orange", "coral", "amber", "ivory", "olive",
    "cat", "dog", "wolf", "bear", "lion", "bird", "fish", "deer",
    "horse", "mouse", "frog", "hawk", "seal", "crab", "moth", "swan",
    "rome", "oslo", "paris", "cairo", "delhi", "tokyo", "lima", "bern",
    "miami", "seoul", "riga", "minsk", "accra", "doha", "bali", "fiji",
    "miner", "baker", "pilot", "nurse", "judge", "guard", "scout", "clerk",
    "actor", "poet", "chef", "guide", "agent", "rider", "mason", "smith",
    "coin", "lamp", "book", "ring", "key", "map", "cup", "hat",
    "stone", "rope", "box", "door", "gate", "bell", "flag", "mask",
    "luna", "moss", "iris", "jade", "opal", "ruby", "pearl", "onyx",
    "nova", "vega", "mira", "lyra", "rory", "nina", "omar", "lars",
    "wind", "rain", "snow", "mist", "fire", "water", "earth", "light",
    "moved", "closed", "waited", "changed", "opened", "passed", "turned", "stayed",
    "someone", "nobody", "people", "something", "nothing", "always", "never", "maybe",
)


@dataclass(frozen=True)
class Atom:
    name: str
    token_id: int
    surface: str  # canonical decoded/stripped form used in readable text


@dataclass
class AtomTable:
    tokenizer_name: str
    arith: dict[str, Atom]
    markers: dict[str, Atom]
    words: tuple[Atom, ...]  # disjoint from markers/arith
    pad_id: int
    bos_id: int | None
    eos_id: int | None
    vocab_size: int
    pools: dict[str, tuple[Atom, ...]] = field(default_factory=dict)

    def tid(self, kind: str, name: str) -> int:
        if kind == "arith":
            return self.arith[name].token_id
        if kind == "marker":
            return self.markers[name].token_id
        raise KeyError(kind)

    def marker(self, name: str) -> int:
        return self.markers[name].token_id

    def marker_surface(self, name: str) -> str:
        return self.markers[name].surface

    def encode_atoms(self, pieces: list[tuple[str, str]]) -> list[int]:
        ids = []
        for kind, name in pieces:
            if kind == "arith":
                ids.append(self.arith[name].token_id)
            elif kind == "marker":
                ids.append(self.markers[name].token_id)
            elif kind == "word":
                # name is the surface; look up by surface in words index
                ids.append(self._word_by_surface[name].token_id)
            else:
                raise KeyError(kind)
        return ids

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_word_by_surface",
            {w.surface: w for w in self.words},
        )

    def word_id(self, surface: str) -> int:
        return self._word_by_surface[surface].token_id

    def surface_of(self, token_id: int) -> str:
        if token_id == self.pad_id:
            return ""
        if token_id == self.bos_id:
            return ""
        if token_id == self.eos_id:
            return ""
        rev = getattr(self, "_id_to_surface", None)
        if rev is None:
            rev = {self.pad_id: ""}
            for a in self.arith.values():
                rev[a.token_id] = a.surface
            for a in self.markers.values():
                rev[a.token_id] = a.surface
            for a in self.words:
                rev[a.token_id] = a.surface
            object.__setattr__(self, "_id_to_surface", rev)
            rev = self._id_to_surface
        return rev.get(int(token_id), f"#{token_id}")

    def render(self, ids: list[int]) -> str:
        parts = [self.surface_of(i) for i in ids]
        return " ".join(p for p in parts if p)


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def _decode_one(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _first_atomic(tokenizer, candidates: tuple[str, ...]) -> tuple[str, int] | None:
    """Return (surface, id) for the first candidate that is exactly one token.

    Prefer the bare form; fall back to a leading-space form (Llama-3 Ġword).
    """
    for word in candidates:
        for surface in (word, " " + word):
            ids = _encode(tokenizer, surface)
            if len(ids) == 1:
                return word, ids[0]
    return None


def _scan_leading_space_words(tokenizer, model_vocab_size: int) -> list[Atom]:
    if hasattr(tokenizer, "probe_word_atoms"):
        return list(tokenizer.probe_word_atoms)
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    out: list[Atom] = []
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    if vocab:
        for token, token_id in vocab.items():
            tid = int(token_id)
            if tid in special or tid >= model_vocab_size:
                continue
            # Llama-3 / SmolLM3: leading space is Ġ in the token string.
            if token.startswith("Ġ"):
                word = token[1:]
            elif token.startswith(" "):
                word = token[1:]
            else:
                continue
            if not re.fullmatch(r"[A-Za-z]{2,16}", word):
                continue
            ids = _encode(tokenizer, " " + word)
            if ids == [tid]:
                out.append(Atom(name=word, token_id=tid, surface=word.lower()))
        return out
    n = min(len(tokenizer), model_vocab_size)
    for token_id in range(n):
        if token_id in special:
            continue
        decoded = _decode_one(tokenizer, token_id)
        word = decoded.strip()
        if not re.fullmatch(r"[A-Za-z]{2,16}", word):
            continue
        ids = _encode(tokenizer, " " + word)
        if ids == [token_id]:
            out.append(Atom(name=word, token_id=int(token_id), surface=word.lower()))
    return out


DEFAULT_POOL_PLAN: tuple[tuple[str, int], ...] = (
    ("colors", 32),
    ("places", 32),
    ("jobs", 32),
    ("agents", 64),
    ("objects", 48),
    ("values", 32),
    ("keys", 256),
    ("filler", 64),
)


def build_atom_table(
    tokenizer,
    *,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    model_vocab_size: int | None = None,
    seed: int = 0,
    pool_plan: tuple[tuple[str, int], ...] | None = None,
) -> AtomTable:
    """Build a 1-token atom table or raise if the tokenizer cannot support the series."""
    import numpy as np

    vocab_size = int(model_vocab_size or len(tokenizer))
    arith: dict[str, Atom] = {}
    for sym in ARITH_ATOMS:
        ids = _encode(tokenizer, sym)
        if len(ids) != 1:
            raise ValueError(
                f"arith atom {sym!r} is not a single token under {tokenizer_name} "
                f"(ids={ids}). The series requires 1:1 atoms."
            )
        arith[sym] = Atom(name=sym, token_id=ids[0], surface=sym)

    markers: dict[str, Atom] = {}
    used_ids = {a.token_id for a in arith.values()}
    for name, cands in MARKER_CANDIDATES.items():
        hit = _first_atomic(tokenizer, cands)
        if hit is None:
            raise ValueError(f"no 1-token encoding for marker {name} among {cands}")
        surface, tid = hit
        if tid in used_ids:
            # Try remaining candidates.
            found = False
            for word in cands:
                again = _first_atomic(tokenizer, (word,))
                if again and again[1] not in used_ids:
                    surface, tid = again
                    found = True
                    break
            if not found:
                raise ValueError(f"marker {name} collides with an arith/marker id {tid}")
        markers[name] = Atom(name=name, token_id=tid, surface=surface)
        used_ids.add(tid)

    scanned = _scan_leading_space_words(tokenizer, vocab_size)
    # Prefer listed words whose leading-space form is a single unused id.
    preferred: list[Atom] = []
    seen_surfaces: set[str] = {m.surface.lower() for m in markers.values()}
    for word in PREFERRED_WORDS:
        ids = _encode(tokenizer, " " + word)
        if len(ids) != 1:
            continue
        tid = ids[0]
        if tid in used_ids:
            continue
        surface = word.lower()
        if surface in seen_surfaces:
            continue
        preferred.append(Atom(name=word, token_id=tid, surface=surface))
        used_ids.add(tid)
        seen_surfaces.add(surface)

    extras = [
        a for a in scanned
        if a.token_id not in used_ids and a.surface not in seen_surfaces
    ]
    words = tuple(preferred + extras)
    if len(words) < 40:
        raise ValueError(
            f"tokenizer {tokenizer_name} only yielded {len(words)} word atoms; need ≥40"
        )

    pad_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = int(bos_id) if bos_id is not None else None
    eos_id = int(eos_id) if eos_id is not None else None

    rng = np.random.default_rng(seed)
    # Deterministic pool split. Pools are disjoint.
    order = list(words)
    # Keep preferred words first so colors/places stay readable; shuffle only extras.
    n_pref = len(preferred)
    head, tail = order[:n_pref], order[n_pref:]
    rng.shuffle(tail)
    order = head + tail

    def take(n: int) -> tuple[Atom, ...]:
        nonlocal order
        if len(order) < n:
            raise ValueError(f"need {n} more word atoms, have {len(order)}")
        chunk, order = order[:n], order[n:]
        return tuple(chunk)

    plan = pool_plan or DEFAULT_POOL_PLAN
    pools = {}
    for name, n in plan:
        pools[name] = take(n)
    return AtomTable(
        tokenizer_name=tokenizer_name,
        arith=arith,
        markers=markers,
        words=words,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        vocab_size=vocab_size,
        pools=pools,
    )


def int_to_arith_atoms(n: int) -> list[str]:
    """Encode an integer as bare arith atoms (optional leading '-', then digits)."""
    if n < 0:
        return ["-", *list(str(abs(int(n))))]
    return list(str(int(n)))


def prize_bits_uniform(n_query: int, n_choices: int) -> float:
    if n_choices < 2:
        return 0.0
    return float(n_query) * math.log2(n_choices)
