"""Paired three-model table for the E37c follow-up: untuned, cut-tuned and no-cut-tuned models on the same seeded probe documents.
Usage: python3 scripts/a2a_paired_compare.py REEXAM_DIR NOCUT_DIR  (e.g. docs/2_Experiments_Registry/results/evaluations/E37c_split_ft/reexam_seed2026 .../control_nocut)
Per cell and condition: found counts of untuned / cut-tuned / no-cut-tuned on the same documents, plus McNemar
exact p for cut vs no-cut and pooled R1 / R2 lines from the spec."""
import json
import sys
from math import comb
from pathlib import Path

re_dir, nc_dir = Path(sys.argv[1]), Path(sys.argv[2])


def cells(*ps):
    out = {}
    for p in ps:
        if p.exists():
            out.update({(c["task"], c["length"]): c for c in json.load(open(p))["cells"]})
    return out


M = {"untuned": cells(re_dir / "split_base.json", re_dir / "split_base_64k.json"),
     "cut": cells(re_dir / "split_cut.json", re_dir / "split_cut_64k.json"),
     "nocut": cells(nc_dir / "split_tuned.json", nc_dir / "split_tuned_64k.json")}


def hit(reply, refs):
    reply = (reply or "").strip().lower()
    return sum(1.0 for r in refs if r.lower() in reply) / len(refs) >= 1.0


def rows(model, key, cond):
    c = M[model].get(key)
    if not c:
        return None
    return {r["index"]: (hit(r[cond], r["ref"]), tuple(r["ref"])) for r in c["rows"] if cond in r}


def mcnemar(b, c):  # exact two-sided on discordant pairs
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(keys, cond, a="cut", b="nocut"):
    disc_ab = disc_ba = n = fa = fb = 0
    for key in keys:
        ra, rb = rows(a, key, cond), rows(b, key, cond)
        if not ra or not rb:
            continue
        for i in ra.keys() & rb.keys():
            assert ra[i][1] == rb[i][1], f"documents differ at {key} row {i}: {ra[i][1]} vs {rb[i][1]}"
            n += 1
            fa += ra[i][0]
            fb += rb[i][0]
            disc_ab += ra[i][0] and not rb[i][0]
            disc_ba += rb[i][0] and not ra[i][0]
    return n, fa, fb, mcnemar(disc_ab, disc_ba)


conds = ["joint", "split", "tail16", "tail32", "tail64", "warm", "bits8", "bits4", "alone", "swap"]
keys = sorted(set().union(*[set(m) for m in M.values()]))
print("found / n on the same documents: untuned | cut-tuned | no-cut-tuned   (p: cut vs no-cut, McNemar exact)")
for key in keys:
    print(f"\n{key[0]} {key[1]}")
    for cond in conds:
        parts = []
        for m in M:
            r = rows(m, key, cond)
            parts.append("   -  " if r is None else f"{sum(v[0] for v in r.values()):>2}/{len(r):<2} ")
        n, fa, fb, p = compare([key], cond)
        if any(p_.strip() != "-" for p_ in parts):
            print(f"  {cond:<7} " + " | ".join(parts) + (f"   p={p:.3f}" if n else ""))

long_keys = [("niah_single_1", L) for L in (16384, 32768, 65536)]
num_keys = [("niah_single_1", L) for L in (4096, 16384, 32768, 65536)]
for label, ks, cond, a, b in [("R1 memory only 16k-64k, cut vs no-cut", long_keys, "split", "cut", "nocut"),
                              ("R2 one agent 4k-64k, no-cut vs cut", num_keys, "joint", "nocut", "cut"),
                              ("   one agent 4k-64k, untuned vs cut", num_keys, "joint", "untuned", "cut"),
                              ("   one agent 4k-64k, untuned vs no-cut", num_keys, "joint", "untuned", "nocut"),
                              ("   memory only 16k-64k, cut vs untuned", long_keys, "split", "cut", "untuned"),
                              ("   memory only 16k-64k, no-cut vs untuned", long_keys, "split", "nocut", "untuned")]:
    n, fa, fb, p = compare(ks, cond, a, b)
    if n:
        print(f"{label}: {a} {fa}/{n} ({100*fa/n:.0f}%) vs {b} {fb}/{n} ({100*fb/n:.0f}%), "
              f"diff {100*(fa-fb)/n:+.0f} points, p={p:.4f}")

print()
all_num = [("niah_single_1", L) for L in (4096, 16384, 32768, 65536)]
for a, b in [("cut", "untuned"), ("nocut", "untuned"), ("cut", "nocut")]:
    n, fa, fb, p = compare(all_num, "split", a, b)
    if n:
        print(f"memory only 4k-64k number haystack: {a} {fa}/{n} vs {b} {fb}/{n}, diff {100*(fa-fb)/n:+.0f} points, p={p:.4f}")
    n, fa, fb, p = compare([("niah_single_2", 4096)], "split", a, b)
    if n:
        print(f"memory only 4k essay haystack: {a} {fa}/{n} vs {b} {fb}/{n}, p={p:.4f}")
print("\nreplies that open with a blank line (':' then newline(s)), number haystack 4k-64k:")
for m in M:
    for cond in ("joint", "split"):
        tot = blank = blank_miss = miss = 0
        for key in all_num:
            c = M[m].get(key)
            if not c:
                continue
            for r in c["rows"]:
                rep = r.get(cond) or ""
                ok = hit(rep, r["ref"])
                bl = rep.lstrip(":").startswith("\n")
                tot += 1; blank += bl; miss += not ok; blank_miss += bl and not ok
        if tot:
            print(f"  {m:<8} {cond:<6} {blank:>3}/{tot} open blank; misses {miss}, of them blank-first {blank_miss}")
