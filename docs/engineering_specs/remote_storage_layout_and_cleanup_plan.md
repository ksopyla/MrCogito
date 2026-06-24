# Remote storage layout & cleanup plan

**Status:** agreed 2026-06-24 (grill-me). **Ops:** manual on servers unless explicitly delegated.

## Progress (2026-06-24)

| Step | Status | Notes |
|------|--------|-------|
| NAS dirs (`mrcogito/*`, `archive_odra`, `archive_polonez`) | **Done** | Created on NAS |
| Checkpoints `training/` → `mrcogito/checkpoints/` | **Done** | 12G, 7 runs |
| Odra `gw` → `archive_odra/home/gw/` | **Done** | NAS archive in place |
| Odra `/home` user cleanup | **Next** | `sudo bash scripts/cleanup_odra_gw_sudo.sh` on Odra |
| Polonez `goodwrite_ml` → sdb | Pending | rsync + symlink |
| Polonez inactive users | Pending | `cleanup_polonez_users_sudo.sh` |

## Storage tiers (both GPU servers)

| Tier | Mount | Role |
|------|-------|------|
| **Hot** | NVMe `/home` | Canonical training paths only (`~/dev/hf_home`, `~/dev/MrCogito/Cache/*`) |
| **Polonez cold (local)** | `/data/mrcogito` (sdb) | **`goodwrite_ml/` only** — symlink from `~/dev/goodwrite_ml` |
| **NAS backup** | `/nas/ml_data` | MrCogito project backups + Odra server archive for `gw` |

Do **not** point `HF_HOME` or `OUTPUT_DIR` at NAS or second disk. NAS is sync/backup, not the training hot path.

## NAS layout (canonical)

```
/nas/ml_data/
├── mrcogito/                          # MrCogito project data ONLY
│   ├── checkpoints/<run_id>/checkpoint-<step>/   # best ckpt per run
│   ├── hf_datasets/                   # optional rsync from ~/dev/hf_home/datasets
│   └── hf_hub/                        # optional rsync from ~/dev/hf_home/hub
├── archive_odra/                      # cold archive from Odra (mirror source paths)
│   └── home/gw/                       # legacy Goodwrite user (archived)
└── archive_polonez/                   # reserved; no current mandatory contents
```

**Not allowed under `mrcogito/`:** retired user homes, other projects, or generic `archive/`.

**Deprecated paths (migrated 2026-06-24):**

| Old path | New path | Status |
|----------|----------|--------|
| `/nas/ml_data/training/*` | `/nas/ml_data/mrcogito/checkpoints/*` | **Done** |
| `/nas/ml_data/mrcogito/archive/gw/` | `/nas/ml_data/archive_odra/home/gw/` | **Done** |

## What to keep vs delete

| Data | Action | Status |
|------|--------|--------|
| MrCogito active runs (NVMe) | Keep best checkpoint only on NVMe; rsync best to `mrcogito/checkpoints/` | Ongoing |
| Odra `/home/gw` (~428G) | NAS archive at `archive_odra/home/gw/`, then **`userdel -r gw`** | Archive **done**; NVMe cleanup **next** |
| Odra other users | **`userdel -r`** all except `ksopyla`, `pgorecki` | Pending |
| Polonez `goodwrite_ml` (~475G) | **Move** to `/data/mrcogito/goodwrite_ml/`, symlink `~/dev/goodwrite_ml` | Pending |
| Polonez inactive users (~111G) | **`userdel -r`** — no NAS copy, no sdb copy | Pending |
| sdb `/data/mrcogito/home_archive/` | **Delete** (partial mistaken rsync) | Pending |

**Odra accounts to remove:** `bjankowski`, `car_recognition`, `ermlab`, `fiona`, `gw`, `jlewalski`, `kropiak`, `lsawaniewski`, `marcel`, `mmelzacki`, `pszocik`, `sdarmofal`, `sjankowski`, `szrek`, `wlipnicki`.

**Odra accounts to keep:** `ksopyla`, `pgorecki`.

**Polonez accounts to remove:** `jlewalski`, `mwrobel`, `bmielczarek`, `kropiak`, `sidziniak`, `kfuchsig`.

## NVMe canonical paths (unchanged)

```bash
PROJECT_ROOT="/home/ksopyla/dev/MrCogito"
HF_HOME="${PROJECT_ROOT}/../hf_home"
OUTPUT_DIR="${PROJECT_ROOT}/Cache/Training"
LOGGING_DIR="${PROJECT_ROOT}/Cache/logs"
```

Launchers source `scripts/remote_paths.sh`.

---

## Execution checklist

Run long jobs **inside byobu/tmux on the server** — not over SSH from Mac.

### Step 0 — NAS directories — **DONE**

```bash
mkdir -p /nas/ml_data/mrcogito/{checkpoints,hf_datasets,hf_hub}
mkdir -p /nas/ml_data/archive_odra/home
mkdir -p /nas/ml_data/archive_polonez/home
```

### Step 1 — NAS path fixes — **DONE**

```bash
mkdir -p /nas/ml_data/mrcogito/checkpoints
mv /nas/ml_data/training/* /nas/ml_data/mrcogito/checkpoints/
rmdir /nas/ml_data/training

mkdir -p /nas/ml_data/archive_odra/home
mv /nas/ml_data/mrcogito/archive/gw /nas/ml_data/archive_odra/home/gw
rmdir /nas/ml_data/mrcogito/archive 2>/dev/null || true
```

**Verify (already passed):**

```bash
du -sh /nas/ml_data/mrcogito/checkpoints /nas/ml_data/archive_odra/home/gw
ls /nas/ml_data/mrcogito/checkpoints/
test -d /nas/ml_data/archive_odra/home/gw && echo "gw archive OK"
```

### Step 2 — Odra: clean `/home` — **NEXT**

Removes all user accounts except `ksopyla` and `pgorecki`. For `gw`, checks that the NAS
archive exists at `archive_odra/home/gw/` before deleting `/home/gw` (~428G NVMe). No
symlinks, no NAS rsync in this script.

```bash
ssh odra
sudo bash ~/dev/MrCogito/scripts/cleanup_odra_gw_sudo.sh
```

Expected: only `/home/ksopyla` and `/home/pgorecki` remain; NVMe `/home` frees several hundred GB.

### Step 3 — Polonez: `goodwrite_ml` → sdb (~475G)

```bash
ssh polonez
byobu  # or tmux

rsync -aH --info=progress2 ~/dev/goodwrite_ml/ /data/mrcogito/goodwrite_ml/

# when rsync completes:
bash ~/dev/MrCogito/scripts/finish_goodwrite_symlink.sh
# verify goodwrite paths, then:
rm -rf ~/dev/goodwrite_ml.bak
```

### Step 4 — Polonez: delete inactive users (~111G NVMe)

```bash
ssh polonez
sudo bash ~/dev/MrCogito/scripts/cleanup_polonez_users_sudo.sh
```

Also removes partial sdb `home_archive/` if present.

### Step 5 — Optional later

- Rsync `~/dev/hf_home/datasets` → `/nas/ml_data/mrcogito/hf_datasets/` (when you want NAS backup of tokenized corpora)
- Rsync `~/dev/hf_home/hub` → `/nas/ml_data/mrcogito/hf_hub/`
- Odra `~/dev/hf_home` (147G FineWeb/MiniPile) — trim only if confirmed unused

---

## Post-cleanup verification

```bash
# Polonez
df -h /home /data/mrcogito
du -sh ~/dev/MrCogito/Cache/Training ~/dev/goodwrite_ml /data/mrcogito/goodwrite_ml
ls -la ~/dev/goodwrite_ml

# Odra
df -h /home
ls /home   # expect: ksopyla pgorecki only
du -sh ~/dev/MrCogito/Cache/Training /nas/ml_data/archive_odra/home/gw
test ! -e /home/gw && echo "gw home removed OK"

# NAS
du -sh /nas/ml_data/mrcogito/checkpoints /nas/ml_data/archive_odra/home/gw
```

## Repo artifacts

| File | Purpose |
|------|---------|
| `scripts/remote_paths.sh` | Canonical NVMe env vars for launchers |
| `scripts/finish_goodwrite_symlink.sh` | After goodwrite rsync to sdb |
| `scripts/cleanup_odra_gw_sudo.sh` | Odra: `userdel -r` all users except `ksopyla`/`pgorecki`; gw requires NAS archive |
| `scripts/cleanup_polonez_users_sudo.sh` | Polonez: `userdel -r` inactive users + wipe sdb mistake |
| `.cursor/skills/remote-servers/SKILL.md` | Live ops reference (must match this spec) |

## Workflow after cleanup

1. Train on NVMe canonical paths.
2. When a run finishes, rsync **best checkpoint only** to `/nas/ml_data/mrcogito/checkpoints/<run_id>/`.
3. Delete intermediate checkpoints and stale runs on NVMe.
4. Server/user cold storage → `archive_<hostname>/` mirroring source paths — **not** under `mrcogito/`.
