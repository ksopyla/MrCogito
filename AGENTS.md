# MrCogito — agent memory (Cursor + Claude Code)

> **Vision:** compress long context into latent *concept* vectors, reason in that
> concept space, decode back to text (audio later). Live agenda:
> `docs/1_Strategy_and_Plans/agenda.md`.

Canonical rules/skills live under `.cursor/` (`.claude/skills` is a symlink).
Project overview and local env conventions are in `.cursor/rules/`.

## Cursor Cloud specific instructions

Cloud Agents run in an isolated Cursor VM. They do **not** inherit the author's
laptop `~/.ssh`, VPN, or the gitignored `remote-servers` skill.

### SSH to GPU servers (odra / polonez)

1. Set these **Runtime Secrets** in
   [Cloud Agents → Secrets](https://cursor.com/dashboard/cloud-agents)
   (HostName/Port/User values live only in the dashboard and local gitignored
   inventory — never commit them):
   - `SSH_PRIVATE_KEY` — dedicated Cloud Agent private key
   - `SSH_CONFIG` — `Host odra` / `Host polonez` blocks (with HostName, User, Port, IdentityFile)
   - optional: `SSH_KNOWN_HOSTS` — precomputed `known_hosts` lines
2. The install script (`.cursor/scripts/cloud-agent-ssh-install.sh`, via
   `.cursor/environment.json`) materializes those secrets into `~/.ssh/` on the VM.
3. Connect with `ssh odra` or `ssh polonez`. Project layout and run workflow:
   `experiment-run` skill.
4. Smoke test: `ssh odra 'hostname; nvidia-smi -L'`
5. Never print, commit, or log private keys, HostName, or Port values. Prefer
   Runtime Secrets (redacted) over plain Environment Variables.

### Skills available in Cloud

- Use committed skills under `.cursor/skills/` (`experiment-run`,
  `experiment-evaluate`, `experiment-track`, …).
- Do **not** expect `.cursor/skills/remote-servers/` — it is gitignored. Resolve
  HostName/Port/LAN details from Dashboard Secrets + local `remote-servers` /
  `~/.ssh/config`, not from public repo files.

### Network

Allow outbound reachability to whatever HostName(s) appear in the injected
`SSH_CONFIG` (or use Allow all for a first test). Private LAN addresses are not
reachable without a tunnel (Tailscale / Cloudflare) configured in the Cloud env.

### Local one-time author setup

On the laptop (not in Cloud), using values from your local `~/.ssh/config` and
gitignored `remote-servers` skill:

```bash
bash .cursor/scripts/cloud-agent-ssh-setup.sh --install --print-secret --print-config
```

Paste `SSH_PRIVATE_KEY` and `SSH_CONFIG` into Dashboard Secrets, then clear the
terminal scrollback. `CURSOR_API_KEY` in `.env` only authenticates the Cursor
API/SDK; it does **not** configure Dashboard Secrets.
