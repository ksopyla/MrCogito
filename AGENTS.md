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

## Claude Code cloud-session specific instructions

Claude Code cloud sessions (claude.ai/code, the mobile/desktop apps, `claude --cloud`,
routines) run in an isolated Anthropic-managed VM. Like Cursor Cloud Agents they do
**not** inherit the author's laptop `~/.ssh`, VPN, or the gitignored `remote-servers`
skill — but the network model is different, and the Cursor recipe does not port over
unchanged.

### The transport constraint (read this before configuring keys)

All outbound traffic leaves through an HTTP/HTTPS proxy that enforces a **domain
allowlist**. It is not a firewall with ports you can open:

- **Port 22 is unreachable.** A plain TCP connection to `:22` never leaves the sandbox.
- A `CONNECT` to a host outside the environment's allowlist is answered `403 Forbidden`.
- `openssh-client` is **not** pre-installed on the VM.

So SSH keys alone are not sufficient. `ssh odra` works from a cloud session only when
the `Host` block targets a hostname that is **(a)** on the environment's allowlist and
**(b)** reachable over port 443. Two workable shapes:

- **Cloudflare Tunnel + Access** — run `cloudflared` on the GPU hosts; no inbound port is
  opened. The `Host` block uses
  `ProxyCommand cloudflared access ssh --hostname %h`, authenticated with a scoped
  service token. Preferred: the servers stay off the public internet.
- **sshd (or `sslh`) on 443** behind a public hostname, tunnelled through the proxy's
  `CONNECT`. Less infrastructure, but it exposes `sshd` to the internet.

A third requirement is easy to miss: **`ssh` does not read `HTTPS_PROXY`.** Left to
itself it opens a direct TCP connection, which the sandbox drops. Every `Host` block
therefore needs

```
ProxyCommand /path/to/repo/.claude/scripts/https-proxy-connect.py %h %p
```

That helper speaks `CONNECT` on ssh's behalf and shuttles bytes. It enforces nothing and
bypasses nothing — the proxy still decides and still answers 403 for a host the policy
excludes; the helper just reports *which* layer said no. With no `HTTPS_PROXY` in the
environment it connects directly, so the same `SSH_CONFIG` also works on the laptop.

**Test the cheap thing first.** There is no port setting anywhere in Claude Code —
`Network access` is a hostname allowlist, nothing more. So before re-plumbing a router,
allowlist the existing hostname and try the port you already forward:

```bash
.claude/scripts/https-proxy-connect.py <hostname> <existing-ssh-port> < /dev/null
```

- silence or a hang → the tunnel opened; the port is fine, keep the current forwards
- `403` → the proxy refused. If the hostname is definitely allowlisted, the **port** is
  what it objects to, and SSH needs to move to 443

**Still unverified:** whether the proxy passes raw SSH bytes through an established
tunnel, or breaks the handshake with TLS interception. Confirm with
`ssh -v odra 'hostname'` before depending on this path.

### Reaching two hosts behind one public IP

Where two servers sit behind one router on different forwarded ports, do not expose a
second 443. Give the first host the 443 entry and reach the second over the LAN through
it, which also matches the existing "treat the other box as a jump host" convention:

```
Host <primary>
  HostName <ddns-hostname>
  Port 443
  User <user>
  IdentityFile ~/.ssh/id_ed25519
  ProxyCommand /path/to/repo/.claude/scripts/https-proxy-connect.py %h %p

Host <secondary>
  HostName <secondary-lan-ip>
  Port 22
  User <user>
  IdentityFile ~/.ssh/id_ed25519
  ProxyJump <primary>
```

Only the primary is exposed publicly; the secondary is reachable only from inside the
LAN. Dynamic DNS is no obstacle — the allowlist matches the *hostname*, so a changing
public IP is irrelevant. `SSH_KNOWN_HOSTS` then needs an entry for both: generate the
secondary's by running `ssh-keyscan` from the primary.

If neither shape is set up, remote GPU work is simply unavailable in a cloud session —
use Cursor Cloud, a local session, or Remote Control on an always-on machine instead.
Do not burn turns retrying `ssh`; report the blocker.

### Configuration (one-time, per cloud environment)

At [claude.ai/code](https://claude.ai/code) → environment settings:

1. **Environment variables** (`.env` format):
   - `SSH_PRIVATE_KEY` — dedicated cloud key
   - `SSH_CONFIG` — `Host odra` / `Host polonez` blocks (HostName, User, Port, IdentityFile,
     and the `ProxyCommand` if using Cloudflare)
   - `SSH_KNOWN_HOSTS` — **required in the cloud.** `ssh-keyscan` dials port 22 and cannot
     run here; generate the lines locally with `ssh-keyscan -p <port> <hostname>`.
   - `WANDB_API_KEY` — the W&B MCP server (`.mcp.json`) fails to start without it, because
     there is no `.env` in a fresh clone. Optionally `HF_TOKEN`.
2. **Setup script** (provisions the VM, cached):
   ```bash
   apt-get update -qq && apt-get install -y -qq openssh-client || true
   ```
3. **Network access**: `Custom`, listing the hostname(s) from `SSH_CONFIG` (keep the
   default package-registry list checked so `uv` still works).

`.claude/scripts/cloud-session-ssh-install.sh` then materializes the key, config and
known_hosts into `~/.ssh` on every session. It is wired as a `SessionStart` hook in
`.claude/settings.json`, and is a silent no-op in local sessions, so it never touches a
real `~/.ssh` on the laptop.

> **Secrets warning — differs from Cursor.** Cursor Runtime Secrets are redacted; Claude
> cloud-environment variables are **not** — anyone who can use the environment, and the
> agent itself, can read them. Use a **dedicated** key, restricted server-side to its own
> user and/or a `restrict`/`command="…"` entry in `authorized_keys`. Never the author's
> personal key. The redacted "API credentials" store only injects HTTP headers and cannot
> carry an SSH key.

### Remote workflow once SSH is up

Unchanged from the `experiment-run` skill: sync source by **Git only**, launch under
Byobu, then poll. One-shot non-interactive commands, never an interactive session:

```bash
ssh odra 'cd ~/dev/MrCogito && git pull && byobu new-session -d -s e18 "bash training/<launcher>.sh"'
ssh odra 'nvidia-smi; tail -n 40 ~/dev/MrCogito/Cache/logs/shell_*.log'
```

Delegate the noisy monitoring to the `server-diagnostics` subagent so SSH/log output stays
out of the main context. `ssh odra:*` and `ssh polonez:*` are already pre-approved in
`.claude/settings.json`.

### Skills and agents in the cloud

- Committed skills under `.cursor/skills/` resolve normally (`.claude/skills` is a
  symlink, and git preserves it).
- `.cursor/skills/remote-servers/` is gitignored and **absent** in the cloud. Host/port
  details come from the injected `SSH_CONFIG`, never from public repo files.
- Never print, commit, or log private keys, HostName, or Port values.
