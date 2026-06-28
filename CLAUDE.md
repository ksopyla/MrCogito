# MrCogito — Claude Code project memory

This repository is developed with **both Cursor and Claude Code**. The `.cursor/` tree is
**canonical** for rules, skills, and agents; `.claude/` mirrors it so Claude Code discovers the
same surfaces without duplicating content. **Edit skills in `.cursor/skills/`** and both editors
see the change — there is one source of truth, never two.

> **Vision:** compress long context into latent *concept* vectors, reason in that concept space,
> decode back to text (audio later). Full README at repo root; live agenda at
> `docs/1_Strategy_and_Plans/agenda.md` (the slim daily driver — read it for current focus).

## How this checkout is wired (Claude Code side)
- **Skills:** `.claude/skills` is a symlink → `../.cursor/skills`. Every skill resolves to the
  single canonical copy. (`remote-servers` is gitignored because it holds server/infra details;
  it exists locally only — same as under Cursor.)
- **Agents:** `.claude/agents/*.md` are thin pointers — Claude-native frontmatter (`model`,
  `tools`) plus an instruction to read the canonical protocol in `.cursor/agents/`. The real
  prompts live in `.cursor/`.
- **MCP:** the W&B MCP server is declared in `.mcp.json` (project root) and reuses the shared
  `.cursor/scripts/wandb-mcp.sh`. Claude Code asks before enabling it on first use.
- **Settings:** `.claude/settings.json` holds a shared permissions allowlist;
  `.claude/settings.local.json` is auto-managed (your per-session allow/deny decisions).

## Project context — imported verbatim from the canonical Cursor rules
The two blocks below are the exact content Cursor auto-applies (`alwaysApply: true`), imported
here so Claude and Cursor can never disagree on the ground truth. (Ignore the `---` frontmatter
lines — they are Cursor metadata.)

@.cursor/rules/project-overview.mdc
@.cursor/rules/local-environment.mdc
