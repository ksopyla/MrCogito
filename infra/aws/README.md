# AWS training nodes (E33n)

Plain AWS CLI scripts, research account (`m21-research`), us-east-1. On-demand H100 quota: 64 vCPU = four
p5.4xlarge (1 × H100 80 GB, 16 vCPU, 3.8 TB local NVMe, $6.88/h) at once; ordinary instances: 16 vCPU (one
c7i.4xlarge, $0.714/h, for data jobs). Budget for the E33n goal: **$2,000** (author, 2026-11-04).

| script | what it does |
|---|---|
| `setup.sh` | one-time, idempotent: private S3 bucket `mrcogito-train-<account>`, node role (that bucket + SSM), security group (SSH from this machine's current IP only — re-run when the IP changes), key pair `~/.ssh/mrcogito-train.pem` |
| `launch.sh NAME MAX_HOURS JOB [BRANCH] [TYPE]` | starts a node; it clones BRANCH over HTTPS (public repo, no credentials on the node), runs `node_run.sh JOB` in tmux `job`; **powers itself off (= terminates) after MAX_HOURS** whatever happens |
| `node_run.sh JOB` | on the node: spend heartbeat + job log to S3 every 5 min, `uv sync` (unless the job says `# mc: no-sync`), the job, a final sync, power-off |
| `status.sh [--account]` | what runs, for how long, and the spend so far (sum of every node's heartbeat in `s3://…/spend/`) |
| `terminate.sh NAME_OR_ID … \| --all` | terminate nodes tagged `Project=mrcogito` |
| `jobs/*.sh` | the jobs: `data_prep.sh` (CPU), `pipeline_check.sh`, the E33n stage jobs |

Layout in S3: `data/` (token shards, tokenizer), `runs/<run>/` (checkpoints, logs, results), `logs/<node>/job.log`,
`spend/<instance-id>.json`.

Safety nets: every node has a hard lifetime (MaxHours tag + `shutdown -h +N` at boot, shutdown behaviour terminate);
jobs end with a power-off; no inbound port except SSH from one IP; no secrets on the nodes (W&B runs offline, synced
later from the laptop; Hugging Face downloads are anonymous).

```bash
bash infra/aws/status.sh
ssh -i ~/.ssh/mrcogito-train.pem ubuntu@<ip> 'tmux attach -t job'
bash infra/aws/terminate.sh --all
```
