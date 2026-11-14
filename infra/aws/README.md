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

**Spot L40S nodes, one model on several GPUs** (2026-11-06; us-east-1 had no on-demand H100 from 2026-11-04 and the
G-family on-demand quota is 0). Spot quota: G family 48 vCPU = one `g6e.12xlarge` (4 × L40S 48 GB, ~$5.7/h spot vs
$10.49 on demand, about the price of half an H100-hour per GPU); P family 8 vCPU = no spot H100. Launch with
`MC_SPOT=1 bash infra/aws/launch.sh NAME MAX_HOURS "infra/aws/jobs/e33p_s1_multi.sh SEED ARM …" BRANCH g6e.12xlarge`:
each arm trains on all four GPUs (`E33N_GPUS`, torchrun, the same rows per optimizer step), the three evals run side
by side, and checkpoints go to S3 (`E33N_CKPT_S3=1`), so if AWS takes the node back the same launch command resumes
from the newest checkpoint and skips finished arms.

Safety nets: every node has a hard lifetime (MaxHours tag + `shutdown -h +N` at boot, shutdown behaviour terminate);
jobs end with a power-off; no inbound port except SSH from one IP; no secrets on the nodes (W&B runs offline, synced
later from the laptop; Hugging Face downloads are anonymous).

```bash
bash infra/aws/status.sh
ssh -i ~/.ssh/mrcogito-train.pem ubuntu@<ip> 'tmux attach -t job'
bash infra/aws/terminate.sh --all
```
