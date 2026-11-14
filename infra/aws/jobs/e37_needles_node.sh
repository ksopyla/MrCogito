#!/bin/bash
# E37 needles pass (author 2026-11-13: "Yes, start checks", relayed by the "100M model review and next steps" session):
# RULER at 64k on all 100 documents per task (the later pass scored the first 25, uncached) and at 128k, for the
# full-budget models, our models with the cached decoding (nn/incremental.py; a model the cache does not cover is
# skipped, not run on the slow loop), the original LSA on its own (uncached) path. RULER 4k with the cache as well, to
# set against each model's stage-B 4k (old loop, the same 100 documents). ARM:ntk: the long-document loss at an
# eval-time RoPE base (the dense_rope control). Pairs in the run's needles/ folder (later/ keeps the old-loop pairs),
# merged into eval_ruler_needles.json; node markers needles/DONE_<node> or needles/INCOMPLETE_<node>.
# A launch.sh job for one-L40S spot nodes (g6e.xlarge / g6e.2xlarge), e.g.
#   MC_SPOT=1 bash infra/aws/launch.sh e37-needles-e37a 14 \
#     "infra/aws/jobs/e37_needles_node.sh e37a:4096,65536 e37a:131072" e37-needles g6e.xlarge
export E37_RULER_SUB=needles E37_RULER_MERGED=eval_ruler_needles.json E37_REQUIRE_CACHE=1
exec bash "$(dirname "$0")/e37_ruler_later_node.sh" "$@"
