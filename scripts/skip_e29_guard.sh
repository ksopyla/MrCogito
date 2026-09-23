# Polonez owns Hub bind (E29). Source after ROOT is set.
#   SKIP_E29=1 bash scripts/launch_e29.sh
#   touch "$ROOT/Cache/logs/SKIP_E29"
e29_skip_requested() {
  [ "${SKIP_E29:-0}" = "1" ] || [ -f "${ROOT}/Cache/logs/SKIP_E29" ]
}

e29_park_if_bind() {
  if ! e29_skip_requested; then
    return 0
  fi
  local fam="${FAMILY:-}"
  local eid="${EXPERIMENT_ID:-}"
  if [ "$fam" = "bind" ] || [ "$fam" = "arith" ] || [ "$eid" = "E19" ] || [[ "$eid" == E29* ]]; then
    echo "SKIP_E29: refuse FAMILY=${fam:-?} EXPERIMENT_ID=${eid:-?} (Hub bind is on Polonez; park after bits)"
    exit 0
  fi
}
