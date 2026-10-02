#!/usr/bin/env bash
# scripts/smoke-test.sh [BASE_URL] — quick correctness check of a running server (the kit's shared smoke test:
# arithmetic, a fact, code, thinking mode and a tool call). SMOKE_NO_THINK=1 sends the three direct-answer checks with
# thinking off: with it on, Qwen3.6-35B-A3B reasons past their 256-400 token budgets. The thinking check is unchanged.
export SMOKE_NO_THINK=1
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/kit/bench/smoke-test.sh" "$@"
