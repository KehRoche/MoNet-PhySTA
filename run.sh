#!/usr/bin/env bash
set -euo pipefail

python experiments/monet/run_best_experiments.py --datasets PEMS-BAY SD KnowAir "$@"
