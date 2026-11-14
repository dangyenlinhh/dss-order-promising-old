#!/bin/bash
# Full pipeline used for the report (approx. 2 h on 2 CPU cores)
set -e
cd "$(dirname "$0")"
mkdir -p results logs
python3 experiments/exp1_testbed.py screening --reps 2 --tl 30 --jobs 2  > logs/exp1_screening.log 2>&1
python3 experiments/exp1_testbed.py grid      --reps 1 --tl 30 --jobs 2  > logs/exp1_grid.log 2>&1
python3 experiments/exp1_testbed.py grid --variant corrected --reps 1 --tl 30 --jobs 2 > logs/exp1_grid_corrected.log 2>&1
python3 experiments/exp1_testbed.py variant   --reps 1 --tl 30 --jobs 2  > logs/exp3a_testbed.log 2>&1
python3 experiments/exp1_testbed.py screening --variant corrected --reps 1 --tl 30 --jobs 2 > logs/exp1_screening_corrected.log 2>&1
python3 experiments/exp_sim.py exp2 --seeds 3 --days 20 --jobs 2 > logs/exp2.log 2>&1
python3 experiments/exp_sim.py 3a   --seeds 3 --days 20 --jobs 2 > logs/exp3a.log 2>&1
for e in 3b 3c 3d 3e 3f; do
  python3 experiments/exp_sim.py $e --seeds 2 --days 15 --jobs 2 > logs/exp$e.log 2>&1
done
python3 experiments/analyze.py > logs/analyze.log 2>&1
