#!/bin/bash

# Constants
REPEAT=12
CORES=(6 5 4 3 2 1)

for core in "${CORES[@]}"
do
  for i in $(seq 1 $REPEAT)
  do
    echo ">>> Evaluation $i with $core Core(s)"
    python3 lstm_eval.py --cpu_cores $core
  done
done
