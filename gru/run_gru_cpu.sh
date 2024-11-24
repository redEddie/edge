#!/bin/bash

# Constant
REPEAT=12

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with CORE 4"
  python3 gru_eval.py --cpu_cores 4
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with CORE 3"
  python3 gru_eval.py --cpu_cores 3
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with CORE 2"
  python3 gru_eval.py --cpu_cores 2
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with CORE 1"
  python3 gru_eval.py --cpu_cores 1
done
