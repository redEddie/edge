#!/bin/bash

# Constant
REPEAT=12

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with 4 Cores"
  python3 lstm_eval.py --cpu_cores 4
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with 3 Cores"
  python3 lstm_eval.py --cpu_cores 3
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with 2 Cores"
  python3 lstm_eval.py --cpu_cores 2
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with 1 Cores"
  python3 lstm_eval.py --cpu_cores 1
done
