#!/bin/bash

# Constant
REPEAT=12

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with Mem_Growth"
  python3 gru_eval.py --gpu --memory_growth
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with mem 2048"
  python3 gru_eval.py --gpu --gpu_mem_limit 2048
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with mem 1024"
  python3 gru_eval.py --gpu --gpu_mem_limit 1024
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with mem 512"
  python3 gru_eval.py --gpu --gpu_mem_limit 512
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with mem 256"
  python3 gru_eval.py --gpu --gpu_mem_limit 256
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with mem 128"
  python3 gru_eval.py --gpu --gpu_mem_limit 128
done

for i in $(seq 1 $REPEAT)
do
  echo ">>> Evaluation $i with mem 64"
  python3 gru_eval.py --gpu --gpu_mem_limit 64
done

