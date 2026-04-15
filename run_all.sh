#!/bin/bash

# =========================
# CONFIG
# =========================
RUNS=3

# 5 different ticker sets (each >= 5 tickers)
TICKER_SETS=(
  "XLK XLV XLF XLE XLI"
  "XLK XLV XLF XLE XLY"
  "XLK XLV XLF XLE XLP"
  "XLK XLV XLF XLE XLB"
  "XLK XLV XLF XLE XLC"
)

MODELS=(
  "ppo.py"
  "ppo_lstm.py"
  "ppo_attention.py"
)

# =========================
# RUN EXPERIMENTS
# =========================
for TICKERS in "${TICKER_SETS[@]}"
do
  echo "====================================="
  echo "Running ticker set: $TICKERS"
  echo "====================================="

  for MODEL in "${MODELS[@]}"
  do
    echo "---- Model: $MODEL ----"

    for ((i=1;i<=RUNS;i++))
    do
      echo "Run $i for $MODEL with tickers: $TICKERS"

      python $MODEL \
        --tickers $TICKERS \
        --total_timesteps 200000 \
        --learning_rate 3e-4 \
        --gamma 0.99 \
        --gae_lambda 0.95 \
        --clip_range 0.2 \
        --n_envs 20 \
        --n_steps 512 \
        --batch_size 1024 \
        --n_epochs 32

    done
  done
done
