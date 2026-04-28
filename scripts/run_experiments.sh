#!/bin/bash

# Configuration
TRAIN_STEPS=${1:-50}
TEST_STEPS=${2:-10}
RUNS=${3:-3}

POLICIES=("topomab" "linucb" "static_fastest" "static_most_accurate" "epsilon_greedy" "random")

echo "====================================================="
echo "Running TopoMAB System Experiments"
echo "Train Steps: $TRAIN_STEPS"
echo "Test Steps: $TEST_STEPS"
echo "Number of Runs per Policy: $RUNS"
echo "====================================================="

mkdir -p data
mkdir -p results

for policy in "${POLICIES[@]}"
do
    echo ""
    echo "====================================================="
    echo "Evaluating Policy: $policy"
    echo "====================================================="
    
    # Overriding the baseline in environment for evaluation
    BASELINE=$policy uv run python evaluate.py \
        --policy "$policy" \
        --train "$TRAIN_STEPS" \
        --test "$TEST_STEPS" \
        --runs "$RUNS" | tee "results/log_${policy}.txt"
        
done

echo ""
echo "All experiments completed. Logs are saved in the results/ directory."
