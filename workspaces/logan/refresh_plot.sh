#!/bin/bash
cd /workspace/tensor-mars/workspaces/logan
while true; do
    sleep 1800  # Every 30 minutes
    echo "$(date): Regenerating plot..."
    python3 plot_sweep_results.py 2>&1 | tail -3
done
