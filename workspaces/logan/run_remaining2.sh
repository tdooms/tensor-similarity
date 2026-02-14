#!/bin/bash
cd /workspace/tensor-mars/workspaces/logan

# Wait for the weight_std PYTHON process specifically
echo "Polling for weight_std to finish..."
while pgrep -f "python3 train_ss_technique.py --technique weight_std" > /dev/null 2>&1; do
    sleep 60
    echo "$(date): weight_std still running..."
done
echo "$(date): weight_std finished!"

# Small delay to free GPU memory
sleep 10

echo "$(date): Starting muP_init..."
python3 train_ss_technique.py --technique muP_init --batch-size 32 2>&1
echo "$(date): muP_init finished."

echo "$(date): Starting ortho_init..."
python3 train_ss_technique.py --technique ortho_init --batch-size 32 2>&1
echo "$(date): ortho_init finished."

# Regenerate the plot
echo "$(date): Regenerating final plot..."
python3 plot_sweep_results.py 2>&1

echo "$(date): All done!"
