#!/bin/bash
cd /workspace/tensor-mars/workspaces/logan

# Wait for weight_std to finish (poll its PID)
WS_PID=$(pgrep -f "train_ss_technique.*weight_std")
if [ -n "$WS_PID" ]; then
    echo "Waiting for weight_std (PID $WS_PID) to finish..."
    while kill -0 "$WS_PID" 2>/dev/null; do
        sleep 60
    done
    echo "weight_std finished."
fi

echo "Starting muP_init..."
python3 train_ss_technique.py --technique muP_init --batch-size 32 2>&1
echo "muP_init finished."

echo "Starting ortho_init..."
python3 train_ss_technique.py --technique ortho_init --batch-size 32 2>&1
echo "ortho_init finished."

echo "All remaining techniques done."
