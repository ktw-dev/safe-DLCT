#!/bin/bash

LOG_FILE="training_log.txt"

echo "Starting training in background with nohup..."
echo "Output will be saved to $LOG_FILE"

nohup bash run_training.sh > "$LOG_FILE" 2>&1 &

WRAPPER_PID=$!
sleep 2  # Wait for the python process to start

# Find the child process (actual python training)
# pgrep -P finds the child process of the wrapper
MAIN_PID=$(pgrep -P $WRAPPER_PID | head -n 1)

if [ -z "$MAIN_PID" ]; then
    # Fallback if child not found
    echo "Training started with Wrapper PID: $WRAPPER_PID"
    echo "Could not detect Python child process automatically."
else
    echo "Training started!"
    echo "Wrapper PID:     $WRAPPER_PID"
    echo "Main Python PID: $MAIN_PID  <-- (Use 'kill $MAIN_PID' to stop)"
fi

echo "You can check the progress using: tail -f $LOG_FILE"
