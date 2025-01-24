#!/bin/bash

# Enable nullglob in case there are no matching files
shopt -s nullglob

# Function to handle cleanup on CTRL-C
cleanup() {
    echo "Interrupted. Killing all child processes..."
    pkill -P $$  # Kill all child processes spawned by this script
    exit 1
}

# Trap SIGINT (CTRL-C) to run the cleanup function
trap cleanup SIGINT

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required but not installed. Please install 'jq' to proceed."
    exit 1
fi

# Directories for input and logs
input_dir="input_data/batch_requests"
log_dir="dl_logs"

# Create log directory if it doesn't exist
mkdir -p "$log_dir"

# Scan directory for JSONL files and iterate over them
for file in "$input_dir"/*.jsonl; do
    # Get the base filename (without path)
    filename=$(basename "$file")

    # Skip files that start with 'gpt'
    if [[ $filename == gpt* ]]; then
        continue
    fi

    # Read the first line of the JSONL file to extract model name
    first_line=$(head -n 1 "$file")
    full_model_name=$(echo "$first_line" | jq -r '.body.model' 2>/dev/null)

    # Check if model name was extracted successfully
    if [[ -z "$full_model_name" || "$full_model_name" == "null" ]]; then
        echo "Warning: Failed to extract model name from '$file'. Skipping."
        continue
    fi

    # Construct the log filename
    log_file="$log_dir/${filename%.jsonl}.log"

    # Construct the huggingface-cli command
    command="huggingface-cli download '$full_model_name' --include \"model*.safetensors\""

    echo "Running command: $command"
    
    # Run the download command, redirect output to the log file
    eval "$command" 2>&1 | tee "$log_file" &
    
    # Wait for the current download to complete, allowing interruption with CTRL-C
    wait $!
    
    echo "Download for model '$full_model_name' from '$filename' completed."
done

# Disable nullglob after script execution
shopt -u nullglob
