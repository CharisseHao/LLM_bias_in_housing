#!/bin/bash

# Enable nullglob in case there are no matching files
shopt -s nullglob

# Default values for flags
tensor_parallel_size=1
max_model_len=1024
dry_run=false
max_num_seqs=1500
max_num_batched_tokens=12000
gpu_memory_utilization=0.95
replace=false

# Function to handle cleanup on CTRL-C
cleanup() {
    echo "Interrupted. Killing all child processes..."
    pkill -P $$
    exit 1
}

# Trap SIGINT (CTRL-C) to run the cleanup function
trap cleanup SIGINT

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tensor-parallel-size)
      tensor_parallel_size="$2"
      shift 2
      ;;
    --max-model-len)
      max_model_len="$2"
      shift 2
      ;;
    --max-num-seqs)
      max_num_seqs="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      max_num_batched_tokens="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      gpu_memory_utilization="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --replace)
      replace=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required but not installed. Please install 'jq' to proceed."
    exit 1
fi

# Directories for input, output, and logs
input_dir="input_data/batch_requests"
output_dir="input_data/batch_results"
log_dir="logs"

# Create output and log directories if they don't exist
mkdir -p "$output_dir" "$log_dir"

# Scan directory for JSONL files and iterate over them
for file in "$input_dir"/*.jsonl; do
    # Get the base filename (without path)
    filename=$(basename "$file")

    # Skip files that include 'gpt'
    if [[ $filename == *gpt* ]]; then
        continue
    fi

    if [[ $filename == *claude* ]]; then
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

    # Construct the output filename and log filename
    output_file="$output_dir/${filename%.jsonl}_output.jsonl"
    output_file_zip="$output_file.zip"
    log_file="$log_dir/${filename%.jsonl}.log"

    # Check if the output file exists and skip unless --replace is set
    if [[ -f "$output_file" || -f "$output_file_zip" ]] && [[ "$replace" != true ]]; then
        echo "Output file for '$filename' already exists. Skipping. Use --replace to overwrite."
        continue
    fi

    # Add tokenizer mode flag if 'mistral' is in the filename
    tokenizer_flag=""
    if [[ "${filename,,}" == *mistral* || "${filename,,}" == *ministral* ]]; then
        tokenizer_flag="--tokenizer_mode mistral --config_format mistral --load_format mistral"
    fi

    # Construct the command
    command="python3 -m vllm.entrypoints.openai.run_batch -i '$file' -o '$output_file' --model '$full_model_name' --pipeline-parallel-size 1 --tensor-parallel-size $tensor_parallel_size --max-model-len $max_model_len --max-num-batched-tokens $max_num_batched_tokens --max-num-seqs $max_num_seqs --gpu-memory-utilization $gpu_memory_utilization  $tokenizer_flag"

    # Check if dry run is enabled
    if [ "$dry_run" = true ]; then
        echo "Dry run: $command"
    else
        echo "Running command: $command"
        # Run the Python command, redirect output to both the log file and grep for live output
        eval "$command" 2>&1 | tee "$log_file" &
        wait $! # Wait for the current process to complete
        echo "Processing of '$filename' completed."
    fi
done

# Disable nullglob after script execution
shopt -u nullglob
