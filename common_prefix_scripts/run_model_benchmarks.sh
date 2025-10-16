#!/bin/bash

set -x

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-name) MODEL_NAME="$2"; shift ;;
        --log-dir) LOG_DIR="$2"; shift ;;
        --tp) TP_SIZE="$2"; shift ;;
        --backend) USE_PYTORCH_BACKEND="$2"; shift ;;
        --isl) ISL="$2"; shift ;;
        --osl) OSL="$2"; shift ;;
        --common-prefix-len) COMMON_PREFIX_LEN="$2"; shift ;;
        --num-prompts) NUM_PROMPTS="$2"; shift ;;
        --quantization) QUANTIZATION="$2"; shift ;;
        --model-path) MODEL_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MODEL_NAME" ] || [ -z "$LOG_DIR" ] || [ -z "$TP_SIZE" ] || [ -z "$ISL" ] || [ -z "$OSL" ] || [ -z "$COMMON_PREFIX_LEN" ] || [ -z "$NUM_PROMPTS" ] || [ -z "$QUANTIZATION" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 --model-name <model_name> --log-dir <log_dir> --tp <tp_size> --isl <isl> --osl <osl> --common-prefix-len <cpl> --num-prompts <num_prompts> --quantization <quantization>"
    exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p /scratch/

# Function to run benchmarks
run_benchmark() {
  local model_name=$1
  local isl=$2
  local osl=$3
  local num_requests=$4
  local tp_size=$5
  local quantization=$6 # Add quantization type as argument
  local common_prefix_len=$7
  local batch_size=$8

  echo "Running benchmark for $model_name with ISL=$isl, OSL=$osl, TP=$tp_size, Quantization=$quantization"

  dataset_file="/scratch/token-norm-dist_${model_name##*/}_${isl}_${osl}_tp${tp_size}_cpl${common_prefix_len}.json"
  output_file="$LOG_DIR/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}_cpl${common_prefix_len}_${quantization}.txt"  # Include quantization in output file

  python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0  --common-prefix-len=$common_prefix_len > $dataset_file
  # python benchmarks/cpp/prepare_dataset.py --output=$dataset_file --tokenizer=$model_name token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0

  pp_size=1

  if [ -n "$USE_PYTORCH_BACKEND" ]; then
    echo "Running benchmark for $model_name with BACKEND_TYPE=pytorch, ISL=$isl, OSL=$osl, TP=$tp_size, Quantization=$quantization"
    trtllm-bench --model $model_name --model_path $MODEL_PATH throughput --dataset $dataset_file --tp $tp_size --pp $pp_size --backend pytorch --kv_cache_free_gpu_mem_fraction 0.95 --enable_chunked_context > $output_file
  else
    if [ -n "$batch_size" ]; then
      echo "Building engine with specified max_batch_size: $batch_size"
      trtllm-bench --model $model_name build --tp_size $tp_size --pp_size $pp_size --quantization $quantization --max_seq_len $(($isl+$osl)) --max_num_tokens $isl --max_batch_size $batch_size
    else
      echo "Building engine with dataset-based tuning"
      trtllm-bench --model $model_name build --tp_size $tp_size --pp_size $pp_size --quantization $quantization --dataset $dataset_file 
    fi

    engine_dir="/tmp/${model_name}/tp_${tp_size}_pp_${pp_size}"

    # Save throughput output to a file
    trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir --kv_cache_free_gpu_mem_fraction 0.95 > $output_file
    # /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir $engine_dir --type IFB --api executor --dataset $dataset_file --eos_id -1 --log_iteration_data --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.95 --output_csv $output_file --request_rate -1.0 --enable_chunked_context --warm_up 0

    rm -rf $engine_dir
  fi
  rm -f $dataset_file
}

echo "==========================================================="
echo "Starting benchmarks for $MODEL_NAME"
echo "==========================================================="
run_benchmark "$MODEL_NAME" "$ISL" "$OSL" "$NUM_PROMPTS" "$TP_SIZE" "$QUANTIZATION" "$COMMON_PREFIX_LEN" ""