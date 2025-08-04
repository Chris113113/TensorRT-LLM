# TensorRT-LLM GKE Benchmark Suite

This project provides a containerized benchmarking suite for running NVIDIA's TensorRT-LLM on a Google Kubernetes Engine (GKE) cluster. It is designed to be highly configurable, allowing users to easily test various models against a matrix of sequence lengths and quantization settings.

The suite is optimized for performance, using local SSDs for model caching and `gsutil` for fast, parallelized data transfers.

## Prerequisites

Before you begin, ensure you have the following tools installed and configured:
-   `gcloud` CLI
-   `kubectl`
-   `docker` (with the `buildx` component)
-   Access to a GKE cluster with NVIDIA GB200 GPUs (or other ARM64-based GPUs).

## Quickstart

### 1. Configure the Benchmark

All benchmark parameters are controlled by `a4x-helm-charts/values.yaml`. Open this file and configure your desired run:

-   **`modelName`**: The Hugging Face identifier for the model you want to test. A list of available, pre-downloaded models in the GCS bucket can be found in `configs/model_list.txt`.

**Note on Custom Quantized Models:**
The `model_list.txt` file also contains paths to custom quantized models (e.g., `gs://...`). These models are not sourced from Hugging Face and must be downloaded directly from their GCS path for local testing. These models can be loaded by TensorRT-LLM when using the `pytorch` backend. The benchmark script expects these to be placed in a specific directory structure that mimics the GCS path.

-   **`tp`**: The tensor parallelism size.
-   **`quantizations`**: A list of quantizations to test (e.g., `["FP8", "NVFP4"]`).
-   **`benchmarks`**: A list of sequence length combinations to run. Each item should define:
    -   `isl`: Input Sequence Length
    -   `osl`: Output Sequence Length
    -   `commonPrefixLen`: The length of a shared prefix for prompts.
    -   `numPrompts`: The number of prompts to generate for the test.

### 2. Set Up Your Hugging Face Token

The benchmark runner requires a Hugging Face token to download models. Create a Kubernetes secret in your cluster with your token:

**IMPORTANT:** Replace `<your-hf-token>` with your actual Hugging Face token.
```bash
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<your-hf-token>
```

### 3. Build the Docker Image

This project targets the `linux/arm64` architecture. If you are building on an x86_64 machine, you must use `docker buildx` to cross-compile.

**One-Time Setup for `buildx`:**
```bash
docker buildx create --name multi-platform-builder --use
```

**Build the Image:**
```bash
docker buildx build --platform linux/arm64 -t YOUR_REGISTRY/trtllm-bench-arm64:latest -f Dockerfile.a4x . --load
```
Remember to replace `YOUR_REGISTRY` with your container registry path and push the image. Update the `image.repository` in `values.yaml` accordingly.

### 4. Deploy the Benchmark

Once your `values.yaml` is configured and the image is pushed, deploy the benchmark pod using Helm:

```bash
helm install trtllm-benchmark ./a4x-helm-charts
```

The pod will start, copy the model to a local SSD, and begin executing the benchmark matrix you defined.

### 5. Monitor and Retrieve Results

You can monitor the pod's progress with `kubectl logs`:

```bash
kubectl logs -f <pod-name>
```

All benchmark outputs are saved to a timestamped directory in your GCS bucket, under the path specified in the pod's logs (e.g., `/mnt/disk/models/trtllm-bench-outputs/20250730_180000-Qwen--Qwen3-4B`).

### 6. Parse the Results

After the benchmark run is complete, use the `extract_trt_metrics.py` script to parse the raw log files into a structured CSV format.

```bash
python3 extract_trt_metrics.py /path/to/your/log/directory
```
This will generate a `tensorrt_llm_perf_metrics.csv` file in the same directory, containing key performance indicators from the run.

---

## Running Locally with Docker

For development and testing, you can run the benchmark suite on a local machine with a compatible NVIDIA GPU, without needing a Kubernetes cluster.

### 1. Prepare Local Directories

You will need two local directories:
-   A directory for your output logs (this will be mounted into the container).
-   A directory to serve as the fast local cache for the models.

```bash
mkdir -p ~/trtllm-bench-files/outputs
mkdir -p ~/trtllm-bench-files/local-cache
```

### 2. Pre-populate the Local Cache

This step copies the model from GCS to your local cache directory. It's designed to be run once and skipped if the model is already present.

```bash
# Set your configuration variables
export MODEL_NAME="Qwen/Qwen3-32B"
export GCS_BUCKET="your-gcs-bucket-name" # <-- IMPORTANT: Set your bucket name
export SANITIZED_MODEL_NAME=$(echo ${MODEL_NAME} | sed 's/\//--/g')
export LOCAL_CACHE_DIR="~/trtllm-bench-files/local-cache/huggingface_model_cache"
export MODEL_CACHE_DEST="${LOCAL_CACHE_DIR}/models--${SANITIZED_MODEL_NAME}"

# Check if the model directory already exists
if [ -d "$MODEL_CACHE_DEST" ]; then
  echo "Model already exists in local cache. Skipping download."
else
  echo "Model not found in local cache. Copying from GCS..."
  # Ensure the parent directory exists
  mkdir -p "$LOCAL_CACHE_DIR"
  # Copy the model using gsutil for parallel performance
  time gsutil -m cp -r "gs://${GCS_BUCKET}/huggingface_model_cache/models--${SANITIZED_MODEL_NAME}" "$LOCAL_CACHE_DIR"
fi
```

### 3. Run the Benchmark

Now, use the `docker run` command to start the container. The script inside the container will now assume the model is already in the cache and will proceed directly to the benchmark.

**Note:** This command is for `linux/arm64` hosts. If you are running on an `x86_64` host, you will need an ARM64-compatible container runtime and GPU drivers.

```bash
# Set your configuration variables
export MODEL_NAME="Qwen/Qwen3-32B"
export TP_SIZE=1
export HF_TOKEN="<your-hf-token>"
export IMAGE_NAME="your-registry/trtllm-bench-arm64:latest"
export SANITIZED_MODEL_NAME=$(echo ${MODEL_NAME} | sed 's/\//--/g')

docker run --gpus all --rm -it \
  -v ~/trtllm-bench-files/outputs:/mnt/disk/models \
  -v ~/trtllm-bench-files/local-cache:/cache \
  -e MODEL_NAME="${MODEL_NAME}" \
  -e TP="${TP_SIZE}" \
  -e BACKEND="pytorch" \
  -e HF_TOKEN="${HF_TOKEN}" \
  ${IMAGE_NAME} \
  /bin/bash -c '
    set -ex

    # --- 1. Resolve model path in cache ---
    CACHE_DIR="/cache/huggingface_model_cache"
    MODEL_DIR_IN_CACHE="${CACHE_DIR}/models--${SANITIZED_MODEL_NAME}"

    if [ -d "$MODEL_DIR_IN_CACHE" ]; then
        SNAPSHOT_ID=$(cat "${MODEL_DIR_IN_CACHE}/refs/main")
        MODEL_PATH_FOR_BENCHMARK="${MODEL_DIR_IN_CACHE}/snapshots/${SNAPSHOT_ID}"
        export HF_HUB_OFFLINE="1"
    else
        echo "ERROR: Model not found in the local cache directory mounted at /cache."
        exit 1
    fi

    export HF_HOME="$CACHE_DIR"
    export HF_HUB_CACHE="$CACHE_DIR"
    
    # --- 2. Run benchmarks ---
    LOG_DIR="/mnt/disk/models/trtllm-bench-outputs/$(date +"%Y%m%d_%H%M%S")-${SANITIZED_MODEL_NAME}"
    mkdir -p "$LOG_DIR"
    echo "Logging to $LOG_DIR"

    # Example of running a single benchmark.
    ./run_model_benchmarks.sh \
      --model-name "${MODEL_PATH_FOR_BENCHMARK}" \
      --log-dir "${LOG_DIR}" \
      --tp "${TP}" \
      --backend "${BACKEND}" \
      --isl 8192 \
      --osl 2048 \
      --common-prefix-len 7900 \
      --num-prompts 50 \
      --quantization "FP8"
  '
```

### 4. Retrieve Results

The benchmark logs and results will be saved to `~/trtllm-bench-files/outputs/trtllm-bench-outputs/`. You can then parse them using the `extract_trt_metrics.py` script as described in the main guide.

```

