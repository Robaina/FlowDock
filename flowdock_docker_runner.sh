#!/bin/bash

# FlowDock Non-Interactive Docker Runner
# This script runs FlowDock docking using Docker without requiring interactive mode

set -e  # Exit on any error

# Default values
PROTEIN_PDB=""
LIGAND_SDF=""
OUTPUT_DIR=""
N_SAMPLES=5
NUM_STEPS=40
SAMPLE_ID=""
CHECKPOINT_PATH="checkpoints/esmfold_prior_paper_weights-EMA.ckpt"
DOCKER_IMAGE="cford38/flowdock:latest"
USE_GPU=true
VERBOSE=false

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

FlowDock Docker Runner - Non-interactive protein-ligand docking

Required Arguments:
  --protein_pdb PATH      Path to protein PDB file
  --ligand_sdf PATH       Path to ligand SDF file  
  --output_dir PATH       Output directory for results

Optional Arguments:
  --sample_id ID          Sample identifier (default: auto-generated)
  --n_samples N           Number of poses to generate (default: 5)
  --num_steps N           Number of sampling steps (default: 40)
  --checkpoint PATH       Path to checkpoint (default: checkpoints/esmfold_prior_paper_weights-EMA.ckpt)
  --docker_image IMAGE    Docker image to use (default: cford38/flowdock:latest)
  --cpu                   Use CPU instead of GPU
  --verbose               Enable verbose output
  --help                  Show this help message

Examples:
  # Basic usage
  $0 --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results

  # Custom parameters
  $0 --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results \\
     --n_samples 10 --num_steps 50 --sample_id my_docking

  # CPU-only (no GPU)
  $0 --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results --cpu

EOF
}

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to log verbose messages
log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo "[VERBOSE] $1"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --protein_pdb)
            PROTEIN_PDB="$2"
            shift 2
            ;;
        --ligand_sdf)
            LIGAND_SDF="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --sample_id)
            SAMPLE_ID="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --docker_image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --cpu)
            USE_GPU=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$PROTEIN_PDB" || -z "$LIGAND_SDF" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Validate input files exist
if [[ ! -f "$PROTEIN_PDB" ]]; then
    echo "Error: Protein PDB file not found: $PROTEIN_PDB"
    exit 1
fi

if [[ ! -f "$LIGAND_SDF" ]]; then
    echo "Error: Ligand SDF file not found: $LIGAND_SDF"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert to absolute paths
PROTEIN_PDB=$(realpath "$PROTEIN_PDB")
LIGAND_SDF=$(realpath "$LIGAND_SDF")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Generate sample ID if not provided
if [[ -z "$SAMPLE_ID" ]]; then
    PROTEIN_NAME=$(basename "$PROTEIN_PDB" .pdb)
    LIGAND_NAME=$(basename "$LIGAND_SDF" .sdf)
    SAMPLE_ID="${PROTEIN_NAME}_${LIGAND_NAME}"
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if nvidia-docker is available for GPU support
if [[ "$USE_GPU" = true ]]; then
    if ! docker info | grep -q nvidia; then
        log "Warning: nvidia-docker not detected, falling back to CPU"
        USE_GPU=false
    fi
fi

# Setup GPU flags
GPU_FLAGS=""
TRAINER="cpu"
if [[ "$USE_GPU" = true ]]; then
    GPU_FLAGS="--gpus all"
    TRAINER="gpu"
fi

# Check if checkpoints directory exists
if [[ ! -d "checkpoints" ]]; then
    echo "Error: checkpoints directory not found"
    echo "Please download FlowDock checkpoints:"
    echo "  wget https://zenodo.org/records/15066450/files/flowdock_checkpoints.tar.gz"
    echo "  tar -xzf flowdock_checkpoints.tar.gz"
    exit 1
fi

# Check if specific checkpoint exists
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Warning: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Using default checkpoint path within container"
fi

log "Starting FlowDock docking..."
log "Protein PDB: $PROTEIN_PDB"
log "Ligand SDF: $LIGAND_SDF"
log "Output directory: $OUTPUT_DIR"
log "Sample ID: $SAMPLE_ID"
log "N samples: $N_SAMPLES"
log "Num steps: $NUM_STEPS"
log "Using GPU: $USE_GPU"

# Build the FlowDock command
FLOWDOCK_CMD="cd /software/flowdock && python flowdock/sample.py \
ckpt_path=$CHECKPOINT_PATH \
model.cfg.prior_type=esmfold \
sampling_task=batched_structure_sampling \
input_receptor=/workspace/$(basename "$PROTEIN_PDB") \
input_ligand=/workspace/$(basename "$LIGAND_SDF") \
input_template=/workspace/$(basename "$PROTEIN_PDB") \
sample_id=$SAMPLE_ID \
out_path=/workspace/output/ \
n_samples=$N_SAMPLES \
chunk_size=$((N_SAMPLES < 5 ? N_SAMPLES : 5)) \
num_steps=$NUM_STEPS \
sampler=VDODE \
sampler_eta=1.0 \
start_time=1.0 \
use_template=true \
separate_pdb=true \
visualize_sample_trajectories=false \
auxiliary_estimation_only=false \
trainer=$TRAINER"

log_verbose "FlowDock command: $FLOWDOCK_CMD"

# Create temporary directory for Docker workspace
TEMP_DIR=$(mktemp -d)
log_verbose "Created temporary directory: $TEMP_DIR"

# Copy input files to temp directory
cp "$PROTEIN_PDB" "$TEMP_DIR/"
cp "$LIGAND_SDF" "$TEMP_DIR/"
mkdir -p "$TEMP_DIR/output"

log_verbose "Copied input files to temporary directory"

# Build Docker command
DOCKER_CMD="docker run --rm $GPU_FLAGS \
    -v $(pwd)/checkpoints:/software/flowdock/checkpoints \
    -v $TEMP_DIR:/workspace \
    $DOCKER_IMAGE \
    bash -c \"$FLOWDOCK_CMD\""

log_verbose "Docker command: $DOCKER_CMD"

# Run FlowDock
log "Running FlowDock in Docker container..."

if [[ "$VERBOSE" = true ]]; then
    # Run with output visible
    eval $DOCKER_CMD
else
    # Run with minimal output
    eval $DOCKER_CMD > /tmp/flowdock.log 2>&1
    if [[ $? -ne 0 ]]; then
        echo "Error: FlowDock failed. Log:"
        cat /tmp/flowdock.log
        exit 1
    fi
fi

# Copy results back to output directory
if [[ -d "$TEMP_DIR/output" ]]; then
    log "Copying results to output directory..."
    cp -r "$TEMP_DIR/output"/* "$OUTPUT_DIR/"
else
    echo "Error: No output directory created by FlowDock"
    exit 1
fi

# Clean up temporary directory
rm -rf "$TEMP_DIR"
log_verbose "Cleaned up temporary directory"

# List generated files
log "Docking completed successfully!"
log "Generated files in $OUTPUT_DIR:"
for file in "$OUTPUT_DIR"/*; do
    if [[ -f "$file" ]]; then
        echo "  - $(basename "$file")"
    fi
done

log "FlowDock docking finished!"