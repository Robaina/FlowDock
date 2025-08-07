# FlowDock Protein-Ligand Docking Guide

This guide explains how to use FlowDock for protein-ligand docking with PDB and SDF input files.

## Overview

**FlowDock** is a geometric flow matching model for generative protein-ligand docking and affinity prediction. It can generate 3D protein-ligand complex structures from protein sequences/structures and ligand SMILES/SDF files.

### Key Features
- **Direct PDB + SDF support**: Accepts protein PDB files and ligand SDF files directly
- **Template-based docking**: Uses existing protein structures as templates for better results
- **Multi-conformer generation**: Generates multiple docked poses with confidence scores
- **Binding affinity prediction**: Optionally predicts binding affinities
- **Docker support**: Can run in Docker container without local installation

## Quick Start (Non-Interactive) 🚀

### Option 1: Super Easy Python Script (Recommended)

The **simplest way** to run FlowDock - just provide 3 paths:

```bash
# Download checkpoints (one-time setup)
wget https://zenodo.org/records/15066450/files/flowdock_checkpoints.tar.gz
tar -xzf flowdock_checkpoints.tar.gz

# Basic docking - just provide the 3 paths!
python flowdock_easy.py your_protein.pdb your_ligand.sdf ./results

# Generate more poses
python flowdock_easy.py your_protein.pdb your_ligand.sdf ./results --samples 10

# Higher quality (slower)
python flowdock_easy.py your_protein.pdb your_ligand.sdf ./results --samples 10 --steps 80

# Force CPU usage (if no GPU)
python flowdock_easy.py your_protein.pdb your_ligand.sdf ./results --cpu
```

### Option 2: Bash Script Runner

Complete bash script for automation:

```bash
# Basic usage
./flowdock_docker_runner.sh --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results

# Custom parameters  
./flowdock_docker_runner.sh --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results \
    --n_samples 10 --num_steps 50 --sample_id my_docking

# CPU-only (no GPU)
./flowdock_docker_runner.sh --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results --cpu

# Verbose output
./flowdock_docker_runner.sh --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results --verbose
```

### Option 3: Advanced Python Script

Full-featured script with more options:

```bash
# Basic usage
python flowdock_simple_docking_example.py \
    --protein_pdb your_protein.pdb \
    --ligand_sdf your_ligand.sdf \
    --output_dir ./docking_results

# With Docker (non-interactive)
python flowdock_simple_docking_example.py \
    --protein_pdb your_protein.pdb \
    --ligand_sdf your_ligand.sdf \
    --output_dir ./docking_results \
    --docker

# Custom parameters
python flowdock_simple_docking_example.py \
    --protein_pdb your_protein.pdb \
    --ligand_sdf your_ligand.sdf \
    --output_dir ./docking_results \
    --n_samples 10 \
    --num_steps 50
```

### ✨ Key Features of Non-Interactive Scripts

- **No manual Docker interaction** - Scripts handle everything automatically
- **Automatic GPU detection** - Falls back to CPU if GPU unavailable  
- **File validation** - Checks inputs before running
- **Temporary workspace** - Clean file handling without cluttering directories
- **Progress feedback** - Clear status messages and error handling
- **Result copying** - Automatically copies outputs to your specified directory
- **Cleanup** - Removes temporary files automatically

## Direct FlowDock Command (Advanced Users)

```bash
python flowdock/sample.py \
    ckpt_path=checkpoints/esmfold_prior_paper_weights-EMA.ckpt \
    model.cfg.prior_type=esmfold \
    sampling_task=batched_structure_sampling \
    input_receptor=your_protein.pdb \
    input_ligand=your_ligand.sdf \
    input_template=your_protein.pdb \
    sample_id=my_docking_run \
    out_path=./results/ \
    n_samples=5 \
    chunk_size=5 \
    num_steps=40 \
    sampler=VDODE \
    use_template=true \
    separate_pdb=true \
    trainer=gpu
```

## Manual Docker Usage (Interactive Mode)

**Note**: The non-interactive scripts above are much easier to use! This section is for advanced users who want manual control.

### Prerequisites
1. **Docker** with GPU support (nvidia-docker)
2. **Input files**: protein PDB and ligand SDF files

### Pull Docker Image
```bash
docker pull cford38/flowdock:latest
```

### Download Checkpoints (Required)
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download pre-trained weights
wget https://zenodo.org/records/15066450/files/flowdock_checkpoints.tar.gz
tar -xzf flowdock_checkpoints.tar.gz
rm flowdock_checkpoints.tar.gz
```

### Run Docker Container (Interactive)
```bash
# Mount your data and checkpoints directories
docker run --gpus all \
    -v $(pwd)/checkpoints:/software/flowdock/checkpoints \
    -v $(pwd):/workspace \
    --rm -it cford38/flowdock:latest /bin/bash

# Inside container, run docking:
cd /software/flowdock
python flowdock/sample.py \
    ckpt_path=checkpoints/esmfold_prior_paper_weights-EMA.ckpt \
    model.cfg.prior_type=esmfold \
    sampling_task=batched_structure_sampling \
    input_receptor=/workspace/your_protein.pdb \
    input_ligand=/workspace/your_ligand.sdf \
    input_template=/workspace/your_protein.pdb \
    sample_id=docker_docking \
    out_path=/workspace/results/ \
    n_samples=5 \
    trainer=gpu
```

## Input File Requirements

### Protein Input (PDB file)
- **Format**: PDB format (.pdb extension)
- **Content**: Should contain protein structure with proper atom coordinates
- **Chains**: Multiple chains supported (separated by TER records in PDB)
- **Requirements**: Clean structure, no missing atoms in binding site region

### Ligand Input (SDF file)
- **Format**: SDF format (.sdf extension)  
- **Content**: 3D molecular structure with coordinates
- **Multiple molecules**: Single SDF can contain multiple ligand conformations
- **Note**: FlowDock will discard input coordinates and generate new ones

### Example File Structure
```
project/
├── protein.pdb          # Your protein structure
├── ligand.sdf           # Your ligand structure
├── checkpoints/         # FlowDock pre-trained weights
├── results/             # Output directory (will be created)
├── flowdock_easy.py                    # Super simple script
├── flowdock_docker_runner.sh          # Bash script
└── flowdock_simple_docking_example.py # Advanced Python script
```

## Configuration Options

### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `n_samples` | Number of docked poses to generate | 5 | 1-50+ |
| `num_steps` | Sampling steps (more = better quality) | 40 | 20-100 |
| `sampler` | Sampling algorithm | VDODE | ODE, VDODE |
| `sampler_eta` | Exploration vs exploitation trade-off | 1.0 | 0.5-2.0 |
| `use_template` | Use input PDB as template | true | true, false |
| `prior_type` | Type of prior model | esmfold | gaussian, harmonic, esmfold |
| `separate_pdb` | Save separate files for each pose | true | true, false |

### Advanced Options
```bash
# High-quality docking (slower)
python flowdock/sample.py \
    ... \
    n_samples=20 \
    num_steps=80 \
    sampler_eta=0.8

# Fast docking (lower quality)
python flowdock/sample.py \
    ... \
    n_samples=3 \
    num_steps=20 \
    sampler_eta=1.5
```

## Output Files

FlowDock generates several output files:

### Structure Files
- `{sample_id}_sample_{i}.pdb` - Individual docked complex structures
- `{sample_id}_ligand_{i}.sdf` - Individual ligand conformations  
- `{sample_id}_samples_concatenated.pdb` - All structures in one file

### Confidence Files
- `{sample_id}_confidence_scores.csv` - Confidence scores for each pose
- `{sample_id}_affinity_predictions.csv` - Predicted binding affinities (if enabled)

### Analysis Files
- `{sample_id}_metrics.json` - Structural and energetic metrics
- `trajectory_visualization/` - Sampling trajectory visualizations (if enabled)

## Batch Processing

For multiple protein-ligand pairs, create a CSV file:

### CSV Format
```csv
id,input_receptor,input_ligand,input_template
complex1,protein1.pdb,ligand1.sdf,protein1.pdb  
complex2,protein2.pdb,ligand2.sdf,protein2.pdb
complex3,protein3.pdb,ligand3.sdf,protein3.pdb
```

### Batch Command
```bash
python flowdock/sample.py \
    ckpt_path=checkpoints/esmfold_prior_paper_weights-EMA.ckpt \
    csv_path=batch_inputs.csv \
    out_path=./batch_results/ \
    n_samples=5 \
    trainer=gpu
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `chunk_size` to 1-3
   - Reduce `n_samples`
   - Use `trainer=cpu` (much slower)

2. **Checkpoint Not Found**
   ```bash
   wget https://zenodo.org/records/15066450/files/flowdock_checkpoints.tar.gz
   tar -xzf flowdock_checkpoints.tar.gz
   ```

3. **PDB Parsing Errors**
   - Clean PDB file (remove HETATM records except ligands)
   - Ensure proper chain identifiers
   - Check for missing atoms

4. **SDF Format Issues**
   - Ensure 3D coordinates are present
   - Check for valid molecular structure
   - Convert from other formats using RDKit/OpenBabel

### Performance Optimization

- **GPU Usage**: Always use `trainer=gpu` if available
- **Memory**: Adjust `chunk_size` based on GPU memory
- **Speed vs Quality**: Balance `num_steps` and `n_samples`
- **Template**: Using `use_template=true` generally improves results

## Alternative Input Methods

### Using Protein Sequences
If you don't have a PDB file, provide protein sequence directly:
```bash
python flowdock/sample.py \
    input_receptor="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALP" \
    input_ligand="CC(=O)OC1=CC=CC=C1C(=O)O" \  # Aspirin SMILES
    input_template=null \
    ...
```

### Using SMILES Strings
For ligands, SMILES strings can be used instead of SDF:
```bash
python flowdock/sample.py \
    input_receptor=protein.pdb \
    input_ligand="CC(=O)OC1=CC=CC=C1C(=O)O" \
    ...
```

## Script Comparison

Here's a comparison of the available scripts to help you choose:

| Script | Best For | Complexity | Features |
|--------|----------|------------|----------|
| `flowdock_easy.py` | **Beginners** | 🟢 Simple | Minimal arguments, emoji feedback, automatic cleanup |
| `flowdock_docker_runner.sh` | **Automation** | 🟡 Medium | Bash scripting, verbose mode, full parameter control |
| `flowdock_simple_docking_example.py` | **Advanced users** | 🔴 Complex | Most features, local/Docker modes, extensive error handling |

### Recommendations

- **First time using FlowDock?** → Use `flowdock_easy.py`
- **Need to automate many runs?** → Use `flowdock_docker_runner.sh` 
- **Want maximum control?** → Use `flowdock_simple_docking_example.py`
- **Just testing FlowDock?** → Use `flowdock_easy.py`

## References

- **Paper**: [FlowDock: Geometric Flow Matching for Generative Protein-Ligand Docking and Affinity Prediction](https://arxiv.org/abs/2412.10966)
- **Code**: [BioinfoMachineLearning/FlowDock](https://github.com/BioinfoMachineLearning/FlowDock)
- **Data**: [Zenodo Dataset](https://doi.org/10.5281/zenodo.15066450)