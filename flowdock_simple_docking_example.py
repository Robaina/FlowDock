#!/usr/bin/env python3
"""
FlowDock Simple Docking Example

This script demonstrates how to use FlowDock to perform protein-ligand docking
using a protein PDB file and a ligand SDF file as inputs.

Usage:
    python flowdock_simple_docking_example.py --protein_pdb path/to/protein.pdb --ligand_sdf path/to/ligand.sdf --output_dir ./results

Requirements:
    - FlowDock installed and configured
    - Pre-trained FlowDock checkpoint available
    - Docker container with FlowDock or local installation
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def validate_inputs(protein_pdb: str, ligand_sdf: str, output_dir: str):
    """Validate input files and output directory."""
    if not os.path.exists(protein_pdb):
        raise FileNotFoundError(f"Protein PDB file not found: {protein_pdb}")
    
    if not os.path.exists(ligand_sdf):
        raise FileNotFoundError(f"Ligand SDF file not found: {ligand_sdf}")
    
    if not protein_pdb.lower().endswith('.pdb'):
        raise ValueError("Protein file must be a PDB file (.pdb)")
    
    if not ligand_sdf.lower().endswith('.sdf'):
        raise ValueError("Ligand file must be an SDF file (.sdf)")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"✓ Protein PDB: {protein_pdb}")
    print(f"✓ Ligand SDF: {ligand_sdf}")
    print(f"✓ Output directory: {output_dir}")


def run_flowdock_docking(protein_pdb: str, ligand_sdf: str, output_dir: str, 
                         n_samples: int = 5, num_steps: int = 40, 
                         checkpoint_path: str = None):
    """
    Run FlowDock protein-ligand docking.
    
    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file  
        output_dir: Directory to save results
        n_samples: Number of structures to sample (default: 5)
        num_steps: Number of sampling steps (default: 40)
        checkpoint_path: Path to FlowDock checkpoint (default: pre-trained weights)
    """
    
    # Default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/esmfold_prior_paper_weights-EMA.ckpt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Please ensure FlowDock checkpoints are downloaded:")
        print("wget https://zenodo.org/records/15066450/files/flowdock_checkpoints.tar.gz")
        print("tar -xzf flowdock_checkpoints.tar.gz")
    
    # Convert paths to absolute paths
    protein_pdb = os.path.abspath(protein_pdb)
    ligand_sdf = os.path.abspath(ligand_sdf)
    output_dir = os.path.abspath(output_dir)
    
    # Generate a sample ID based on input filenames
    protein_name = Path(protein_pdb).stem
    ligand_name = Path(ligand_sdf).stem
    sample_id = f"{protein_name}_{ligand_name}"
    
    # FlowDock command
    cmd = [
        "python", "flowdock/sample.py",
        f"ckpt_path={checkpoint_path}",
        "model.cfg.prior_type=esmfold",
        "sampling_task=batched_structure_sampling",
        f"input_receptor={protein_pdb}",
        f"input_ligand={ligand_sdf}",
        f"input_template={protein_pdb}",
        f"sample_id={sample_id}",
        f"out_path={output_dir}",
        f"n_samples={n_samples}",
        f"chunk_size={min(n_samples, 5)}",
        f"num_steps={num_steps}",
        "sampler=VDODE",
        "sampler_eta=1.0",
        "start_time=1.0",
        "use_template=true",
        "separate_pdb=true",
        "visualize_sample_trajectories=false",
        "auxiliary_estimation_only=false",
        "trainer=gpu"
    ]
    
    print("\n" + "="*60)
    print("RUNNING FLOWDOCK DOCKING")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    # Run FlowDock
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ FlowDock completed successfully!")
        print("\nOutput:")
        print(result.stdout)
        
        # List generated files
        print(f"\nGenerated files in {output_dir}:")
        for file in os.listdir(output_dir):
            if file.endswith(('.pdb', '.sdf')):
                print(f"  - {file}")
                
    except subprocess.CalledProcessError as e:
        print(f"✗ FlowDock failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("✗ FlowDock not found. Please ensure you are in the FlowDock directory and it's properly installed.")
        print("Alternative: Use Docker container:")
        print("docker run --gpus all -v ./checkpoints:/software/flowdock/checkpoints -v $(pwd):/workspace --rm -it cford38/flowdock:latest")
        sys.exit(1)


def docker_run_flowdock(protein_pdb: str, ligand_sdf: str, output_dir: str, 
                        n_samples: int = 5, num_steps: int = 40):
    """
    Run FlowDock using Docker container (non-interactive).
    
    This is useful if you don't have FlowDock installed locally.
    """
    
    # Convert paths to absolute paths
    protein_pdb = os.path.abspath(protein_pdb)
    ligand_sdf = os.path.abspath(ligand_sdf)
    output_dir = os.path.abspath(output_dir)
    
    # Generate sample ID
    protein_name = Path(protein_pdb).stem
    ligand_name = Path(ligand_sdf).stem
    sample_id = f"{protein_name}_{ligand_name}"
    
    # Create temporary directory for Docker workspace
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    temp_output = os.path.join(temp_dir, "output")
    os.makedirs(temp_output, exist_ok=True)
    
    try:
        # Copy input files to temp directory
        shutil.copy2(protein_pdb, temp_dir)
        shutil.copy2(ligand_sdf, temp_dir)
        
        protein_basename = os.path.basename(protein_pdb)
        ligand_basename = os.path.basename(ligand_sdf)
        
        # Check GPU availability
        gpu_flags = ["--gpus", "all"]
        trainer = "gpu"
        try:
            # Check if nvidia-docker is available
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if "nvidia" not in result.stdout.lower():
                print("Warning: GPU support not detected, using CPU")
                gpu_flags = []
                trainer = "cpu"
        except:
            gpu_flags = []
            trainer = "cpu"
        
        # Build FlowDock command
        flowdock_cmd = f"""cd /software/flowdock && python flowdock/sample.py \
ckpt_path=checkpoints/esmfold_prior_paper_weights-EMA.ckpt \
model.cfg.prior_type=esmfold \
sampling_task=batched_structure_sampling \
input_receptor=/workspace/{protein_basename} \
input_ligand=/workspace/{ligand_basename} \
input_template=/workspace/{protein_basename} \
sample_id={sample_id} \
out_path=/workspace/output/ \
n_samples={n_samples} \
chunk_size={min(n_samples, 5)} \
num_steps={num_steps} \
sampler=VDODE \
sampler_eta=1.0 \
start_time=1.0 \
use_template=true \
separate_pdb=true \
visualize_sample_trajectories=false \
auxiliary_estimation_only=false \
trainer={trainer}"""
        
        # Build Docker command (non-interactive)
        docker_cmd = [
            "docker", "run", "--rm"
        ] + gpu_flags + [
            "-v", f"{os.getcwd()}/checkpoints:/software/flowdock/checkpoints",
            "-v", f"{temp_dir}:/workspace",
            "cford38/flowdock:latest",
            "bash", "-c", flowdock_cmd
        ]
        
        print("\n" + "="*60)
        print("RUNNING FLOWDOCK DOCKING WITH DOCKER (NON-INTERACTIVE)")
        print("="*60)
        print(f"Protein: {protein_pdb}")
        print(f"Ligand: {ligand_sdf}")
        print(f"Output: {output_dir}")
        print(f"Sample ID: {sample_id}")
        print(f"N samples: {n_samples}")
        print(f"Trainer: {trainer}")
        print("="*60)
        
        # Run Docker command
        result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True)
        print("✓ FlowDock Docker run completed successfully!")
        
        # Copy results back to output directory
        if os.path.exists(temp_output):
            for file in os.listdir(temp_output):
                shutil.copy2(os.path.join(temp_output, file), output_dir)
            
            # List generated files
            print(f"\nGenerated files in {output_dir}:")
            for file in os.listdir(output_dir):
                if file.endswith(('.pdb', '.sdf', '.csv', '.json')):
                    print(f"  - {file}")
        else:
            print("Warning: No output directory created by FlowDock")
            
    except subprocess.CalledProcessError as e:
        print(f"✗ FlowDock Docker run failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("✗ Docker not found. Please install Docker first.")
        sys.exit(1)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Simple FlowDock protein-ligand docking example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python flowdock_simple_docking_example.py --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results
    
    # With custom parameters
    python flowdock_simple_docking_example.py --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results --n_samples 10 --num_steps 50
    
    # Using Docker
    python flowdock_simple_docking_example.py --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results --docker
    
    # Custom checkpoint
    python flowdock_simple_docking_example.py --protein_pdb protein.pdb --ligand_sdf ligand.sdf --output_dir ./results --checkpoint_path ./my_checkpoint.ckpt
        """
    )
    
    parser.add_argument("--protein_pdb", required=True,
                       help="Path to protein PDB file")
    parser.add_argument("--ligand_sdf", required=True,
                       help="Path to ligand SDF file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--n_samples", type=int, default=5,
                       help="Number of structures to sample (default: 5)")
    parser.add_argument("--num_steps", type=int, default=40,
                       help="Number of sampling steps (default: 40)")
    parser.add_argument("--checkpoint_path", default=None,
                       help="Path to FlowDock checkpoint (default: pre-trained weights)")
    parser.add_argument("--docker", action="store_true",
                       help="Use Docker container instead of local installation")
    
    args = parser.parse_args()
    
    print("FlowDock Simple Docking Example")
    print("=" * 40)
    
    # Validate inputs
    validate_inputs(args.protein_pdb, args.ligand_sdf, args.output_dir)
    
    # Run docking
    if args.docker:
        print("\nUsing Docker container...")
        docker_run_flowdock(args.protein_pdb, args.ligand_sdf, args.output_dir,
                           args.n_samples, args.num_steps)
    else:
        print("\nUsing local FlowDock installation...")
        run_flowdock_docking(args.protein_pdb, args.ligand_sdf, args.output_dir,
                            args.n_samples, args.num_steps, args.checkpoint_path)
    
    print(f"\n✓ Docking completed! Check results in: {args.output_dir}")


if __name__ == "__main__":
    main()