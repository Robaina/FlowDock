#!/usr/bin/env python3
"""
FlowDock Easy Wrapper

Simple, non-interactive Python script for protein-ligand docking using FlowDock.
Just provide paths to your PDB and SDF files and get results!

Usage:
    python flowdock_easy.py protein.pdb ligand.sdf results_dir
    python flowdock_easy.py protein.pdb ligand.sdf results_dir --samples 10
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path


def run_flowdock_docking(protein_pdb, ligand_sdf, output_dir, n_samples=5, num_steps=40, use_cpu=False):
    """
    Run FlowDock docking in Docker container (non-interactive).
    
    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file
        output_dir: Output directory for results
        n_samples: Number of poses to generate
        num_steps: Number of sampling steps
        use_cpu: Force CPU usage instead of GPU
    
    Returns:
        List of generated output files
    """
    
    # Validate inputs
    if not os.path.exists(protein_pdb):
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb}")
    if not os.path.exists(ligand_sdf):
        raise FileNotFoundError(f"Ligand SDF not found: {ligand_sdf}")
    
    # Convert to absolute paths
    protein_pdb = os.path.abspath(protein_pdb)
    ligand_sdf = os.path.abspath(ligand_sdf)
    output_dir = os.path.abspath(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample ID
    protein_name = Path(protein_pdb).stem
    ligand_name = Path(ligand_sdf).stem
    sample_id = f"{protein_name}_{ligand_name}"
    
    print(f"🧬 FlowDock Easy Docking")
    print(f"📂 Protein: {os.path.basename(protein_pdb)}")
    print(f"💊 Ligand: {os.path.basename(ligand_sdf)}")
    print(f"🎯 Sample ID: {sample_id}")
    print(f"🔢 Poses: {n_samples}")
    print(f"⚡ Steps: {num_steps}")
    
    # Create temporary workspace
    temp_dir = tempfile.mkdtemp(prefix="flowdock_")
    temp_output = os.path.join(temp_dir, "output")
    os.makedirs(temp_output, exist_ok=True)
    
    try:
        # Copy input files to temp directory
        shutil.copy2(protein_pdb, temp_dir)
        shutil.copy2(ligand_sdf, temp_dir)
        
        protein_basename = os.path.basename(protein_pdb)
        ligand_basename = os.path.basename(ligand_sdf)
        
        # Determine GPU/CPU usage
        gpu_flags = []
        trainer = "cpu"
        
        if not use_cpu:
            try:
                # Check if nvidia-docker is available
                result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
                if "nvidia" in result.stdout.lower():
                    gpu_flags = ["--gpus", "all"]
                    trainer = "gpu"
                    print("🚀 Using GPU acceleration")
                else:
                    print("💻 Using CPU (GPU not available)")
            except:
                print("💻 Using CPU (Docker info check failed)")
        else:
            print("💻 Using CPU (forced)")
        
        # Check for checkpoints
        checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
        if not os.path.exists(checkpoints_dir):
            print("\n⚠️  WARNING: checkpoints directory not found!")
            print("Please download FlowDock checkpoints first:")
            print("wget https://zenodo.org/records/15066450/files/flowdock_checkpoints.tar.gz")
            print("tar -xzf flowdock_checkpoints.tar.gz")
            print("\nContinuing anyway (checkpoint might be in container)...")
        
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
discard_sdf_coords=true \
trainer={trainer}"""
        
        # Build Docker command
        docker_cmd = [
            "docker", "run", "--rm"
        ] + gpu_flags + [
            "-v", f"{checkpoints_dir}:/software/flowdock/checkpoints" if os.path.exists(checkpoints_dir) else "flowdock_checkpoints:/software/flowdock/checkpoints",
            "-v", f"{temp_dir}:/workspace",
            "cford38/flowdock:latest",
            "bash", "-c", flowdock_cmd
        ]
        
        print("\n🐳 Running FlowDock in Docker...")
        print("⏳ This may take several minutes...")
        
        # Run FlowDock
        result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        print("✅ FlowDock completed successfully!")
        
        # Copy results to output directory
        generated_files = []
        if os.path.exists(temp_output) and os.listdir(temp_output):
            for file in os.listdir(temp_output):
                src = os.path.join(temp_output, file)
                dst = os.path.join(output_dir, file)
                shutil.copy2(src, dst)
                generated_files.append(dst)
            
            print(f"\n📁 Results saved to: {output_dir}")
            print("📄 Generated files:")
            for file in generated_files:
                size = os.path.getsize(file)
                size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                print(f"  📄 {os.path.basename(file)} ({size_str})")
        else:
            print("⚠️  Warning: No output files generated")
            if result.stderr:
                print("Error output:", result.stderr)
        
        return generated_files
        
    except subprocess.TimeoutExpired:
        print("❌ FlowDock timed out (>1 hour). Try reducing n_samples or num_steps.")
        return []
    except subprocess.CalledProcessError as e:
        print(f"❌ FlowDock failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout[-1000:])  # Last 1000 chars
        print("STDERR:", e.stderr[-1000:])  # Last 1000 chars
        return []
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
        print("Visit: https://docs.docker.com/get-docker/")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="FlowDock Easy - Simple protein-ligand docking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic docking
    python flowdock_easy.py protein.pdb ligand.sdf ./results
    
    # Generate more poses
    python flowdock_easy.py protein.pdb ligand.sdf ./results --samples 10
    
    # Higher quality (slower)
    python flowdock_easy.py protein.pdb ligand.sdf ./results --samples 10 --steps 80
    
    # Force CPU usage
    python flowdock_easy.py protein.pdb ligand.sdf ./results --cpu
        """
    )
    
    parser.add_argument("protein_pdb", help="Path to protein PDB file")
    parser.add_argument("ligand_sdf", help="Path to ligand SDF file")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--samples", type=int, default=5, 
                       help="Number of poses to generate (default: 5)")
    parser.add_argument("--steps", type=int, default=40,
                       help="Number of sampling steps (default: 40, try 80 for better quality)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage instead of GPU")
    
    args = parser.parse_args()
    
    # Run docking
    try:
        files = run_flowdock_docking(
            args.protein_pdb, 
            args.ligand_sdf, 
            args.output_dir,
            args.samples, 
            args.steps,
            args.cpu
        )
        
        if files:
            print(f"\n🎉 Docking completed! Generated {len(files)} files.")
            print(f"📂 Check results in: {args.output_dir}")
            
            # Look for the best pose (if confidence scoring was done)
            pdb_files = [f for f in files if f.endswith('.pdb')]
            if pdb_files:
                print(f"\n💡 TIP: The generated PDB files contain docked poses.")
                print(f"   Look for files like '*_sample_0.pdb' (usually the best pose)")
        else:
            print("\n❌ No files were generated. Check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()