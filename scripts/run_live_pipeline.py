import os
import sys
import subprocess

# Ensure config is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

def run_step(name, script_path):
    print(f"\n" + "="*60)
    print(f"  STEP: {name}")
    print("="*60)
    
    # Use the same python executable that is running this script
    cmd = [sys.executable, script_path]
    
    try:
        # We use check_call to ensure the process completes before next step
        # Output is directed to terminal in real-time
        subprocess.check_call(cmd)
        print(f"\n[SUCCESS] {name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {name} failed with exit code {e.returncode}. Pipeline aborted.")
        sys.exit(e.returncode)

def main():
    print("\n" + "#"*60)
    print("  LIUMON LIVE PRODUCTION PIPELINE v1.0")
    print("#"*60)
    
    # Define script paths
    fetch_cn      = os.path.join(BASE_DIR, "scripts", "data_fetch_cn.py")
    fetch_macro   = os.path.join(BASE_DIR, "scripts", "data_fetch_macro.py")
    preprocess    = os.path.join(BASE_DIR, "features", "preprocess_cn.py")
    train_predict = os.path.join(BASE_DIR, "scripts", "train_pipeline_cn.py")
    
    # Execution Flow
    run_step("Data Ingestion (A-Share)", fetch_cn)
    run_step("Market Regime Sensing", fetch_macro)
    run_step("Feature Engineering & Cleaning", preprocess)
    run_step("Model Training & Latest Prediction", train_predict)
    
    print("\n" + "#"*60)
    print("  PIPELINE FINISHED SUCCESSFULLY!")
    print("  Recommended Actions: Check 'models/' for weights and terminal for picks.")
    print("#"*60)

if __name__ == "__main__":
    main()
