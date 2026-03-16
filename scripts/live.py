import os
import sys
import subprocess

# 确保配置可访问
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

def run_step(name, script_path):
    print(f"\n" + "="*60)
    print(f"  STEP: {name}")
    print("="*60)
    
    # 使用正在运行此脚本的相同 python 可执行文件
    cmd = [sys.executable, script_path]
    
    try:
        # 我们使用 check_call 确保进程在下一步之前完成
        # 输出实时定向到终端
        subprocess.check_call(cmd)
        print(f"\n[SUCCESS] {name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {name} failed with exit code {e.returncode}. Pipeline aborted.")
        sys.exit(e.returncode)

def main():
    print("\n" + "#"*60)
    print("  LIUMON LIVE PRODUCTION PIPELINE v1.0")
    print("#"*60)
    
    # 定义脚本路径
    fetch_cn      = os.path.join(BASE_DIR, "liumon", "data", "data_fetch_cn.py")
    fetch_macro   = os.path.join(BASE_DIR, "liumon", "data", "data_fetch_macro.py")
    preprocess    = os.path.join(BASE_DIR, "features", "preprocess_cn.py")
    train_predict = os.path.join(BASE_DIR, "scripts", "train.py")
    
    # 执行流程
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
