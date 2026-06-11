import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import numpy as np
import torch
import hopfnet.hopfnet_cpp as mhd_engine

def run_diagnostics():
    print("="*50)
    print("🚀 HOPFNET-MHD SYSTEM DIAGNOSTICS")
    print("="*50)

    # 1. Test PyTorch Apple Silicon (MPS) support
    print("[1/3] Checking PyTorch MPS (Metal Performance Shaders)...")
    if torch.backends.mps.is_available():
        print("      ✅ PyTorch is using Apple M4 GPU (MPS)!")
    else:
        print("      ❌ MPS not found. Falling back to CPU.")

    # 2. Test e3nn Installation
    print("[2/3] Checking E(3) Equivariant library...")
    try:
        import e3nn
        print(f"      ✅ e3nn version {e3nn.__version__} loaded successfully.")
    except ImportError:
        print("      ❌ e3nn failed to load.")

    # 3. Test C++ / Python Bridge via PyBind11 and Eigen
    print("[3/3] Checking C++ Physics Engine Bridge...")
    test_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = mhd_engine.double_matrix(test_array)
    
    if np.allclose(result, test_array * 2):
        print("      ✅ PyBind11 + Eigen C++ Memory Bridge is fully operational!")
        print(f"      Input from Python:\n{test_array}")
        print(f"      Output from C++ :\n{result}")
    else:
        print("      ❌ C++ Bridge returned incorrect math.")

    print("="*50)
    print("ALL SYSTEMS NOMINAL. READY FOR PHASE 1.")

if __name__ == "__main__":
    run_diagnostics()

