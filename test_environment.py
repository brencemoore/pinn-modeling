import sys

print("=" * 70)
print("PINN ENVIRONMENT TEST")
print("=" * 70)

print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")

# ---------------------------------------------------------
# Core scientific Python
# ---------------------------------------------------------

print("\n[1/8] Testing NumPy...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except Exception as e:
    print(f"  ✗ NumPy FAILED: {e}")

print("\n[2/8] Testing SciPy...")
try:
    import scipy
    print(f"  ✓ SciPy {scipy.__version__}")
except Exception as e:
    print(f"  ✗ SciPy FAILED: {e}")

print("\n[3/8] Testing Pandas...")
try:
    import pandas as pd
    print(f"  ✓ Pandas {pd.__version__}")
except Exception as e:
    print(f"  ✗ Pandas FAILED: {e}")

print("\n[4/8] Testing Matplotlib...")
try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"  ✗ Matplotlib FAILED: {e}")

# ---------------------------------------------------------
# Machine learning
# ---------------------------------------------------------

print("\n[5/8] Testing PyTorch...")
try:
    import torch

    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  CUDA compiled: {torch.version.cuda}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU count: {torch.cuda.device_count()}")

except Exception as e:
    print(f"  ✗ PyTorch FAILED: {e}")

# ---------------------------------------------------------
# PINN framework
# ---------------------------------------------------------

print("\n[6/8] Testing DeepXDE...")
try:
    import deepxde as dde
    print(f"  ✓ DeepXDE {dde.__version__}")
    print(f"  DeepXDE backend: {dde.backend.backend_name}")
except Exception as e:
    print(f"  ✗ DeepXDE FAILED: {e}")

# ---------------------------------------------------------
# Jupyter
# ---------------------------------------------------------

print("\n[7/8] Testing Jupyter...")
try:
    import jupyter
    print(f"  ✓ Jupyter available")
except Exception as e:
    print(f"  ✗ Jupyter FAILED: {e}")

print("\n[8/8] Testing IPython...")
try:
    import IPython
    print(f"  ✓ IPython {IPython.__version__}")
except Exception as e:
    print(f"  ✗ IPython FAILED: {e}")

# ---------------------------------------------------------
# Functional PyTorch test
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FUNCTIONAL TEST")
print("=" * 70)

try:
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2

    print("\nPyTorch calculation:")
    print(f"  Input:  {x}")
    print(f"  Output: {y}")

    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = x_gpu * 2

        print("\nGPU calculation:")
        print(f"  Device: {x_gpu.device}")
        print(f"  Output: {y_gpu}")

        print("\n✓ GPU test passed")
    else:
        print("\n⚠ CUDA is not available.")
        print("  PyTorch is working, but this machine will use CPU.")

except Exception as e:
    print(f"\n✗ Functional PyTorch test FAILED: {e}")

# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ENVIRONMENT TEST COMPLETE")
print("=" * 70)