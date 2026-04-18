# CyScat Python Translation

This is a Python translation of the MATLAB CyScat electromagnetic scattering code, maintaining numerical precision with the original implementation.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Test

```bash
cd CyScat
python sample_test.py
```

This will:
- Generate random cylinder positions using Latin Hypercube sampling
- Compute the S-matrix (scattering matrix) for the structure
- Display the results and a visualization of the slab geometry

## Key Files

- `sample_test.py` - Main test script demonstrating the workflow
- `interface.py` - Main interface for S-matrix computation
- `Scattering_Code/` - Core electromagnetic scattering algorithms
  - `smatrix.py` - S-matrix generation
  - `transall.py` - Translation matrix calculations
  - `smatrix_parameters.py` - Spectral parameter setup
  - `vall.py` - Plane wave to cylinder harmonics conversion
  - `ky.py` - Wave vector component calculations

## Translation Notes

All numerical precision issues from MATLAB to Python have been resolved:
- Complex number handling for evanescent modes
- Spectral sum overflow protection
- Array indexing differences (0-based vs 1-based)
- Matrix multiplication semantics
- Scalar extraction from numpy arrays

The code now produces results matching MATLAB within numerical precision.
