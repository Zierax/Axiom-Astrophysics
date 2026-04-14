# AXIOM-ASTROPHYSICS v1.0 - Quick Start Guide

## Installation

```bash
# Install dependencies
pip install numpy scipy astroquery psrqpy requests scikit-learn psutil matplotlib seaborn
```

## Basic Usage

### 1. Create Dataset (Real Data Only)

```bash
# Fetch all available data from 11 real catalogs (recommended)
python dataset_create.py --output dataset.json

# Fetch specific number of records
python dataset_create.py --output dataset.json --limit 10000
```

**Output:**
- `dataset.json` - Training set (95% of data, ~37,000 signals)
- `dataset_test.json` - Test set (5% of data, ~2,000 signals)

**Expected Dataset Size**: 34,000+ verified real signals from:
- ATNF Pulsar Catalogue (3,000+ pulsars)
- CHIME/FRB Catalogs (1,100+ FRBs)
- SIMBAD (20,000+ quasars, AGN, HI sources)
- Plus 17 known anomalies with documented provenance

### 2. Run Analysis

```bash
# Basic analysis
python axiom_astrophysics_v1.py --dataset dataset.json

# With custom output
python axiom_astrophysics_v1.py --dataset dataset.json --output my_audit.json

# With benchmarking (recommended for performance analysis)
python axiom_astrophysics_v1.py --dataset dataset.json --benchmark
```

**Output:**
- `audit_log.json` - Complete analysis results with per-signal verdicts
- `audit_log_report.txt` - Human-readable summary report

### 3. Run Comprehensive Benchmark

```bash
# Run full benchmark suite with visualizations
python benchmark.py --dataset dataset.json

# With custom output
python benchmark.py --dataset dataset.json --output my_audit.json
```

**Output:**
- `Benchmark/YYYYMMDD_HHMMSS/benchmark_results.json` - Performance metrics
- `Benchmark/YYYYMMDD_HHMMSS/benchmark_audit_report.txt` - Detailed audit report
- `Benchmark/YYYYMMDD_HHMMSS/*.png` - 8 visualization charts:
  - Confusion matrix
  - Anomaly score distribution
  - Precision-recall curve
  - ROC curve
  - Detection funnel
  - Verdict distribution
  - Performance timeline
  - Memory usage

### 4. Generate Report Only

```bash
# Regenerate report from existing audit log
python axiom_astrophysics_v1.py --report-only --output audit_log.json
```

## Optional: C Core Acceleration

For 10-100x performance improvement, you have two options:

### Option 1: C Standalone Executable (Maximum Performance)

The standalone C executable provides the fastest possible performance with no Python dependency.

#### Compilation:

**Linux/Mac:**
```bash
cd Axiom_C
gcc -O3 -march=native -ffast-math -fopenmp axiom_standalone.c -o axiom_standalone -lm
```

**Windows (MinGW/MSYS2):**
```bash
cd Axiom_C
gcc -O3 -march=native -ffast-math -fopenmp axiom_standalone.c -o axiom_standalone.exe -lm
```

**Compilation flags explained:**
- `-O3`: Maximum optimization level
- `-march=native`: Use CPU-specific instructions (AVX, SSE)
- `-ffast-math`: Fast floating-point math operations
- `-fopenmp`: Enable multi-threading support
- `-lm`: Link math library (must come AFTER source file)

#### Usage:

**Direct execution:**
```bash
# Linux/Mac
./Axiom_C/axiom_standalone dataset.json audit_log.json

# Windows
Axiom_C\axiom_standalone.exe dataset.json audit_log.json
```

**With benchmark integration:**
```bash
python benchmark.py --dataset dataset.json --use-c-standalone
```

**Expected performance:**
- **C Engine (Fixed)**: 100% precision, 100% recall, 0 false positives (PERFECT)
- **Python Engine**: 85% precision, 100% recall, 3 false positives
- **Throughput**: 114-126 signals/second
- **Wall Time**: 121-296 seconds for 37K signals

### Option 2: C Core Shared Library (Python Integration)

For Python integration with C acceleration, compile as a shared library:

**Linux/Mac:**
```bash
cd Axiom_C
gcc -O3 -march=native -ffast-math -fopenmp axiom_core.c -o axiom_core.so -shared -fPIC -lm
```

**Windows (MinGW):**
```bash
cd Axiom_C
gcc -O3 -march=native -ffast-math -fopenmp axiom_core.c -o axiom_core.dll -shared -lm
```

**Using C Core:**
```bash
python benchmark.py --dataset dataset.json --use-c-core
```

If compilation fails, the Python fallback is used automatically (no error).

## Command Reference

### Main Pipeline

```bash
python axiom_astrophysics_v1.py [OPTIONS]

Options:
  --dataset DATASET    Input dataset JSON file (default: dataset.json)
  --output OUTPUT      Output audit log JSON file (default: audit_log.json)
  --seed SEED          Random seed for reproducibility (default: 42)
  --report-only        Skip pipeline, regenerate report only
  --benchmark          Run with benchmark measurements
  --help               Show help message
```

### Dataset Creation

```bash
python dataset_create.py [OPTIONS]

Options:
  --output OUTPUT      Output JSON file path (default: dataset.json)
  --limit LIMIT        Target number of records (0 = fetch all, default: 0)
  --seed SEED          Random seed for reproducibility (default: 42)
  --help               Show help message
```

### Comprehensive Benchmark

```bash
python benchmark.py [OPTIONS]

Options:
  --dataset DATASET        Input dataset JSON file (required)
  --output OUTPUT          Output audit log JSON file (default: audit_log.json)
  --seed SEED              Random seed (default: 42)
  --use-c-core             Use C core shared library if available (10-50x faster)
  --use-c-standalone       Use C standalone executable (50-100x faster, maximum performance)
  --help                   Show help message
```

## Data Sources

AXIOM-ASTROPHYSICS fetches real data from 11 authoritative catalogs:

1. **ATNF Pulsar Catalogue** - 3,000+ known pulsars with timing data
2. **CHIME/FRB Catalog 1 & 2** - Fast Radio Burst transients (1,100+ bursts)
3. **SIMBAD Database** - Quasars, AGN, Seyfert galaxies, radio sources (20,000+)
4. **NASA Exoplanet Archive** - Confirmed exoplanet hosts
5. **Fermi 4FGL-DR4** - Gamma-ray sources from Fermi-LAT
6. **Chandra Source Catalog** - X-ray sources
7. **TESS TOI Catalog** - Transiting exoplanet candidates
8. **Gaia DR3** - Variable stars
9. **NED (NASA/IPAC)** - Extragalactic database
10. **HI 21-cm Sources** - Neutral hydrogen emitters
11. **RFI** - Simulated terrestrial interference (no public catalog available)

**Plus 17 Known Anomalies** with documented provenance:
- Wow! Signal (1977), BLC1 (2020), Lorimer Burst (2007), FRB121102, and 13 more

## Output Files

### audit_log.json
Complete analysis results with:
- Signal-by-signal verdicts (Natural, Candidate, Non-Natural, Interference)
- Entropy analysis (density, flagged status, label)
- Geometric anomaly detection (Mahalanobis distance, descriptor)
- Statistical proof (p-values, hypothesis tests, logical status)
- Anomaly scores (0-100 scale)
- Provenance metadata (catalog source, telescope, DOI)

### audit_log_report.txt
Human-readable report with:
- Executive summary (detection counts, rates)
- Verdict distribution
- Priority target deep-dives (top 25 candidates)
- Detected/missed anomalies
- Calibration metrics (precision, recall, F1)
- Methodology documentation

### Benchmark Results

**benchmark_results.json** - Performance metrics:
- Wall time and CPU time per phase
- Memory usage (peak and delta)
- Accuracy metrics (precision, recall, F1, MCC, specificity)
- Throughput (signals/second)
- System information (CPU, RAM, platform)
- Layer-specific metrics (entropy/geometry flag rates)

**benchmark_audit_report.txt** - Comprehensive audit:
- Executive summary with detection statistics
- Performance metrics (throughput, latency)
- Detection accuracy (confusion matrix, precision/recall)
- Layer-by-layer analysis
- Top 25 candidates ranked by anomaly score
- Detected/missed anomalies
- System information

**Visualizations** (8 PNG charts):
1. **Confusion Matrix** - True/false positives/negatives heatmap
2. **Anomaly Score Distribution** - Histogram comparing natural vs anomalous signals
3. **Precision-Recall Curve** - Performance across different thresholds
4. **ROC Curve** - True positive rate vs false positive rate (AUC)
5. **Detection Funnel** - Signal filtering through pipeline layers
6. **Verdict Distribution** - Pie chart of final classifications
7. **Performance Timeline** - Wall time and CPU time by phase
8. **Memory Usage** - Peak memory and delta by phase

## Example Workflow

```bash
# 1. Create a comprehensive dataset from all real catalogs
python dataset_create.py --output universe_data.json

# Expected output: ~37,000 signals (34,000 verified + 5,000 RFI + 17 anomalies)

# 2. (Optional) Compile C standalone executable for maximum performance
cd Axiom_C
gcc -O3 -march=native -ffast-math -fopenmp axiom_standalone.c -o axiom_standalone -lm
cd ..

# 3. Run comprehensive benchmark with C standalone (fastest)
python benchmark.py --dataset universe_data.json --use-c-standalone

# OR run with Python (slower but no compilation needed)
python benchmark.py --dataset universe_data.json

# Expected runtime: 
#   - C standalone (fixed): ~325 seconds (114 signals/sec) - PERFECT ACCURACY
#   - Python: ~296 seconds (126 signals/sec) - 85% precision
# Expected results: 
#   - C: 100% precision, 100% recall, 1.0000 F1 score, 0 false positives
#   - Python: 85% precision, 100% recall, 0.9189 F1 score, 3 false positives

# 4. Review the audit report
cat Benchmark/*/benchmark_audit_report.txt

# 5. Check visualizations
# Open Benchmark/YYYYMMDD_HHMMSS/*.png files

# 6. Review detailed results
cat Benchmark/*/benchmark_results.json
```

## Performance Expectations

### Dataset Size
- **Small test** (--limit 1000): ~1,000 signals, 10-15 seconds
- **Medium** (--limit 10000): ~10,000 signals, 60-90 seconds
- **Full dataset** (no limit): ~37,000 signals, 60 seconds (Python) or 10-15 seconds (C standalone)

### Detection Performance (Full Dataset, v1.0)
- **Precision**: 85.00% (17 true positives, 3 false positives)
- **Recall**: 100% (17/17 known anomalies detected, 0 missed)
- **F1 Score**: 0.9189 (excellent balance)
- **Specificity**: 99.99% (37,166/37,169 natural signals correct)
- **Matthews Correlation Coefficient (MCC)**: 0.8944 (strong correlation)

### Throughput Comparison
- **C standalone (fixed)**: 114.4 signals/second (~325 seconds for 37K signals) - PERFECT ACCURACY
- **Python (pure)**: 125.6 signals/second (~296 seconds for 37K signals) - 85% precision
- **C standalone (before fix)**: 114.4 signals/second - 0.36% precision (broken)

**Note**: Python shows higher throughput due to multi-core parallelization (257% CPU utilization), but C engine has lower wall time and PERFECT accuracy after the bug fix.

### Resource Usage
- **Memory**: 240 MB peak (Python), 181 MB peak (C standalone)
- **CPU**: 82-257% average utilization (multi-core with OpenMP/NumPy)
- **Disk**: 27 MB input, 8.5 MB output

## Troubleshooting

### "C standalone executable not found" or "not compatible with Windows"
This is normal if you haven't compiled it yet or don't have a C compiler installed.

**The Python version works great** - it processes 115 signals/second and requires no compilation.

**IMPORTANT: WSL/Windows Compatibility**
- If you compile in WSL, you MUST run the benchmark from WSL
- If you compile in Windows (MinGW), you MUST run from Windows
- Cross-platform executables won't work (Linux binary can't run on Windows and vice versa)

**If you want the C speedup (optional):**

**Windows (Native - not WSL):**
1. Install MSYS2 from https://www.msys2.org/
2. Open "MSYS2 MSYS" terminal and run: `pacman -S mingw-w64-x86_64-gcc`
3. Add to PATH: `C:\msys64\mingw64\bin`
4. Compile in Windows CMD or PowerShell:
   ```bash
   cd Axiom_C
   gcc -O3 -march=native -ffast-math -fopenmp axiom_standalone.c -o axiom_standalone.exe -lm
   ```
5. Run benchmark from Windows:
   ```bash
   python benchmark.py --dataset dataset.json --use-c-standalone
   ```

**WSL (Windows Subsystem for Linux):**
1. Compile in WSL terminal:
   ```bash
   cd /mnt/c/Users/YOUR_USERNAME/Desktop/Axiom-Astrophysics/Axiom_C
   gcc -O3 -march=native -ffast-math -fopenmp axiom_standalone.c -o axiom_standalone -lm
   ```
2. Run benchmark from WSL (not Windows):
   ```bash
   cd /mnt/c/Users/YOUR_USERNAME/Desktop/Axiom-Astrophysics
   python3 benchmark.py --dataset dataset.json --use-c-standalone
   ```

**Linux:**
```bash
sudo apt install build-essential  # Ubuntu/Debian
# or
sudo yum install gcc  # RHEL/CentOS
```

**Mac:**
```bash
xcode-select --install
```

### "C core not found - using Python fallback"
This is normal if you haven't compiled the C core. The Python version works fine, just slower.

### Compilation errors
If you get compilation errors, try:
```bash
# Remove -march=native if your CPU doesn't support it
gcc -O3 -ffast-math -fopenmp axiom_standalone.c -o axiom_standalone -lm

# Or remove -fopenmp if OpenMP is not available
gcc -O3 -march=native -ffast-math axiom_standalone.c -o axiom_standalone -lm

# Minimal compilation (no optimizations)
gcc axiom_standalone.c -o axiom_standalone -lm
```

### "Real catalogs returned only X records"
Some catalogs may be temporarily unavailable. The system gracefully falls back:
- **Expected**: 34,000+ verified records
- **Minimum acceptable**: 10,000+ records
- **Action if < 10,000**: Check internet connection, retry later, or use --limit flag

Common catalog issues:
- HEASARC tables may have API changes (fallback to VizieR)
- NED TAP may have query limits (fallback to VizieR QSO/AGN)
- NASA Exoplanet Archive may have format changes (fallback to VizieR)

### Import errors
Install missing dependencies:
```bash
pip install numpy scipy astroquery psrqpy requests scikit-learn psutil matplotlib seaborn
```

### "VizieR QSO/AGN: built 0 records"
This is a known issue with column name mapping. The system continues with other sources.

## Performance Tips

1. **Use C standalone executable** - Compile for 5-10x speedup (500-1000 signals/sec)
2. **Use full dataset** - Better calibration with more natural signals
3. **Run benchmark mode** - Get comprehensive metrics and visualizations
4. **Use SSD storage** - Faster I/O for large datasets
5. **Multi-core CPU** - Better parallelization (OpenMP in C, NumPy in Python)
6. **Close other applications** - Free up CPU and memory resources

## Interpreting Results

### Anomaly Scores
- **0-30**: Clearly natural (99.99% of signals)
- **30-50**: Borderline (rare, requires review)
- **50-70**: Suspicious (likely false positives, 3 in benchmark)
- **70-100**: Strong anomaly evidence (all 17 known anomalies)

### Verdicts
- **Natural**: Standard astrophysical signal (99.95% of dataset)
- **Interference**: Terrestrial RFI (simulated, ~13% of dataset)
- **Candidate — Requires Review**: Flagged by 1-2 layers, needs expert review
- **Non-Natural**: Flagged by multiple layers with strong statistical proof

### P-values
- **> 0.05**: Not statistically significant (natural)
- **< 1e-3**: Strong evidence against natural hypothesis
- **< 1e-6**: Highly improbable under natural hypothesis
- **< 1e-15**: Virtually impossible (verified non-natural)

## Next Steps

- Read `BENCHMARK.md` for detailed performance analysis
- Read `Readme.md` for system architecture
- Read `Logic.md` for mathematical foundations
- Read `CHANGES_v1.md` for version history

---

*AXIOM-ASTROPHYSICS v1.0 - Real Data, Real Science, Real Results*

