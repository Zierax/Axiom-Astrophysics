#!/bin/bash
# Recompile AXIOM-ASTROPHYSICS C Standalone with Bug Fixes

echo "================================================================================"
echo "AXIOM-ASTROPHYSICS C Standalone Recompilation"
echo "================================================================================"
echo ""

# Check for gcc
if ! command -v gcc &> /dev/null; then
    echo "ERROR: gcc not found. Please install gcc first."
    echo "  Ubuntu/Debian: sudo apt-get install gcc"
    echo "  Fedora/RHEL:   sudo dnf install gcc"
    echo "  macOS:         xcode-select --install"
    exit 1
fi

echo "[COMPILING] axiom_standalone.c with optimizations..."
echo "  Flags: -O3 -march=native -ffast-math -fopenmp"
echo ""

# Compile
gcc -O3 -march=native -ffast-math -fopenmp Axiom_C/axiom_standalone.c -o Axiom_C/axiom_standalone -lm

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Compilation complete!"
    echo ""
    echo "Executable: Axiom_C/axiom_standalone"
    ls -lh Axiom_C/axiom_standalone
    echo ""
    echo "================================================================================"
    echo "NEXT STEPS:"
    echo "================================================================================"
    echo "1. Run benchmark to verify fix:"
    echo "   python3 benchmark.py --dataset dataset.json --use-c-standalone"
    echo ""
    echo "2. Compare with Python baseline:"
    echo "   python3 benchmark.py --dataset dataset.json"
    echo ""
    echo "Expected: C and Python should now have ~identical accuracy (Precision ~85%)"
    echo "================================================================================"
else
    echo "[ERROR] Compilation failed!"
    exit 1
fi
