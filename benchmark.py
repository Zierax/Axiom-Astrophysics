"""
AXIOM-ASTROPHYSICS Comprehensive Benchmark Suite v1.0
Measures latency, accuracy, resource usage, and detection performance
Features:
- Multi-layer performance profiling
- C core integration via ctypes
- Extensive visualization suite (10+ charts)
- Audit report generation
- Per-signal timing analysis
"""
import time, psutil, os, json, sys, ctypes
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict

# Scientific visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

class AxiomBenchmark:
    def __init__(self, output_dir="Benchmark", use_c_core=False, use_c_standalone=False):
        """Initialize benchmark with timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self.use_c_core = use_c_core
        self.use_c_standalone = use_c_standalone
        self.c_core_lib = None
        self.c_standalone_path = None
        
        # Try to load C core if requested
        if use_c_core:
            self._load_c_core()
        
        # Try to find C standalone executable if requested
        if use_c_standalone:
            self._find_c_standalone()
        
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_info": self._get_system_info(),
            "c_core_enabled": self.c_core_lib is not None,
            "c_standalone_enabled": self.c_standalone_path is not None,
            "metrics": {},
            "layer_metrics": {},
            "per_signal_stats": {}
        }
        self.process = psutil.Process()
        self.layer_timings = defaultdict(list)
    
    def _load_c_core(self):
        """Load C core library via ctypes"""
        try:
            # Try different possible locations
            possible_paths = [
                "Axiom_C/axiom_core.so",
                "Axiom_C/axiom_core.dll",
                "Axiom_C/axiom_core.dylib",
                "./axiom_core.so",
                "./axiom_core.dll",
                "./axiom_core.dylib"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.c_core_lib = ctypes.CDLL(path)
                    print(f"[C CORE] Loaded from: {path}")
                    return
            
            print("[C CORE] Not found - using Python fallback")
        except Exception as e:
            print(f"[C CORE] Failed to load: {e}")
            self.c_core_lib = None
    
    def _find_c_standalone(self):
        """Find C standalone executable"""
        try:
            import platform
            is_wsl = 'microsoft' in platform.uname().release.lower()
            
            possible_paths = []
            
            # Prioritize based on platform
            if sys.platform == 'win32' and not is_wsl:
                # Native Windows
                possible_paths = [
                    "Axiom_C/axiom_standalone.exe",
                    "./axiom_standalone.exe"
                ]
            else:
                # Linux/Mac/WSL
                possible_paths = [
                    "Axiom_C/axiom_standalone",
                    "./axiom_standalone",
                    "Axiom_C/axiom_standalone.exe",  # Fallback
                    "./axiom_standalone.exe"
                ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    # Verify it's executable
                    if os.access(path, os.X_OK) or sys.platform == 'win32':
                        self.c_standalone_path = path
                        print(f"[C STANDALONE] Found executable: {path}")
                        return
            
            print("[C STANDALONE] Not found - compile with: gcc -O3 -march=native -ffast-math -fopenmp Axiom_C/axiom_standalone.c -o Axiom_C/axiom_standalone -lm")
        except Exception as e:
            print(f"[C STANDALONE] Error: {e}")
            self.c_standalone_path = None
    
    def run_c_standalone(self, dataset_path: str, output_path: str) -> Dict[str, Any]:
        """Run C standalone executable and measure performance"""
        if not self.c_standalone_path:
            raise RuntimeError("C standalone executable not found")
        
        import subprocess
        import platform
        import threading
        import re
        
        # Check for WSL/Windows compatibility issue
        is_wsl = 'microsoft' in platform.uname().release.lower()
        if sys.platform == 'win32' and not is_wsl and 'axiom_standalone.exe' in self.c_standalone_path:
            # Check if this is a Linux binary with .exe extension
            try:
                test_result = subprocess.run(
                    [self.c_standalone_path],
                    capture_output=True,
                    timeout=1
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                if isinstance(e, OSError) and e.winerror == 216:
                    raise RuntimeError(
                        "The C executable was compiled in WSL/Linux but you're running from Windows. "
                        "Either: (1) Run benchmark from WSL, or (2) Compile in Windows with MinGW. "
                        "See QUICKSTART_v1.md for instructions."
                    )
        
        print(f"\n[C STANDALONE] Executing: {self.c_standalone_path}")
        print(f"[C STANDALONE] Input: {dataset_path}")
        print(f"[C STANDALONE] Output: {output_path}")
        
        start_time = time.perf_counter()
        start_mem = self.process.memory_info().rss / (1024**2)
        
        try:
            # Start process with stdout/stderr pipes
            process = subprocess.Popen(
                [self.c_standalone_path, dataset_path, output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            stdout_lines = []
            stderr_lines = []
            
            # Read stdout in real-time and show progress
            def read_stdout():
                for line in process.stdout:
                    stdout_lines.append(line)
                    line_stripped = line.strip()
                    if "Progress:" in line_stripped:
                        # Show progress on same line
                        print(f"\r[C STANDALONE] {line_stripped}", end='', flush=True)
                    elif line_stripped and not line_stripped.startswith('='):
                        # Show other messages on new lines
                        if stdout_lines and "Progress:" in stdout_lines[-2] if len(stdout_lines) > 1 else False:
                            print()  # New line after progress
                        print(f"[C STANDALONE] {line_stripped}")
            
            def read_stderr():
                for line in process.stderr:
                    stderr_lines.append(line)
                    if line.strip():
                        print(f"[C STANDALONE ERROR] {line.strip()}")
            
            # Start reader threads
            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process with timeout
            try:
                returncode = process.wait(timeout=600)  # 10 minute timeout
            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError("C standalone execution timed out (>10 minutes)")
            
            # Wait for threads to finish reading
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            wall_time = time.perf_counter() - start_time
            peak_mem = self.process.memory_info().rss / (1024**2)
            
            if returncode != 0:
                stderr_text = ''.join(stderr_lines)
                print(f"\n[C STANDALONE] Error: {stderr_text}")
                raise RuntimeError(f"C standalone failed with code {returncode}")
            
            # Parse output for throughput
            stdout_text = ''.join(stdout_lines)
            processed_signals = 0
            c_elapsed = wall_time
            
            # Look for completion message
            for line in stdout_lines:
                match = re.search(r'Processed (\d+) signals in ([\d.]+) seconds', line)
                if match:
                    processed_signals = int(match.group(1))
                    c_elapsed = float(match.group(2))
                    break
            
            print(f"\n[C STANDALONE] Completed in {wall_time:.2f}s")
            if processed_signals > 0:
                print(f"[C STANDALONE] Throughput: {processed_signals/c_elapsed:.1f} signals/sec")
            
            return {
                "wall_time_sec": round(wall_time, 4),
                "c_reported_time_sec": round(c_elapsed, 4),
                "peak_memory_mb": round(peak_mem, 2),
                "memory_delta_mb": round(peak_mem - start_mem, 2),
                "signals_processed": processed_signals,
                "throughput_signals_per_sec": round(processed_signals / c_elapsed if c_elapsed > 0 else 0, 2),
                "stdout": stdout_text,
                "stderr": ''.join(stderr_lines)
            }
            
        except OSError as e:
            if hasattr(e, 'winerror') and e.winerror == 216:
                raise RuntimeError(
                    "The C executable was compiled in WSL/Linux but you're running from Windows. "
                    "Either: (1) Run benchmark from WSL, or (2) Compile in Windows with MinGW. "
                    "See QUICKSTART_v1.md for instructions."
                )
            raise RuntimeError(f"C standalone execution failed: {e}")
        except subprocess.TimeoutExpired:
            progress_stop.set()
            raise RuntimeError("C standalone execution timed out (>10 minutes)")
        except OSError as e:
            progress_stop.set()
            if hasattr(e, 'winerror') and e.winerror == 216:
                raise RuntimeError(
                    "The C executable was compiled in WSL/Linux but you're running from Windows. "
                    "Either: (1) Run benchmark from WSL, or (2) Compile in Windows with MinGW. "
                    "See QUICKSTART_v1.md for instructions."
                )
            raise RuntimeError(f"C standalone execution failed: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        cpu_freq = psutil.cpu_freq()
        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": cpu_freq.current if cpu_freq else 0,
            "cpu_freq_max_mhz": cpu_freq.max if cpu_freq else 0,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "python_version": sys.version,
            "platform": sys.platform,
            "numpy_version": np.__version__,
            "matplotlib_available": MATPLOTLIB_AVAILABLE,
            "seaborn_available": SEABORN_AVAILABLE
        }
    
    def record_layer_timing(self, layer_name: str, duration: float):
        """Record timing for individual layer"""
        self.layer_timings[layer_name].append(duration)
    
    def record_per_signal_metrics(self, signal_id: str, metrics: Dict[str, Any]):
        """Record per-signal performance metrics"""
        self.results["per_signal_stats"][signal_id] = metrics
    
    def start_phase(self, phase_name: str):
        """Start timing a benchmark phase"""
        self.current_phase = phase_name
        self.phase_start_time = time.perf_counter()
        self.phase_start_cpu = self.process.cpu_times()
        self.phase_start_mem = self.process.memory_info().rss / (1024**2)  # MB
        print(f"\n[BENCHMARK] Starting: {phase_name}")
    
    def end_phase(self, additional_metrics: Dict[str, Any] = None):
        """End timing and record metrics"""
        wall_time = time.perf_counter() - self.phase_start_time
        cpu_times = self.process.cpu_times()
        cpu_time = (cpu_times.user - self.phase_start_cpu.user + 
                   cpu_times.system - self.phase_start_cpu.system)
        peak_mem = self.process.memory_info().rss / (1024**2)  # MB
        mem_delta = peak_mem - self.phase_start_mem
        
        metrics = {
            "wall_time_sec": round(wall_time, 4),
            "cpu_time_sec": round(cpu_time, 4),
            "cpu_utilization_pct": round((cpu_time / wall_time * 100) if wall_time > 0 else 0, 2),
            "peak_memory_mb": round(peak_mem, 2),
            "memory_delta_mb": round(mem_delta, 2)
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.results["metrics"][self.current_phase] = metrics
        
        print(f"[BENCHMARK] {self.current_phase}:")
        print(f"  Wall time: {wall_time:.4f}s")
        print(f"  CPU time: {cpu_time:.4f}s ({metrics['cpu_utilization_pct']}%)")
        print(f"  Memory: {peak_mem:.2f} MB (Δ {mem_delta:+.2f} MB)")
        if additional_metrics:
            for k, v in additional_metrics.items():
                print(f"  {k}: {v}")
    
    def record_accuracy_metrics(self, audit_records: List[Dict]):
        """Compute comprehensive detection accuracy metrics - TEST SET ONLY"""
        # CRITICAL FIX: Only evaluate on test set (is_training_data=False)
        test_records = [r for r in audit_records if not r.get("is_training_data", False)]
        
        if not test_records:
            print("[WARNING] No test records found - evaluating on all data (legacy mode)")
            test_records = audit_records
        else:
            print(f"[ACCURACY] Evaluating on TEST SET ONLY: {len(test_records)} records (excluding {len(audit_records) - len(test_records)} training records)")
        
        tp = fp = tn = fn = 0
        verdicts = []
        anomaly_scores = []
        detected_anomalies = []
        missed_anomalies = []
        false_positives = []
        
        # Layer-specific metrics
        entropy_flags = 0
        geometry_flags = 0
        both_layers = 0
        
        for record in test_records:
            verdict = record.get("verdict", "")
            signal_id = record.get("signal_id", "")
            score = record.get("anomaly_score", 0)
            entropy_label = record.get("entropy_label", "")
            geo_anomaly = record.get("geometric_anomaly", "")
            
            verdicts.append(verdict)
            anomaly_scores.append(score)
            
            # Count layer flags
            if "Low" in entropy_label or "Non-Natural" in entropy_label:
                entropy_flags += 1
            if geo_anomaly != "None Detected":
                geometry_flags += 1
            if entropy_flags > 0 and geometry_flags > 0:
                both_layers += 1
            
            # Identify known anomalies by signal_id prefix "ANOMALY_"
            is_known_anomaly = signal_id.startswith("ANOMALY_")
            is_known_natural = not is_known_anomaly
            
            # Count both "Non-Natural" and "Candidate" as detections
            is_detected = verdict in ("Non-Natural", "Candidate — Requires Review")
            
            if is_detected:
                if is_known_anomaly:
                    tp += 1
                    detected_anomalies.append(signal_id)
                elif is_known_natural:
                    fp += 1
                    false_positives.append(signal_id)
            elif verdict in ("Natural", "Interference"):
                if is_known_natural:
                    tn += 1
                elif is_known_anomaly:
                    fn += 1
                    missed_anomalies.append(signal_id)
        
        total = len(audit_records)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        # Fix accuracy calculation
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        
        self.results["accuracy"] = {
            "total_signals": total,
            "total_signals_all": len(audit_records),
            "training_signals": len(audit_records) - len(test_records),
            "test_signals": len(test_records),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "mcc": round(mcc, 4),
            "mean_anomaly_score": round(np.mean(anomaly_scores), 2),
            "std_anomaly_score": round(np.std(anomaly_scores), 2),
            "median_anomaly_score": round(np.median(anomaly_scores), 2),
            "verdict_distribution": {v: verdicts.count(v) for v in set(verdicts)},
            "detected_anomalies": detected_anomalies,
            "missed_anomalies": missed_anomalies,
            "false_positive_count": len(false_positives),
            "layer_metrics": {
                "entropy_flagged": entropy_flags,
                "geometry_flagged": geometry_flags,
                "both_layers_flagged": both_layers,
                "entropy_flag_rate": round(entropy_flags / total, 4) if total > 0 else 0,
                "geometry_flag_rate": round(geometry_flags / total, 4) if total > 0 else 0
            }
        }
        
        print("\n[ACCURACY METRICS] TEST SET ONLY")
        print(f"  Test signals: {len(test_records)} (Training: {len(audit_records) - len(test_records)})")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  MCC: {mcc:.4f}")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        if detected_anomalies:
            print(f"  Detected Anomalies: {len(detected_anomalies)}")
        if missed_anomalies:
            print(f"  Missed Anomalies: {missed_anomalies}")
    
    def record_throughput(self, n_signals: int, total_time: float):
        """Record processing throughput"""
        throughput = n_signals / total_time if total_time > 0 else 0
        self.results["throughput"] = {
            "signals_per_second": round(throughput, 2),
            "seconds_per_signal": round(1/throughput if throughput > 0 else 0, 6),
            "total_signals": n_signals,
            "total_time_sec": round(total_time, 4)
        }
        print(f"\n[THROUGHPUT]")
        print(f"  {throughput:.2f} signals/sec")
        print(f"  {1/throughput if throughput > 0 else 0:.6f} sec/signal")
    
    def generate_visualizations(self, audit_records: List[Dict]):
        """Generate comprehensive scientific visualization suite - TEST SET ONLY"""
        if not MATPLOTLIB_AVAILABLE:
            print("\n[VISUALIZATION] matplotlib not available, skipping chart generation")
            return
        
        # CRITICAL FIX: Only visualize test set
        test_records = [r for r in audit_records if not r.get("is_training_data", False)]
        if not test_records:
            test_records = audit_records
        
        # Set scientific styling
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_context("paper")
        else:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        print(f"\n[VISUALIZATION] Generating charts for TEST SET: {len(test_records)} signals...")
        
        # Extract data for visualizations
        verdicts = []
        origins = []
        anomaly_scores = []
        y_true = []  # Ground truth labels
        y_pred = []  # Predicted labels
        
        for record in test_records:
            verdict = record.get("verdict", "")
            signal_id = record.get("signal_id", "")
            score = record.get("anomaly_score", 0)
            
            verdicts.append(verdict)
            anomaly_scores.append(score)
            
            # Ground truth: 1 for anomaly, 0 for natural
            is_anomaly = signal_id.startswith("ANOMALY_")
            y_true.append(1 if is_anomaly else 0)
            
            # Prediction: 1 for detected, 0 for not detected
            is_detected = verdict in ("Non-Natural", "Candidate — Requires Review")
            y_pred.append(1 if is_detected else 0)
        
        # Get accuracy metrics from results
        acc = self.results.get("accuracy", {})
        tp = acc.get("true_positives", 0)
        fp = acc.get("false_positives", 0)
        tn = acc.get("true_negatives", 0)
        fn = acc.get("false_negatives", 0)
        
        # Generate all visualizations
        self._plot_confusion_matrix(tp, fp, tn, fn)
        self._plot_anomaly_score_distribution(anomaly_scores, y_true)
        self._plot_precision_recall_curve(anomaly_scores, y_true)
        self._plot_roc_curve(anomaly_scores, y_true)
        self._plot_layer_performance()
        self._plot_detection_funnel(test_records)  # Use test_records
        self._plot_verdict_distribution(test_records)  # Use test_records
        self._plot_performance_timeline()
        self._plot_memory_usage()
        
        print(f"[VISUALIZATION] All charts saved to: {self.output_dir}/")
    
    def generate_audit_report(self, audit_records: List[Dict], output_path: str = None):
        """Generate comprehensive audit report similar to audit_log_report.txt - TEST SET ONLY"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "benchmark_audit_report.txt")
        
        # CRITICAL FIX: Only report on test set
        test_records = [r for r in audit_records if not r.get("is_training_data", False)]
        if not test_records:
            test_records = audit_records
        
        acc = self.results.get("accuracy", {})
        
        # Collect statistics from TEST SET ONLY
        anomalies = [r for r in test_records if r.get("signal_id", "").startswith("ANOMALY_")]
        candidates = [r for r in test_records if r.get("verdict") == "Candidate — Requires Review"]
        non_natural = [r for r in test_records if r.get("verdict") == "Non-Natural"]
        
        # Sort by anomaly score
        top_candidates = sorted(candidates, key=lambda x: x.get("anomaly_score", 0), reverse=True)[:25]
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("  AXIOM-ASTROPHYSICS BENCHMARK AUDIT REPORT\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Total signals analyzed: {len(audit_records)}\n")
            f.write(f"  TEST SET ONLY: {len(test_records)} signals\n")
            f.write(f"  Training set: {len(audit_records) - len(test_records)} signals (excluded from metrics)\n")
            f.write(f"  Training set: {len(audit_records) - len(test_records)} signals (excluded from metrics)\n")
            f.write("="*80 + "\n\n")
            
            # Executive Summary
            f.write("="*80 + "\n")
            f.write("  I. EXECUTIVE SUMMARY (TEST SET ONLY)\n")
            f.write("="*80 + "\n")
            f.write(f"  Test signals analyzed:           {len(test_records)}\n")
            f.write(f"  Training signals (excluded):     {len(audit_records) - len(test_records)}\n")
            f.write(f"  Non-Natural detections:          {len(non_natural)}\n")
            f.write(f"  Candidates requiring review:     {len(candidates)}\n")
            f.write(f"  Known anomalies in dataset:      {len(anomalies)}\n")
            f.write(f"  Detection rate (Recall):         {acc.get('recall', 0):.2%}\n")
            f.write(f"  False positive rate:             {1 - acc.get('specificity', 1):.2%}\n\n")
            
            # Performance Metrics
            f.write("="*80 + "\n")
            f.write("  II. PERFORMANCE METRICS\n")
            f.write("="*80 + "\n")
            
            total_time = sum(m.get('wall_time_sec', 0) for m in self.results.get('metrics', {}).values())
            f.write(f"  Total wall time:                 {total_time:.2f}s\n")
            f.write(f"  Throughput:                      {len(audit_records)/total_time:.2f} signals/sec\n")
            f.write(f"  Average time per signal:         {total_time/len(audit_records)*1000:.2f}ms\n")
            
            if self.results.get("c_core_enabled"):
                f.write(f"  C Core acceleration:             ENABLED\n")
            else:
                f.write(f"  C Core acceleration:             DISABLED (Python fallback)\n")
            f.write("\n")
            
            # Accuracy Metrics
            f.write("="*80 + "\n")
            f.write("  III. DETECTION ACCURACY\n")
            f.write("="*80 + "\n")
            f.write(f"  Precision:                       {acc.get('precision', 0):.4f}\n")
            f.write(f"  Recall (Sensitivity):            {acc.get('recall', 0):.4f}\n")
            f.write(f"  Specificity:                     {acc.get('specificity', 0):.4f}\n")
            f.write(f"  F1 Score:                        {acc.get('f1_score', 0):.4f}\n")
            f.write(f"  Accuracy:                        {acc.get('accuracy', 0):.4f}\n")
            f.write(f"  Matthews Correlation (MCC):      {acc.get('mcc', 0):.4f}\n")
            f.write(f"  True Positives:                  {acc.get('true_positives', 0)}\n")
            f.write(f"  False Positives:                 {acc.get('false_positives', 0)}\n")
            f.write(f"  True Negatives:                  {acc.get('true_negatives', 0)}\n")
            f.write(f"  False Negatives:                 {acc.get('false_negatives', 0)}\n\n")
            
            # Layer Performance
            layer_metrics = acc.get('layer_metrics', {})
            if layer_metrics:
                f.write("="*80 + "\n")
                f.write("  IV. LAYER-BY-LAYER ANALYSIS\n")
                f.write("="*80 + "\n")
                f.write(f"  Entropy layer flagged:           {layer_metrics.get('entropy_flagged', 0)} ({layer_metrics.get('entropy_flag_rate', 0):.2%})\n")
                f.write(f"  Geometry layer flagged:          {layer_metrics.get('geometry_flagged', 0)} ({layer_metrics.get('geometry_flag_rate', 0):.2%})\n")
                f.write(f"  Both layers flagged:             {layer_metrics.get('both_layers_flagged', 0)}\n\n")
            
            # Top Candidates
            if top_candidates:
                f.write("="*80 + "\n")
                f.write("  V. TOP 25 CANDIDATES (Ranked by Anomaly Score)\n")
                f.write("="*80 + "\n")
                f.write(f"  {'#':<4} {'Signal ID':<32} {'Score':<6} {'Verdict':<30}\n")
                f.write("  " + "-"*76 + "\n")
                
                for i, cand in enumerate(top_candidates, 1):
                    sig_id = cand.get('signal_id', 'Unknown')[:30]
                    score = cand.get('anomaly_score', 0)
                    verdict = cand.get('verdict', 'Unknown')[:28]
                    f.write(f"  {i:<4} {sig_id:<32} {score:<6.1f} {verdict:<30}\n")
                f.write("\n")
            
            # Detected Anomalies
            detected = acc.get('detected_anomalies', [])
            if detected:
                f.write("="*80 + "\n")
                f.write("  VI. DETECTED KNOWN ANOMALIES\n")
                f.write("="*80 + "\n")
                for anom in detected:
                    f.write(f"  ✓ {anom}\n")
                f.write("\n")
            
            # Missed Anomalies
            missed = acc.get('missed_anomalies', [])
            if missed:
                f.write("="*80 + "\n")
                f.write("  VII. MISSED ANOMALIES (FALSE NEGATIVES)\n")
                f.write("="*80 + "\n")
                for anom in missed:
                    f.write(f"  ✗ {anom}\n")
                f.write("\n")
            
            # System Info
            f.write("="*80 + "\n")
            f.write("  VIII. SYSTEM INFORMATION\n")
            f.write("="*80 + "\n")
            sys_info = self.results.get('system_info', {})
            f.write(f"  CPU cores (physical):            {sys_info.get('cpu_count', 'N/A')}\n")
            f.write(f"  CPU cores (logical):             {sys_info.get('cpu_count_logical', 'N/A')}\n")
            f.write(f"  CPU frequency:                   {sys_info.get('cpu_freq_mhz', 0):.0f} MHz\n")
            f.write(f"  Total memory:                    {sys_info.get('total_memory_gb', 0):.2f} GB\n")
            f.write(f"  Platform:                        {sys_info.get('platform', 'Unknown')}\n")
            f.write(f"  Python version:                  {sys_info.get('python_version', 'Unknown').split()[0]}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("  END OF BENCHMARK AUDIT REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n[AUDIT REPORT] Generated: {output_path}")
        return output_path
    
    def _plot_confusion_matrix(self, tp, fp, tn, fn):
        """Generate confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        
        if SEABORN_AVAILABLE:
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Natural', 'Non-Natural'],
                       yticklabels=['Natural', 'Non-Natural'],
                       cbar_kws={'label': 'Count'}, ax=ax)
        else:
            im = ax.imshow(confusion_matrix, cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Natural', 'Non-Natural'])
            ax.set_yticklabels(['Natural', 'Non-Natural'])
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(confusion_matrix[i, j]),
                           ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax, label='Count')
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix - AXIOM Detection System', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Confusion matrix: {output_path}")
    
    def _plot_anomaly_score_distribution(self, anomaly_scores, y_true):
        """Generate anomaly score distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate scores by ground truth
        anomaly_scores = np.array(anomaly_scores)
        y_true = np.array(y_true)
        
        natural_scores = anomaly_scores[y_true == 0]
        anomaly_scores_true = anomaly_scores[y_true == 1]
        
        # Plot histograms
        bins = np.linspace(0, 100, 21)
        ax.hist(natural_scores, bins=bins, alpha=0.6, label='Natural Signals', 
               color='blue', edgecolor='black')
        ax.hist(anomaly_scores_true, bins=bins, alpha=0.6, label='Known Anomalies', 
               color='red', edgecolor='black')
        
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "anomaly_score_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Anomaly score distribution: {output_path}")
    
    def _plot_precision_recall_curve(self, anomaly_scores, y_true):
        """Generate precision-recall curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        anomaly_scores = np.array(anomaly_scores)
        y_true = np.array(y_true)
        
        # Calculate precision and recall at different thresholds
        thresholds = np.linspace(0, 100, 101)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (anomaly_scores >= threshold).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        ax.plot(recalls, precisions, linewidth=2, color='darkblue')
        ax.fill_between(recalls, precisions, alpha=0.2, color='blue')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # Add diagonal reference line
        ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, linewidth=1, label='Random Classifier')
        ax.legend(loc='lower left', fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "precision_recall_curve.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Precision-recall curve: {output_path}")
    
    def _plot_roc_curve(self, anomaly_scores, y_true):
        """Generate ROC curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        anomaly_scores = np.array(anomaly_scores)
        y_true = np.array(y_true)
        
        # Calculate TPR and FPR at different thresholds
        thresholds = np.linspace(0, 100, 101)
        tprs = []
        fprs = []
        
        for threshold in thresholds:
            y_pred = (anomaly_scores >= threshold).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        # Sort by FPR for proper curve
        sorted_indices = np.argsort(fprs)
        fprs = np.array(fprs)[sorted_indices]
        tprs = np.array(tprs)[sorted_indices]
        
        # Calculate AUC using trapezoidal rule
        try:
            auc = np.trapezoid(tprs, fprs)  # NumPy 2.0+
        except AttributeError:
            auc = np.trapz(tprs, fprs)  # NumPy < 2.0
        
        ax.plot(fprs, tprs, linewidth=2, color='darkgreen', label=f'ROC Curve (AUC = {auc:.3f})')
        ax.fill_between(fprs, tprs, alpha=0.2, color='green')
        
        # Add diagonal reference line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random Classifier (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ROC curve: {output_path}")
    
    def _plot_layer_performance(self):
        """Generate layer-by-layer performance breakdown"""
        if not self.layer_timings:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of average layer times
        layers = list(self.layer_timings.keys())
        avg_times = [np.mean(self.layer_timings[l]) for l in layers]
        
        ax1.barh(layers, avg_times, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Average Time (seconds)', fontsize=12)
        ax1.set_title('Layer Performance Breakdown', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Pie chart of time distribution
        ax2.pie(avg_times, labels=layers, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Time Distribution by Layer', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "layer_performance.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Layer performance: {output_path}")
    
    def _plot_detection_funnel(self, audit_records: List[Dict]):
        """Generate detection funnel visualization"""
        total = len(audit_records)
        entropy_flagged = sum(1 for r in audit_records if "Low" in r.get("entropy_label", ""))
        geometry_flagged = sum(1 for r in audit_records if r.get("geometric_anomaly", "") != "None Detected")
        both_flagged = sum(1 for r in audit_records 
                          if "Low" in r.get("entropy_label", "") and r.get("geometric_anomaly", "") != "None Detected")
        candidates = sum(1 for r in audit_records if r.get("verdict") == "Candidate — Requires Review")
        non_natural = sum(1 for r in audit_records if r.get("verdict") == "Non-Natural")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        stages = ['Total Signals', 'Entropy Flagged', 'Geometry Flagged', 
                 'Both Layers', 'Candidates', 'Non-Natural']
        values = [total, entropy_flagged, geometry_flagged, both_flagged, candidates, non_natural]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#c0392b']
        
        # Create funnel
        for i, (stage, value, color) in enumerate(zip(stages, values, colors)):
            width = value / total
            ax.barh(i, width, height=0.8, color=color, edgecolor='black', linewidth=2)
            
            # Add labels
            percentage = (value / total) * 100
            ax.text(width/2, i, f'{stage}\n{value:,} ({percentage:.2f}%)', 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(stages) - 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Proportion of Total Signals', fontsize=12)
        ax.set_title('Detection Funnel - Signal Processing Pipeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "detection_funnel.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Detection funnel: {output_path}")
    
    def _plot_verdict_distribution(self, audit_records: List[Dict]):
        """Generate verdict distribution pie chart"""
        verdicts = [r.get("verdict", "Unknown") for r in audit_records]
        verdict_counts = {}
        for v in set(verdicts):
            verdict_counts[v] = verdicts.count(v)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {
            'Natural': '#2ecc71',
            'Interference': '#95a5a6',
            'Candidate — Requires Review': '#f39c12',
            'Non-Natural': '#e74c3c'
        }
        
        labels = list(verdict_counts.keys())
        sizes = list(verdict_counts.values())
        plot_colors = [colors.get(l, '#3498db') for l in labels]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=plot_colors, startangle=90,
                                           textprops={'fontsize': 11, 'weight': 'bold'})
        
        # Add count to labels
        for i, (label, size) in enumerate(zip(labels, sizes)):
            texts[i].set_text(f'{label}\n({size:,})')
        
        ax.set_title('Verdict Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "verdict_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Verdict distribution: {output_path}")
    
    def _plot_performance_timeline(self):
        """Generate performance timeline chart"""
        if not self.results.get('metrics'):
            return
        
        phases = list(self.results['metrics'].keys())
        wall_times = [self.results['metrics'][p]['wall_time_sec'] for p in phases]
        cpu_times = [self.results['metrics'][p]['cpu_time_sec'] for p in phases]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(phases))
        width = 0.35
        
        ax.bar(x - width/2, wall_times, width, label='Wall Time', color='steelblue', edgecolor='black')
        ax.bar(x + width/2, cpu_times, width, label='CPU Time', color='coral', edgecolor='black')
        
        ax.set_xlabel('Phase', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Performance Timeline by Phase', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "performance_timeline.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Performance timeline: {output_path}")
    
    def _plot_memory_usage(self):
        """Generate memory usage chart"""
        if not self.results.get('metrics'):
            return
        
        phases = list(self.results['metrics'].keys())
        peak_mem = [self.results['metrics'][p]['peak_memory_mb'] for p in phases]
        mem_delta = [self.results['metrics'][p]['memory_delta_mb'] for p in phases]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Peak memory
        ax1.plot(phases, peak_mem, marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax1.fill_between(range(len(phases)), peak_mem, alpha=0.3, color='green')
        ax1.set_xlabel('Phase', fontsize=12)
        ax1.set_ylabel('Memory (MB)', fontsize=12)
        ax1.set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory delta
        colors = ['red' if d > 0 else 'blue' for d in mem_delta]
        ax2.bar(phases, mem_delta, color=colors, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Phase', fontsize=12)
        ax2.set_ylabel('Memory Delta (MB)', fontsize=12)
        ax2.set_title('Memory Change by Phase', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "memory_usage.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Memory usage: {output_path}")
    
    def save(self):
        """Save benchmark results to JSON in timestamped directory"""
        output_path = os.path.join(self.output_dir, "benchmark_results.json")
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n[BENCHMARK] Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("AXIOM-ASTROPHYSICS BENCHMARK SUMMARY")
        print("="*80)
        
        print("\nSYSTEM INFO:")
        for k, v in self.results["system_info"].items():
            print(f"  {k}: {v}")
        
        print("\nPHASE TIMINGS:")
        total_wall = 0
        for phase, metrics in self.results["metrics"].items():
            print(f"  {phase}:")
            print(f"    Wall: {metrics['wall_time_sec']:.4f}s | "
                  f"CPU: {metrics['cpu_time_sec']:.4f}s | "
                  f"Mem: {metrics['peak_memory_mb']:.2f} MB")
            total_wall += metrics['wall_time_sec']
        print(f"  TOTAL WALL TIME: {total_wall:.4f}s")
        
        if "accuracy" in self.results:
            print("\nACCURACY:")
            acc = self.results["accuracy"]
            print(f"  Precision: {acc['precision']:.4f}")
            print(f"  Recall: {acc['recall']:.4f}")
            print(f"  F1 Score: {acc['f1_score']:.4f}")
            print(f"  Specificity: {acc['specificity']:.4f}")
        
        if "throughput" in self.results:
            print("\nTHROUGHPUT:")
            tp = self.results["throughput"]
            print(f"  {tp['signals_per_second']:.2f} signals/sec")
        
        print("="*80)

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="AXIOM-ASTROPHYSICS Comprehensive Benchmark Suite v1.0")
    parser.add_argument("--dataset", required=True, help="Input dataset JSON file")
    parser.add_argument("--output", default="audit_log.json", help="Output audit log JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-c-core", action="store_true", help="Use C core shared library if available")
    parser.add_argument("--use-c-standalone", action="store_true", help="Use C standalone executable (fastest)")
    args = parser.parse_args()
    
    # Import the pipeline
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from axiom_astrophysics_v1 import AxiomPipeline, PipelineConfig
    
    # Create benchmark instance
    bench = AxiomBenchmark(use_c_core=args.use_c_core, use_c_standalone=args.use_c_standalone)
    
    print("\n" + "="*80)
    print("AXIOM-ASTROPHYSICS COMPREHENSIVE BENCHMARK v1.0")
    print("="*80)
    
    if args.use_c_standalone and bench.c_standalone_path:
        print("[MODE] C Standalone Executable (Maximum Performance)")
        print()
        
        # Run C standalone executable
        bench.start_phase("C Standalone Execution")
        try:
            c_metrics = bench.run_c_standalone(args.dataset, args.output)
            bench.end_phase(c_metrics)
        except Exception as e:
            print(f"[ERROR] C standalone failed: {e}")
            print("[FALLBACK] Switching to Python pipeline")
            bench.start_phase("Full Pipeline Execution")
            config = PipelineConfig(
                dataset_path=args.dataset,
                output_path=args.output,
                random_seed=args.seed,
            )
            AxiomPipeline().run(config)
            bench.end_phase()
    else:
        if bench.c_core_lib:
            print("[MODE] Python Pipeline with C Core Acceleration")
        else:
            print("[MODE] Python Pipeline (Pure Python)")
        print()
        
        # Run full pipeline with benchmarking
        bench.start_phase("Full Pipeline Execution")
        config = PipelineConfig(
            dataset_path=args.dataset,
            output_path=args.output,
            random_seed=args.seed,
        )
        AxiomPipeline().run(config)
        bench.end_phase()
    
    # Load results and compute accuracy
    bench.start_phase("Accuracy Analysis")
    with open(args.output, "r") as f:
        audit_data = json.load(f)
    audit_records = audit_data.get("audit_records", [])
    bench.record_accuracy_metrics(audit_records)
    bench.end_phase()
    
    # Generate audit report
    bench.start_phase("Audit Report Generation")
    bench.generate_audit_report(audit_records)
    bench.end_phase()
    
    # Generate comprehensive visualizations
    bench.start_phase("Visualization Generation")
    bench.generate_visualizations(audit_records)
    bench.end_phase()
    
    # Save and display results
    output_path = bench.save()
    bench.print_summary()
    
    print(f"\n[COMPLETE] Benchmark results saved to: {output_path}")
    print(f"[COMPLETE] Audit report: {os.path.join(bench.output_dir, 'benchmark_audit_report.txt')}")
    print(f"[COMPLETE] Visualizations: {bench.output_dir}/")
    print("="*80)
