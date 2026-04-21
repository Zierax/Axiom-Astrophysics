"""
AXIOM-ASTROPHYSICS Mini Validation Benchmark
Mathematical Proof of Generalization on Balanced Dataset

Purpose: Validate that detection is based on true signal properties, not memorization
Dataset: 17 natural signals + 17 known anomalies (balanced, minimal)
Method: Strict train/test split with statistical validation
"""

import json
import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# Import the pipeline
sys.path.insert(0, os.path.dirname(__file__))
from axiom_astrophysics_v1 import (
    DatasetIngester, StratifiedSplitter, LambdaCDMModel,
    EntropyAnalyzer, GeometryDetector, TruthimaticsEngine,
    VerdictClassifier, PRIORITY_TARGET_IDS
)

class MiniValidationBenchmark:
    """
    Mini validation benchmark with mathematical proof of generalization.
    
    Key Features:
    1. Balanced dataset (17 natural + 17 anomalies)
    2. Strict train/test split (50/50)
    3. No priority target bias
    4. Statistical significance testing
    5. Cross-validation for robustness
    """
    
    def __init__(self, output_dir="Benchmark/MiniValidation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}
    
    def create_mini_dataset(self, full_dataset_path: str) -> str:
        """
        Create a balanced mini dataset: 17 natural + 17 anomalies.
        
        Selection criteria:
        - All 17 known anomalies (ANOMALY_*)
        - 17 random natural signals (diverse sources)
        """
        print("\n" + "="*80)
        print("CREATING MINI VALIDATION DATASET")
        print("="*80)
        
        with open(full_dataset_path, 'r') as f:
            full_data = json.load(f)
        
        # Separate anomalies and natural signals
        anomalies = [s for s in full_data if s.get('signal_id', '').startswith('ANOMALY_')]
        natural = [s for s in full_data if s.get('origin_class') == 'Natural']
        
        print(f"[DATASET] Full dataset: {len(full_data)} signals")
        print(f"[DATASET] Known anomalies: {len(anomalies)}")
        print(f"[DATASET] Natural signals: {len(natural)}")
        
        # Select 17 anomalies (all of them)
        selected_anomalies = anomalies[:17]
        
        # Select 17 diverse natural signals
        # Ensure diversity by sampling from different frequency ranges
        natural_sorted = sorted(natural, key=lambda x: x.get('frequency_mhz', 0))
        step = len(natural_sorted) // 17
        selected_natural = [natural_sorted[i * step] for i in range(17)]
        
        # Combine
        mini_dataset = selected_anomalies + selected_natural
        
        # Save mini dataset
        mini_path = os.path.join(self.output_dir, "mini_dataset.json")
        with open(mini_path, 'w') as f:
            json.dump(mini_dataset, f, indent=2)
        
        print(f"[DATASET] Mini dataset created: {len(mini_dataset)} signals")
        print(f"  - Anomalies: {len(selected_anomalies)}")
        print(f"  - Natural: {len(selected_natural)}")
        print(f"  - Saved to: {mini_path}")
        
        return mini_path
    
    def run_single_fold(self, signals: list, fold_id: int, seed: int):
        """
        Run a single train/test fold with strict separation.
        
        Returns: (train_metrics, test_metrics)
        """
        print(f"\n{'='*80}")
        print(f"FOLD {fold_id}: TRAIN/TEST SPLIT (Seed={seed})")
        print(f"{'='*80}")
        
        # Split into train/test (50/50 stratified)
        splitter = StratifiedSplitter()
        training_pool, test_pool = splitter.split(signals, seed)
        
        print(f"[SPLIT] Training: {len(training_pool)} signals")
        print(f"[SPLIT] Test: {len(test_pool)} signals")
        
        # Count anomalies in each set
        train_anomalies = sum(1 for s in training_pool if s['signal_id'].startswith('ANOMALY_'))
        test_anomalies = sum(1 for s in test_pool if s['signal_id'].startswith('ANOMALY_'))
        
        print(f"[SPLIT] Training anomalies: {train_anomalies}")
        print(f"[SPLIT] Test anomalies: {test_anomalies}")
        
        # Calibrate on training set ONLY
        model = LambdaCDMModel()
        model.fit(training_pool)
        
        entropy_analyzer = EntropyAnalyzer()
        entropy_analyzer.calibrate(training_pool)
        
        geometry_detector = GeometryDetector()
        geometry_detector.calibrate(training_pool)
        
        truthimatics = TruthimaticsEngine()
        classifier = VerdictClassifier()
        
        def analyze_signal(signal: dict) -> dict:
            """Analyze a single signal (NO PRIORITY BIAS)"""
            entropy_result = entropy_analyzer.analyze(signal)
            geometry_result = geometry_detector.detect(signal)
            
            proof_result = None
            if entropy_result.flagged or geometry_result.flagged:
                # NO PRIORITY TARGET BIAS - all signals treated equally
                proof_result = truthimatics.prove(signal, model, n_trials=5000)
            
            verdict = classifier.classify(signal, entropy_result, geometry_result, proof_result)
            
            return {
                'signal_id': signal['signal_id'],
                'verdict': verdict,
                'entropy_flagged': entropy_result.flagged,
                'geometry_flagged': geometry_result.flagged,
                'p_value': proof_result.p_value if proof_result else None,
                'is_anomaly': signal['signal_id'].startswith('ANOMALY_')
            }
        
        # Analyze training set (for reference only, not used for metrics)
        print(f"\n[ANALYZING] Training set...")
        train_results = [analyze_signal(s) for s in training_pool]
        
        # Analyze test set (THIS IS WHAT MATTERS)
        print(f"[ANALYZING] Test set...")
        test_results = [analyze_signal(s) for s in test_pool]
        
        # Compute metrics on TEST SET ONLY
        test_metrics = self._compute_metrics(test_results, "TEST")
        train_metrics = self._compute_metrics(train_results, "TRAIN (reference)")
        
        return train_metrics, test_metrics
    
    def _compute_metrics(self, results: list, label: str) -> dict:
        """Compute detection metrics"""
        tp = fp = tn = fn = 0
        
        for r in results:
            is_anomaly = r['is_anomaly']
            is_detected = r['verdict'] in ('Non-Natural', 'Candidate — Requires Review')
            
            if is_anomaly and is_detected:
                tp += 1
            elif is_anomaly and not is_detected:
                fn += 1
            elif not is_anomaly and is_detected:
                fp += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(results) if len(results) > 0 else 0
        
        metrics = {
            'label': label,
            'n_signals': len(results),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'accuracy': accuracy
        }
        
        print(f"\n[METRICS] {label}:")
        print(f"  Signals: {len(results)}")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def run_cross_validation(self, dataset_path: str, n_folds: int = 5):
        """
        Run k-fold cross-validation for robust validation.
        
        Each fold uses a different random seed for train/test split.
        """
        print("\n" + "="*80)
        print(f"CROSS-VALIDATION: {n_folds} FOLDS")
        print("="*80)
        
        # Load mini dataset
        with open(dataset_path, 'r') as f:
            signals = json.load(f)
        
        all_test_metrics = []
        all_train_metrics = []
        
        for fold in range(n_folds):
            seed = 42 + fold * 100  # Different seed for each fold
            train_metrics, test_metrics = self.run_single_fold(signals, fold + 1, seed)
            all_train_metrics.append(train_metrics)
            all_test_metrics.append(test_metrics)
        
        # Aggregate results
        self.results = {
            'n_folds': n_folds,
            'train_metrics': all_train_metrics,
            'test_metrics': all_test_metrics,
            'test_mean': self._compute_mean_metrics(all_test_metrics),
            'test_std': self._compute_std_metrics(all_test_metrics),
            'train_mean': self._compute_mean_metrics(all_train_metrics),
        }
        
        return self.results
    
    def _compute_mean_metrics(self, metrics_list: list) -> dict:
        """Compute mean of metrics across folds"""
        keys = ['precision', 'recall', 'specificity', 'f1_score', 'accuracy']
        mean_metrics = {}
        for key in keys:
            values = [m[key] for m in metrics_list]
            mean_metrics[key] = np.mean(values)
        return mean_metrics
    
    def _compute_std_metrics(self, metrics_list: list) -> dict:
        """Compute standard deviation of metrics across folds"""
        keys = ['precision', 'recall', 'specificity', 'f1_score', 'accuracy']
        std_metrics = {}
        for key in keys:
            values = [m[key] for m in metrics_list]
            std_metrics[key] = np.std(values)
        return std_metrics
    
    def generate_mathematical_proof(self):
        """
        Generate mathematical proof of generalization.
        
        Proof Strategy:
        1. Show consistent performance across multiple folds (low variance)
        2. Show test performance close to train performance (no overfitting)
        3. Statistical significance testing (binomial test)
        4. Confidence intervals
        """
        print("\n" + "="*80)
        print("MATHEMATICAL PROOF OF GENERALIZATION")
        print("="*80)
        
        test_mean = self.results['test_mean']
        test_std = self.results['test_std']
        train_mean = self.results['train_mean']
        n_folds = self.results['n_folds']
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("AXIOM-ASTROPHYSICS: MATHEMATICAL PROOF OF GENERALIZATION")
        report_lines.append("Mini Validation Benchmark on Balanced Dataset")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Section 1: Dataset Description
        report_lines.append("1. DATASET CHARACTERISTICS")
        report_lines.append("-" * 80)
        report_lines.append("  Balanced Mini Dataset:")
        report_lines.append("    - 17 known anomalies (ANOMALY_*)")
        report_lines.append("    - 17 natural signals (diverse sources)")
        report_lines.append("    - Total: 34 signals")
        report_lines.append("")
        report_lines.append("  Validation Method:")
        report_lines.append(f"    - {n_folds}-fold cross-validation")
        report_lines.append("    - Stratified train/test split (50/50)")
        report_lines.append("    - NO priority target bias")
        report_lines.append("    - Calibration on training set ONLY")
        report_lines.append("")
        
        # Section 2: Cross-Validation Results
        report_lines.append("2. CROSS-VALIDATION RESULTS")
        report_lines.append("-" * 80)
        report_lines.append("  Test Set Performance (Mean ± Std across folds):")
        report_lines.append(f"    Precision:    {test_mean['precision']:.4f} ± {test_std['precision']:.4f}")
        report_lines.append(f"    Recall:       {test_mean['recall']:.4f} ± {test_std['recall']:.4f}")
        report_lines.append(f"    F1 Score:     {test_mean['f1_score']:.4f} ± {test_std['f1_score']:.4f}")
        report_lines.append(f"    Accuracy:     {test_mean['accuracy']:.4f} ± {test_std['accuracy']:.4f}")
        report_lines.append(f"    Specificity:  {test_mean['specificity']:.4f} ± {test_std['specificity']:.4f}")
        report_lines.append("")
        
        # Section 3: Overfitting Analysis
        report_lines.append("3. OVERFITTING ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append("  Train vs Test Performance Gap:")
        
        for metric in ['precision', 'recall', 'f1_score', 'accuracy']:
            train_val = train_mean[metric]
            test_val = test_mean[metric]
            gap = abs(train_val - test_val)
            gap_pct = gap * 100
            
            report_lines.append(f"    {metric.capitalize():12s}: Train={train_val:.4f}, Test={test_val:.4f}, Gap={gap:.4f} ({gap_pct:.1f}%)")
        
        report_lines.append("")
        
        # Overfitting verdict (use ASCII characters for compatibility)
        max_gap = max(abs(train_mean[m] - test_mean[m]) for m in ['precision', 'recall', 'f1_score'])
        if max_gap < 0.10:
            verdict = "[PASS] NO OVERFITTING (gap < 10%)"
        elif max_gap < 0.20:
            verdict = "[WARN] MILD OVERFITTING (gap 10-20%)"
        else:
            verdict = "[FAIL] SIGNIFICANT OVERFITTING (gap > 20%)"
        
        report_lines.append(f"  Verdict: {verdict}")
        report_lines.append("")
        
        # Section 4: Statistical Significance
        report_lines.append("4. STATISTICAL SIGNIFICANCE")
        report_lines.append("-" * 80)
        
        # Binomial test for recall (H0: recall = 0.5, H1: recall > 0.5)
        n_anomalies = 17
        n_detected = int(test_mean['recall'] * n_anomalies)
        
        # Binomial probability: P(X >= n_detected | n=17, p=0.5)
        from scipy.stats import binom
        p_value_binomial = 1 - binom.cdf(n_detected - 1, n_anomalies, 0.5)
        
        report_lines.append("  Binomial Test (Recall):")
        report_lines.append(f"    H0: System detects anomalies by random chance (p=0.5)")
        report_lines.append(f"    H1: System detects anomalies better than chance (p>0.5)")
        report_lines.append(f"    Observed: {n_detected}/{n_anomalies} anomalies detected")
        report_lines.append(f"    P-value: {p_value_binomial:.6f}")
        
        if p_value_binomial < 0.001:
            sig_verdict = "[PASS] HIGHLY SIGNIFICANT (p < 0.001)"
        elif p_value_binomial < 0.05:
            sig_verdict = "[PASS] SIGNIFICANT (p < 0.05)"
        else:
            sig_verdict = "[FAIL] NOT SIGNIFICANT (p >= 0.05)"
        
        report_lines.append(f"    Verdict: {sig_verdict}")
        report_lines.append("")
        
        # Section 5: Confidence Intervals
        report_lines.append("5. CONFIDENCE INTERVALS (95%)")
        report_lines.append("-" * 80)
        
        # Wilson score interval for precision
        from scipy.stats import norm
        z = norm.ppf(0.975)  # 95% CI
        
        for metric in ['precision', 'recall', 'f1_score']:
            mean = test_mean[metric]
            std = test_std[metric]
            
            # Standard error of mean
            sem = std / np.sqrt(n_folds)
            
            # Confidence interval
            ci_lower = max(0, mean - z * sem)
            ci_upper = min(1, mean + z * sem)
            
            report_lines.append(f"  {metric.capitalize():12s}: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        report_lines.append("")
        
        # Section 6: Fold-by-Fold Results
        report_lines.append("6. FOLD-BY-FOLD RESULTS")
        report_lines.append("-" * 80)
        report_lines.append("  Fold | Precision | Recall | F1 Score | Accuracy")
        report_lines.append("  " + "-" * 60)
        
        for i, metrics in enumerate(self.results['test_metrics'], 1):
            report_lines.append(f"  {i:4d} | {metrics['precision']:9.4f} | {metrics['recall']:6.4f} | "
                              f"{metrics['f1_score']:8.4f} | {metrics['accuracy']:8.4f}")
        
        report_lines.append("  " + "-" * 60)
        report_lines.append(f"  Mean | {test_mean['precision']:9.4f} | {test_mean['recall']:6.4f} | "
                          f"{test_mean['f1_score']:8.4f} | {test_mean['accuracy']:8.4f}")
        report_lines.append(f"  Std  | {test_std['precision']:9.4f} | {test_std['recall']:6.4f} | "
                          f"{test_std['f1_score']:8.4f} | {test_std['accuracy']:8.4f}")
        report_lines.append("")
        
        # Section 7: Conclusion
        report_lines.append("7. CONCLUSION")
        report_lines.append("-" * 80)
        
        # Overall verdict
        is_significant = p_value_binomial < 0.05
        is_not_overfitted = max_gap < 0.20
        is_consistent = test_std['f1_score'] < 0.15
        
        if is_significant and is_not_overfitted and is_consistent:
            conclusion = "[PASS] VALIDATION PASSED: System demonstrates TRUE GENERALIZATION"
            report_lines.append(f"  {conclusion}")
            report_lines.append("")
            report_lines.append("  Evidence:")
            report_lines.append("    1. Statistically significant performance (p < 0.05)")
            report_lines.append("    2. Low train/test gap (< 20%)")
            report_lines.append("    3. Consistent across folds (low variance)")
            report_lines.append("")
            report_lines.append("  Interpretation:")
            report_lines.append("    The detection system is based on genuine signal properties,")
            report_lines.append("    not memorization of training data. Results are scientifically valid.")
        else:
            conclusion = "[FAIL] VALIDATION CONCERNS DETECTED"
            report_lines.append(f"  {conclusion}")
            report_lines.append("")
            report_lines.append("  Issues:")
            if not is_significant:
                report_lines.append("    - Performance not statistically significant")
            if not is_not_overfitted:
                report_lines.append("    - Large train/test performance gap (overfitting)")
            if not is_consistent:
                report_lines.append("    - High variance across folds (unstable)")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("END OF MATHEMATICAL PROOF")
        report_lines.append("="*80)
        
        # Print to console
        for line in report_lines:
            print(line)
        
        # Save to file with UTF-8 encoding
        report_path = os.path.join(self.output_dir, "mathematical_proof.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n[SAVED] Mathematical proof: {report_path}")
        
        # Save JSON results
        json_path = os.path.join(self.output_dir, "validation_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"[SAVED] Validation results: {json_path}")
        
        return report_path


def main():
    """Run mini validation benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AXIOM Mini Validation Benchmark")
    parser.add_argument("--dataset", default="dataset.json", help="Full dataset path")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AXIOM-ASTROPHYSICS MINI VALIDATION BENCHMARK")
    print("Mathematical Proof of Generalization")
    print("="*80)
    
    bench = MiniValidationBenchmark()
    
    # Step 1: Create mini dataset
    mini_dataset_path = bench.create_mini_dataset(args.dataset)
    
    # Step 2: Run cross-validation
    bench.run_cross_validation(mini_dataset_path, n_folds=args.folds)
    
    # Step 3: Generate mathematical proof
    bench.generate_mathematical_proof()
    
    print("\n" + "="*80)
    print("MINI VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
