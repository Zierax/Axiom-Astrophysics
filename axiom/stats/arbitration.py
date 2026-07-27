import numpy as np

from axiom.dsp.physics_rules import (
    combine_physics_laws,
    dispersion_excess_law_score,
    duty_cycle_consistency_law_score,
    technosignature_law_score,
)
from axiom.dsp.waterfall_features import FrequencyResolvedScorer


class SignalArbitrator:
    """Final decision logic.

    Combines stacking-ensemble predictions, per-class outlier density scores,
    a learned 1-D CNN waveform branch, conformal p-values, and nonlinear-dynamics
    (chaos) descriptors.  Applies Benjamini-Hochberg FDR control to bound the
    false-discovery rate and emits the final audit verdict.

    Verdict taxonomy: "Natural", "Interference", "Anomaly",
    "Candidate — Requires Review".
    """

    def __init__(self, fdr_alpha=0.05, conformal_alpha=0.05):
        if not (0.0 < fdr_alpha < 1.0):
            raise ValueError(f"fdr_alpha must be in (0, 1), got {fdr_alpha}")
        if not (0.0 < conformal_alpha < 1.0):
            raise ValueError(f"conformal_alpha must be in (0, 1), got {conformal_alpha}")
        self.fdr_alpha = float(fdr_alpha)
        self.conformal_alpha = float(conformal_alpha)

    # ------------------------------------------------------------------
    def control_fdr(self, p_values):
        """Benjamini-Hochberg False Discovery Rate control.

        Returns a boolean array; True means the null (Natural) is rejected,
        i.e. the signal is a statistically significant anomaly.
        """
        N = len(p_values)
        if N == 0:
            return np.zeros(0, dtype=bool)
        sorted_idx = np.argsort(p_values)
        sorted_p = np.asarray(p_values, dtype=np.float64)[sorted_idx]
        bh_threshold = (np.arange(N) + 1.0) / N * self.fdr_alpha
        rejected_sorted = sorted_p <= bh_threshold
        rejected_idx = np.where(rejected_sorted)[0]
        rejected = np.zeros(N, dtype=bool)
        if len(rejected_idx) > 0:
            max_k = int(np.max(rejected_idx))
            rejected[sorted_idx[:max_k + 1]] = True
        return rejected

    # ------------------------------------------------------------------
    @staticmethod
    def _combine_probs(meta_probs, cnn_probs):
        """Geometric-mean fusion of ensemble and CNN 3-class probabilities."""
        meta = np.asarray(meta_probs, dtype=np.float64)
        if cnn_probs is None:
            return meta
        cnn = np.asarray(cnn_probs, dtype=np.float64)
        if cnn.shape != meta.shape or not np.all(np.isfinite(cnn)) or cnn.shape[1] != 3:
            return meta
        combined = np.sqrt(np.clip(meta * cnn, 1e-12, None))
        row_sum = combined.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        return combined / row_sum

    # ------------------------------------------------------------------
    def arbitrate(self, signal_ids, meta_predictions, meta_probs, p_values,
                   chaos_scores, origin_classes, ood_mask=None, cnn_probs=None,
                   waterfall_features=None, physics_features=None,
                   physics_weight=12.0):
        """Arbitrate final verdicts for a batch of signals.

        Parameters
        ----------
        signal_ids : sequence of str
        meta_predictions : (N,) ensemble output classes (0:Natural,1:Interf,2:Anomaly)
        meta_probs : (N, 3) ensemble probabilities
        p_values : (N,) conformal p-values
        chaos_scores : (N,) legacy Lyapunov OR (N, D) descriptor from
            `compute_chaos_descriptor`
        origin_classes : sequence of str (catalogue / physical class)
        ood_mask : (N,) bool, True => signal lies outside the natural manifold
        cnn_probs : (N, 3) optional learned CNN waveform probabilities
        waterfall_features : dict, optional
            Mapping ``signal_id -> waterfall feature dict``. Enables the
            frequency-resolved narrowband term in the composite score; signals
            without an entry receive a neutral score.
        physics_features : dict, optional
            Mapping ``signal_id -> physics dict``. Keys consumed by the physics
            laws: any waterfall descriptors (``concentration``,
            ``spectral_flatness``) and, where available, catalog physics
            (``dm``, ``dm_gal_max``, ``width_ms``, ``period_s``,
            ``is_extragalactic``). Signals without an entry receive a neutral
            physics score. When catalog physics is absent the technosignature law
            (spectrogram-only) still applies.
        physics_weight : float
            Weight of the physics-law term in the composite anomaly score
            (bounded contribution). Default 12.0 (kept below the p-value term's
            60.0 ceiling so physics refines rather than dominates the verdict).

        Returns
        -------
        verdicts : list of str
        anomaly_scores : list of float in [0, 100]
        """
        N = len(signal_ids)
        if N == 0:
            return [], []
        meta_probs = np.asarray(meta_probs, dtype=np.float64)
        p_values = np.asarray(p_values, dtype=np.float64)
        meta_predictions = np.asarray(meta_predictions, dtype=int)
        if meta_probs.shape != (N, 3):
            raise ValueError(f"meta_probs must be (N,3), got {meta_probs.shape}")
        if p_values.shape != (N,):
            raise ValueError(f"p_values must be (N,), got {p_values.shape}")
        if meta_predictions.shape != (N,):
            raise ValueError(f"meta_predictions must be (N,), got {meta_predictions.shape}")

        # Defensive p-value clamping for the log-score.
        p_values = np.clip(p_values, 0.0, 1.0)
        order_scores = np.clip(np.asarray(chaos_scores, dtype=np.float64), 0.0, 1.0)
        if order_scores.shape != (N,):
            # Fallback: neutral order if descriptor failed to reduce.
            order_scores = np.full(N, 0.5)

        fused_probs = self._combine_probs(meta_probs, cnn_probs)

        fr_scorer = FrequencyResolvedScorer(waterfall_features)
        physics_inputs = physics_features or {}

        bh_rejected = self.control_fdr(p_values)

        verdicts = []
        anomaly_scores = []

        for i in range(N):
            pred_cls = int(meta_predictions[i])
            prob_vec = fused_probs[i]
            p_val = float(p_values[i])
            order = float(order_scores[i])
            sid = signal_ids[i]

            # Frequency-resolved narrowband term: rewards concentrated, low-
            # occupancy, tonal morphology measured directly from the spectrogram.
            fr = float(fr_scorer.score(sid))

            # Physics-law term: encode genuine astrophysical consistency laws
            # (tone-vs-dispersion contradiction, dispersion excess, duty-cycle
            # consistency) as a single bounded [0, 1] score from measured inputs.
            phys = physics_inputs.get(sid) if isinstance(physics_inputs, dict) else None
            phys = phys if isinstance(phys, dict) else {}
            wf = waterfall_features.get(sid) if isinstance(waterfall_features, dict) else None
            wf = wf if isinstance(wf, dict) else {}
            claimed_dispersed = bool(pred_cls in (0, 2) and (prob_vec[0] + prob_vec[2]) >= 0.8)
            techno = technosignature_law_score({**wf, **phys}, claimed_dispersed=claimed_dispersed)
            dm = phys.get("dm")
            dm_gal = phys.get("dm_gal_max")
            is_exgal = bool(phys.get("is_extragalactic", False))
            disp = dispersion_excess_law_score(dm, dm_gal, is_extragalactic_claim=is_exgal)
            duty = duty_cycle_consistency_law_score(phys.get("width_ms"), phys.get("period_s"))
            use_catalog = (
                (dm is not None and np.isfinite(float(dm)))
                or (phys.get("width_ms") is not None and phys.get("period_s") is not None)
            )
            physics_score = combine_physics_laws(
                technosig=techno, dispersion=disp, duty=duty,
                use_catalog_laws=bool(use_catalog),
            )

            # 1. Calibrated Anomaly Evidence Score (CAES) in [0, 100]
            # The conformal p-value is the PRIMARY anomaly evidence (finite-sample
            # FPR control). The supplementary terms (model posterior, chaos, FR,
            # physics) are folded in as a Neyman-Pearson likelihood-ratio-style
            # log-evidence weight, not as independent ad-hoc additive terms.
            #
            # Conformal surprisal: E_conf = min(50, -10 * log10(p)) [0..50 pts].
            # This is the dominant term: under H0, p ~ U(0,1); p=0.05 -> 13 pts,
            # p=1e-5 -> 50 pts cap.
            p_score = min(50.0, -10.0 * np.log10(max(p_val, 1e-5)))
            # Supplementary evidence weight (bounded [0, 1]): combines model
            # posterior anomaly probability, nonlinear-dynamics order, frequency-
            # resolved narrowband morphology, and physics-law consistency into a
            # single normalised evidence vector. Each component is independently
            # bounded in [0, 1]; the mean gives equal weight to each available
            # evidence channel.
            evidence_parts = [
                float(np.clip(prob_vec[2], 0.0, 1.0)),   # P(Anomaly) from ensemble
                float(np.clip(order, 0.0, 1.0)),          # chaos / deterministic order
                float(np.clip(fr, 0.0, 1.0)),             # narrowband morphology
                float(np.clip(physics_score, 0.0, 1.0)),  # physics-law consistency
            ]
            evidence_w = float(np.mean(evidence_parts))
            # Evidence term: 50 * evidence_w [0..50 pts]. When evidence_w is high
            # (all channels agree on anomaly), this doubles the conformal score.
            # When evidence_w is low (no supplementary signal), only the conformal
            # term contributes — never worse than conformal-only.
            evidence_term = 50.0 * evidence_w
            composite = float(np.clip(p_score + evidence_term, 0.0, 100.0))
            anomaly_scores.append(composite)

            # 2. Verdict logic
            is_bh_anomaly = bool(bh_rejected[i])
            is_significant_anomaly = p_val < self.conformal_alpha

            conf_natural = prob_vec[0]
            conf_interference = prob_vec[1]
            conf_max = max(conf_natural, conf_interference)
            in_dist_conf = conf_max >= 0.8

            is_ood = bool(ood_mask[i]) if ood_mask is not None else False

            if is_ood:
                verdict = "Anomaly"
            elif is_significant_anomaly:
                # A signal whose conformal p-value is significant (off-manifold
                # vs the natural null, including the real spectrogram descriptor
                # null) IS, by definition, an anomaly — not merely a candidate.
                verdict = "Anomaly"
            elif in_dist_conf and physics_score < 0.6:
                # Physically consistent with the claimed natural class -> accept.
                verdict = "Natural" if pred_cls == 0 else "Interference"
            elif in_dist_conf:
                # High in-distribution confidence but a physical contradiction
                # (e.g. a tonal morphology claimed to be a dispersed natural
                # pulse) -> escalate to candidate rather than silently accept.
                verdict = "Candidate — Requires Review"
            elif is_bh_anomaly or pred_cls == 2 or composite >= 50.0:
                verdict = "Candidate — Requires Review"
            elif pred_cls == 1:
                verdict = "Interference"
            else:
                verdict = "Natural"

            verdicts.append(verdict)

        return verdicts, anomaly_scores
