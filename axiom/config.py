import logging
import os

log = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

DEFAULT_CONFIG = {
    "data": {
        "default_limit": 10000,
        "split_seed": 42,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "cache_path": "axiom/data/universe_cache.json"
    },
    "dsp": {
        "waveform_length": 256,
        "sample_rate_hz": 1000.0,
        "noise_amplitude": 0.1,
        "pulse_period_range": [1.0, 10.0],
        "pulse_width_fraction": 0.05,
        "rfi_channels": [50, 120, 200]
    },
    "models": {
        "random_seed": 42,
        "cnn": {
            "epochs": 15,
            "batch_size": 64,
            "learning_rate": 0.001,
            "filters": [16, 32, 64],
            "dropout": 0.3
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 5
        },
        "density": {
            "flow_layers": 4,
            "hidden_dim": 128,
            "learning_rate": 0.005,
            "epochs": 10
        }
    },
    "stats": {
        "conformal_alpha": 0.05,
        "lyapunov_embedding_dim": 5,
        "lyapunov_delay": 2,
        "bh_fdr_alpha": 0.05
    },
    "physics": {
        # Weight of the physics-law term in the arbitrator's composite anomaly
        # score (bounded contribution, kept below the p-value ceiling so physics
        # refines rather than dominates the verdict).
        "arbitrator_weight": 12.0,
        # Fold catalog DM / duty-cycle consistency laws into the physics score
        # (only where a measured DM / width / period is available).
        "use_catalog_laws": True,
    }
}

class PipelineConfig:
    def __init__(self, config_path=None):
        if config_path is None:
            # Try to locate configs/pipeline_config.yaml relative to package root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "configs", "pipeline_config.yaml")
        
        self.config = DEFAULT_CONFIG.copy()
        if os.path.exists(config_path):
            if not YAML_AVAILABLE:
                log.warning("PyYAML is not installed — using default config. Install with: pip install pyyaml")
            else:
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        user_config = yaml.safe_load(f)
                        if user_config:
                            # Deep merge configs
                            self._deep_merge(self.config, user_config)
                except Exception as e:
                    log.warning("Failed to load config from %s: %s — using defaults", config_path, e)
        else:
            log.info("Config path %s not found — using defaults", config_path)

    def _deep_merge(self, base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def get(self, key_path, default=None):
        """Get configuration parameter via a dot-separated string path (e.g. 'models.cnn.epochs')"""
        keys = key_path.split(".")
        val = self.config
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val
