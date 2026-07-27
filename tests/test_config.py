"""Tests for axiom.config.PipelineConfig."""

import yaml

from axiom.config import DEFAULT_CONFIG, PipelineConfig


class TestPipelineConfig:
    def test_default_config_exists(self):
        assert isinstance(DEFAULT_CONFIG, dict)
        assert "data" in DEFAULT_CONFIG
        assert "models" in DEFAULT_CONFIG
        assert "stats" in DEFAULT_CONFIG

    def test_init_no_config_path(self):
        """Without a config path, should use defaults."""
        cfg = PipelineConfig(config_path="/nonexistent/path.yaml")
        assert cfg.get("stats.conformal_alpha") == 0.05

    def test_get_dot_path(self):
        cfg = PipelineConfig(config_path="/nonexistent/path.yaml")
        assert cfg.get("models.cnn.epochs") == 15
        assert cfg.get("data.train_ratio") == 0.8
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_get_nested_missing(self):
        cfg = PipelineConfig(config_path="/nonexistent/path.yaml")
        assert cfg.get("a.b.c.d.e", 42) == 42

    def test_yaml_override(self, tmp_path):
        """YAML overrides should deep-merge into defaults."""
        override = {"models": {"cnn": {"epochs": 99}}}
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(override))
        cfg = PipelineConfig(config_path=str(config_file))
        assert cfg.get("models.cnn.epochs") == 99
        # Other defaults should remain
        assert cfg.get("models.cnn.learning_rate") == 0.001

    def test_yaml_top_level_add(self, tmp_path):
        override = {"custom_key": "custom_value"}
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(override))
        cfg = PipelineConfig(config_path=str(config_file))
        assert cfg.get("custom_key") == "custom_value"

    def test_deterministic_defaults(self):
        c1 = PipelineConfig(config_path="/nonexistent")
        c2 = PipelineConfig(config_path="/nonexistent")
        assert c1.get("models.cnn.epochs") == c2.get("models.cnn.epochs")
