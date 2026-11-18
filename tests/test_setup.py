import yaml


def test_config_is_synthetic():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["project"]["data_source"] == "synthetic"