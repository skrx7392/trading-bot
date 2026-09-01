from pathlib import Path

from tbot import config


def test_data_root_defaults_to_repo_data(monkeypatch):
    monkeypatch.delenv("TBOT_DATA", raising=False)
    assert config.data_root() == config.REPO_ROOT / "data"


def test_repo_root_is_the_repo_not_src():
    # src/tbot/config.py -> repo root must contain pyproject.toml and src/
    assert (config.REPO_ROOT / "pyproject.toml").is_file()
    assert (config.REPO_ROOT / "src" / "tbot").is_dir()


def test_data_root_honours_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert config.data_root() == Path(tmp_path)


def test_data_root_is_read_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path / "a"))
    first = config.data_root()
    monkeypatch.setenv("TBOT_DATA", str(tmp_path / "b"))
    assert config.data_root() != first


def test_blank_env_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TBOT_DATA", "   ")
    assert config.data_root() == config.REPO_ROOT / "data"


def test_tax_rates():
    assert config.TAX_RATE_ST == 0.35
    assert config.TAX_RATE_LT == 0.15
