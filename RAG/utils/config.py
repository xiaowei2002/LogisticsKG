from pathlib import Path
from types import SimpleNamespace
from typing import Any
import yaml
from tenacity import retry_unless_exception_type

# 配置文件路径
CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.yaml'


class ConfigNode(dict):
    def __init__(self, data):
        super().__init__()
        data = data
        for key, value in data.items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"配置项不存在: {name}")

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"配置项不存在: {name}")

    def __repr__(self):
        return f"ConfigNode({dict(self.items())})"


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ConfigNode(data)


config = load_config()

