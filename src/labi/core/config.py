from pathlib import Path


class Config:
    def __init__(self, config_path=None):
        self._data = {}
        if config_path:
            self._load(config_path)

    def _load(self, config_path):
        path = Path(config_path)
        if not path.exists():
            return
        try:
            import yaml
            with open(path, "r") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                self._data = loaded
        except Exception:
            # Bad/missing config shouldn't crash the runtime — fall back
            # to defaults elsewhere via .get(key, default).
            self._data = {}

    def get(self, key, default=None):
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
