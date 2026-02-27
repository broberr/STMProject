from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Config(raw)

    def __getitem__(self, item):
        return self.raw[item]
