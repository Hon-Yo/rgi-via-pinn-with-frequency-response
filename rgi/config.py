"""Typed settings shared by CLI, manifests, and checkpoints."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from .audio import AudioConfig
from .model import LossConfig, ModelConfig
from .physics import SimulatorConfig


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.001
    physics_weight: float = 0.0
    physics_warmup_epochs: int = 5
    physics_ramp_epochs: int = 5
    physics_batch_size: int = 2
    physics_every_steps: int = 1
    cpu_threads: int = 2
    threshold_min: float = 0.2
    threshold_max: float = 0.8
    threshold_count: int = 13

    def __post_init__(self):
        if not (self.epochs >= 1 and self.batch_size >= 1 and self.learning_rate > 0 and self.cpu_threads >= 1):
            raise ValueError("invalid training settings")
        if not (self.physics_weight >= 0 and self.physics_warmup_epochs >= 0 and self.physics_ramp_epochs >= 1
                and self.physics_batch_size >= 1 and self.physics_every_steps >= 1):
            raise ValueError("invalid physics schedule")
        if not (0 < self.threshold_min <= self.threshold_max < 1 and self.threshold_count >= 1):
            raise ValueError("invalid presence threshold grid")


@dataclass(frozen=True)
class SyntheticConfig:
    rooms: int = 12
    poses: int = 4
    seed: int = 101
    absorption_jitter: float = 0.0
    gain_jitter_db: float = 0.0
    coloration_jitter: float = 0.0  # bound on each outer tap of [a, 1, a]

    def __post_init__(self):
        if self.rooms < 1 or self.poses < 1:
            raise ValueError("synthetic generation needs rooms and poses")
        if not (0 <= self.absorption_jitter < 1 and self.gain_jitter_db >= 0 and 0 <= self.coloration_jitter < 0.5):
            raise ValueError("invalid synthetic augmentation settings")


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    @classmethod
    def from_dict(cls, obj):
        types = {"audio": AudioConfig, "model": ModelConfig, "loss": LossConfig,
                 "simulator": SimulatorConfig, "training": TrainingConfig, "synthetic": SyntheticConfig}
        if "conv_channels" in obj.get("model", {}):
            obj = {**obj, "model": {**obj["model"], "conv_channels": tuple(obj["model"]["conv_channels"])}}
        return cls(**{key: types[key](**value) for key, value in obj.items()})

    @classmethod
    def load(cls, path):
        return cls.from_dict(read_json(path)) if path else cls()

    def to_dict(self):
        return asdict(self)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
