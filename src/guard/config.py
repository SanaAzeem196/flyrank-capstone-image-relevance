from dataclasses import dataclass


@dataclass
class GuardConfig:
    conf_min: float = 0.70
    sim_min: float = 0.60
    max_calls_per_run: int = 300
    max_est_usd: float = 1.00
