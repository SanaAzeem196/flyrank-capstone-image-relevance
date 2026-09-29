from pydantic_settings import BaseSettings

from src.guard.config import GuardConfig


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str
    vision_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768

    # Database
    database_url: str = "postgresql+psycopg://capstone:capstone@localhost:5432/capstone"

    # Worker
    gemini_rpm: int = 10
    job_max_attempts: int = 3

    # Guard (mirrors GuardConfig fields so they're settable from .env)
    conf_min: float = 0.70
    sim_min: float = 0.60
    max_calls_per_run: int = 300
    max_est_usd: float = 1.00

    # Tenancy
    default_tenant_id: str = "demo"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()

guard_config = GuardConfig(
    conf_min=settings.conf_min,
    sim_min=settings.sim_min,
    max_calls_per_run=settings.max_calls_per_run,
    max_est_usd=settings.max_est_usd,
)

# Pricing (for cost tracking)
PRICING = {
    "gemini-3.5-flash-lite": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000
    },
    "gemini-embedding-2": {
        "input": 0.02 / 1_000_000,
        "output": 0.0
    }
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {})
    return input_tokens * p.get("input", 0) + output_tokens * p.get("output", 0)
