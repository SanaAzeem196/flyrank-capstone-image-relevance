from pydantic_settings import BaseSettings
from dataclasses import dataclass

@dataclass
class GuardConfig:
    conf_min: float = 0.70
    sim_min: float = 0.60
    max_calls_per_run: int = 300
    max_est_usd: float = 1.00

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
    
    # Guard
    guard_config: GuardConfig = GuardConfig()
    
    # Tenancy
    default_tenant_id: str = "demo"
    
    class Config:
        env_file = ".env"

settings = Settings()

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