from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    portfolio_path: Path = Field(default=..., description="Path to the portfolio file")

    @field_validator("portfolio_path")
    @classmethod
    def validate_portfolio_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Portfolio file does not exist at: {v}")
        return v


settings = Settings()
