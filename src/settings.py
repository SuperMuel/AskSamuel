from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    portfolio_content_url: HttpUrl = Field(
        default=...,
        description="URL to download the portfolio content from",
    )

    langsmith_prompt_reference: str = Field(
        default=...,
        description="Reference name for the LangSmith prompt template",
    )

    default_llm_provider: str = Field(
        default="openai",
        description="Default LLM provider",
    )

    default_llm_model: str = Field(
        default="gpt-4.1-mini",
        description="Default LLM model",
    )


settings = Settings()
