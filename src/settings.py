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

    allow_model_selection: bool = Field(
        default=False,
        description="Allow users to select from available models",
    )

    allowed_models: list[str] = Field(
        default=[
            "openai/gpt-4o-mini",
            "openai/gpt-4.1-mini",
        ],
        description="List of allowed models in the format provider/model_name",
    )

    telegram_bot_token: str = Field(
        default=...,
        description="Telegram bot token for sending notifications",
    )

    telegram_chat_id: str = Field(
        default=...,
        description="Telegram chat ID to send notifications to",
    )


settings = Settings()
