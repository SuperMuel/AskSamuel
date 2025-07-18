from typing import Self

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
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
        default="google_genai",
        description="Default LLM provider",
    )

    default_llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Default LLM model",
    )

    allow_model_selection: bool = Field(
        default=False,
        description="Allow users to select from available models",
    )

    allowed_models: list[str] = Field(
        default=[
            "google_genai/gemini-2.5-flash",
            "openai/gpt-4o-mini",
            "openai/gpt-4.1-mini",
        ],
        description="List of allowed models in the format provider/model_name",
    )

    @field_validator("allowed_models")
    @classmethod
    def validate_models_can_be_instantiated(cls, v: list[str]) -> list[str]:
        """Validate that all models in allowed_models can be instantiated correctly."""
        from langchain.chat_models import init_chat_model

        for model_string in v:
            if "/" not in model_string:
                raise ValueError(
                    f"Model string '{model_string}' must be in format 'provider/model_name'"
                )

            provider, model_name = model_string.split("/", 1)

            try:
                # Try to instantiate the model to validate it works
                init_chat_model(
                    model=model_name,
                    model_provider=provider,
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot instantiate model '{model_string}': {e}"
                ) from e

        return v

    telegram_bot_token: str = Field(
        default=...,
        description="Telegram bot token for sending notifications",
    )

    telegram_chat_id: str = Field(
        default=...,
        description="Telegram chat ID to send notifications to",
    )

    welcome_message: str = Field(
        default="Hello! I'm Samuel's AI portfolio assistant. Ask me about his projects, skills, or how to contact him.",
        description="Welcome message displayed to users",
    )

    starter_questions: list[str] = Field(
        default=[
            "Let's talk about his experience.",
            "What are his skills?",
            "Can you contact him for me ?",
            "What's his master thesis about ?",
        ],
        description="Predefined questions to help users start conversations",
    )

    enable_voice_input: bool = Field(
        default=False,
        description="Enable voice input for users",
    )

    mistral_api_key: SecretStr | None = Field(
        default=...,
        description="Mistral API key for using Mistral models",
    )

    @model_validator(mode="after")
    def check_voice_input_and_mistral_key(self) -> Self:
        if self.enable_voice_input:
            mistral_key = self.mistral_api_key
            key_missing = (
                mistral_key is None or not mistral_key.get_secret_value().strip()
            )
            if key_missing:
                raise ValueError(
                    "Mistral API key must be set if voice input is enabled."
                )
        return self


settings = Settings()
