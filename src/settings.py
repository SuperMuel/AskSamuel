from typing import Self

from pydantic import Field, HttpUrl, SecretStr, model_validator
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

    @model_validator(mode="after")
    def validate_models_can_be_instantiated(self) -> Self:
        """Validate that allowed models can be instantiated according to allow_model_selection."""
        from dotenv import load_dotenv
        from langchain.chat_models import init_chat_model

        load_dotenv()

        if not self.allow_model_selection:
            # Only test the default model
            model_string = f"{self.default_llm_provider}/{self.default_llm_model}"
            if "/" not in model_string:
                raise ValueError(
                    f"Default model string '{model_string}' must be in format 'provider/model_name'"
                )
            provider, model_name = model_string.split("/", 1)
            try:
                init_chat_model(
                    model=model_name,
                    model_provider=provider,
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot instantiate default model '{model_string}': {e}"
                ) from e
            return self

        # If allow_model_selection is True, test all models in allowed_models
        for model_string in self.allowed_models:
            if "/" not in model_string:
                raise ValueError(
                    f"Model string '{model_string}' must be in format 'provider/model_name'"
                )

            provider, model_name = model_string.split("/", 1)

            try:
                init_chat_model(
                    model=model_name,
                    model_provider=provider,
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot instantiate model '{model_string}': {e}"
                ) from e

        return self

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

    def get_mistral_api_key_or_raise(self) -> str:
        mistral_key = self.mistral_api_key
        if not mistral_key or not mistral_key.get_secret_value().strip():
            raise ValueError("Mistral API key is not set or is empty")

        return mistral_key.get_secret_value()

    # --- File Upload & OCR Settings ---
    enable_file_upload: bool = Field(
        default=True,
        description="Enable file uploads in the chat input.",
    )
    max_file_size_mb: int = Field(
        default=5,
        description="Maximum size for a single uploaded file in MB.",
    )
    max_files_per_session: int = Field(
        default=5,
        description="Maximum number of files that can be uploaded in a single session.",
    )
    max_pages_per_file: int = Field(
        default=10,
        description="Maximum number of pages to process in a single document.",
    )
    allowed_file_types: list[str] = Field(
        default=[
            "pdf",
            "docx",
            "doc",
            "txt",
            "md",
        ],  # TODO: we should not use mistral-ocr for txt and md
        description="Allowed file extensions for upload.",
    )

    logging_level: str = Field(
        default="DEBUG",
        description="Logging level for the application. E.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'",
    )

    @model_validator(mode="after")
    def check_file_upload_and_mistral_key(self) -> Self:
        if self.enable_voice_input or self.enable_file_upload:
            mistral_key = self.mistral_api_key
            key_missing = (
                mistral_key is None or not mistral_key.get_secret_value().strip()
            )
            if key_missing:
                raise ValueError(
                    "Mistral API key must be set if voice input or file upload is enabled."
                )
        return self


settings = Settings()
