import logging
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from textwrap import dedent
from typing import Literal

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import StateSnapshot
from langsmith import Client
from langsmith.schemas import Attachment
from pydantic import BaseModel, Field, field_validator

from src.audio import transcribe_audio_with_mistral
from src.ocr import uncached_ocr_document_with_mistral
from src.settings import settings
from src.utils import hide_documents_from_user_message, truncate_display_filename

load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("App started. Setting Streamlit page config.")
st.set_page_config(page_title="Samuel's AI Portfolio Chatbot", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    logger.info(f"Generated thread_id: {st.session_state.thread_id}")

if "msg_id_to_trace_id" not in st.session_state:
    st.session_state["msg_id_to_trace_id"] = {}
    logger.debug("Initialized msg_id_to_trace_id in session state.")

if "msg_id_to_documents" not in st.session_state:
    st.session_state["msg_id_to_documents"] = {}
    logger.debug("Initialized msg_id_to_documents in session state.")

if "used_starters" not in st.session_state:
    st.session_state["used_starters"] = set()
    logger.debug("Initialized used_starters in session state.")

# Session state to hold processed file content
if "processed_files_content" not in st.session_state:
    st.session_state.processed_files_content = []
    logger.debug("Initialized processed_files_content in session state.")

if "file_sha256_to_markdown" not in st.session_state:
    st.session_state["file_sha256_to_markdown"] = {}
    logger.debug("Initialized file_sha256_to_markdown in session state.")


@st.cache_data(show_spinner="Loading portfolio...")
def load_portfolio() -> str:
    logger.info("Loading portfolio content from remote URL.")
    with httpx.Client() as client:
        response = client.get(str(settings.portfolio_content_url), timeout=10)
        logger.info(f"Portfolio content fetched. Status code: {response.status_code}")
        response.raise_for_status()
        return response.text


@st.cache_data(show_spinner="Loading portfolio...")
def load_system_prompt(portfolio_content: str) -> SystemMessage:
    """Load system prompt from LangSmith and format it with portfolio content."""
    logger.info("Loading system prompt from LangSmith.")
    try:
        client = Client()
        logger.debug(f"Pulling prompt: {settings.langsmith_prompt_reference}")
        prompt_template = client.pull_prompt(
            settings.langsmith_prompt_reference,
        )

        # Format the prompt with the portfolio content
        logger.debug("Formatting system prompt with portfolio content and date.")
        formatted_prompt = prompt_template.invoke(
            {
                "portfolio_content": portfolio_content,
                "formatted_date": datetime.now(tz=UTC).date().strftime("%B %d, %Y"),
            }
        )

        # Extract just the system message
        messages = formatted_prompt.to_messages()
        logger.debug(f"System prompt messages extracted: {messages}")
        assert len(messages) == 1
        system_message = messages[0]
        assert isinstance(system_message, SystemMessage)

        logger.info("System prompt loaded successfully.")
        return system_message
    except Exception as e:
        logger.error(f"Error loading prompt from LangSmith: {e}")
        raise e


@st.cache_data(show_spinner="Processing uploaded documents...")
def cached_ocr_document_with_mistral(file_content: bytes, filename: str) -> str:
    logger.info(f"Caching OCR for {filename}")
    return uncached_ocr_document_with_mistral(file_content, filename)


logger.info("Calling load_portfolio()")
portfolio_content: str = load_portfolio()

# Load system prompt from LangSmith
logger.info("Calling load_system_prompt()")
system_prompt = load_system_prompt(portfolio_content)


class Sender(BaseModel):
    """Sender of the message. Mandatory so Samuel can contact them back."""

    name: str = Field(
        ..., description="Name of the sender. Can be inferred from the email address."
    )
    email: str = Field(
        ...,
        description="Email of the sender. Should never be inferred from the context. Always ask for it.",
    )
    company: str | None = Field(
        default=None,
        description="Company of the sender. Can be inferred from the email address or the conversation context.",
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        import re

        # Simple but robust email regex
        email_regex = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        if not re.match(email_regex, value):
            raise ValueError("Invalid email address format.")
        return value


def send_telegram_notification(sender: Sender, subject: str, content: str) -> None:
    """Send a notification to Telegram about a new contact inquiry using HTTP API.

    Args:
        sender: Sender of the message
        subject: Subject of the message
        content: Content of the message
    """
    logger.info(
        "Sending Telegram notification.",
        extra={"sender": sender, "subject": subject, "content": content},
    )

    # Format the message
    message = dedent(f"""\
        🔔 New Contact Inquiry
        👤 **Name:** {sender.name or "Not provided"}
        📧 **Email:** {sender.email or "Not provided"}
        🏢 **Company:** {sender.company or "Not provided"}
        📋 **Subject:** {subject}
        💬 **Message:**\n{content}
        """)

    # Telegram Bot API endpoint
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"

    # Prepare the payload
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }

    logger.debug(f"Sending POST to Telegram API: {url} with payload: {payload}")
    # Send the HTTP request
    with httpx.Client() as client:
        response = client.post(url, json=payload, timeout=10)
        logger.info(f"Telegram API response status: {response.status_code}")
        response.raise_for_status()

    logger.info("Telegram notification sent successfully")


@tool
def contact(sender: Sender, subject: str, content: str) -> str:
    """Contact Samuel on behalf of the user.

    Args:
        sender: Sender of the message
        subject: Subject of the message. Infer it from the content of the message, don't ask for it.
        content: Content of the message
    """

    logger.debug(
        f"contact() called with sender={sender}, subject={subject}, content={content}"
    )
    if not subject:
        subject = "[No subject]"
        logger.debug("No subject provided, using default '[No subject]'")
    if not content:
        logger.warning("No content provided to contact().")
        return "Error: No content provided."

    name = sender.name
    email = sender.email
    company = sender.company

    try:
        # Log the contact request
        logger.info(
            f"Contact request received - Name: {name}, Email: {email}, Company: {company}, Subject: {subject}, Content: {content}"
        )

        # Send Telegram notification
        send_telegram_notification(sender, subject, content)

        logger.info("Contact function completed successfully.")
        return "Success: Message sent."

    except Exception as e:
        logger.error(f"Error in contact function: {e}")
        return "Error: Could not send message."


tools = [contact]
logger.info("Tools initialized.")


@st.cache_resource
def get_memory() -> MemorySaver:
    logger.info("Initializing MemorySaver for LangGraph checkpointing.")
    return MemorySaver()


memory = get_memory()


@st.cache_resource
def get_model(model_string: str) -> BaseChatModel:
    assert "/" in model_string

    provider, model_name = model_string.split("/", 1)
    logger.debug(f"Model provider: {provider}, model name: {model_name}")

    if provider == "google_genai":
        logger.info("Instantiating ChatGoogleGenerativeAI model.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            thinking_budget=4096,  # 0 = no reasoning, 1024 = light, 8192+ = heavy
            # include_thoughts=True,  # get thought summaries back
        )

    return init_chat_model(
        model=model_name,
        model_provider=provider,
    )


with st.sidebar:
    # Main introduction
    st.markdown("### 👋 Welcome!")

    st.markdown("""
    **This is Samuel Mallet's AI Portfolio Assistant** - an intelligent chatbot that knows everything about his experience, projects, and skills as an AI Engineer.

    🎯 **Currently seeking full-time AI Engineer roles from November 2025**
    """)

    # Links section
    st.markdown("### 🔗 Links")

    st.link_button(
        "Real Portfolio",
        "https://www.notion.so/supermuel/Samuel-Mallet-AI-Engineer-0cc25a4537884ed09e2c24e06af0bffe",
        help="View his complete Notion portfolio",
        use_container_width=True,
        icon=":material/book:",
    )
    st.link_button(
        "GitHub Profile",
        "https://github.com/supermuel",
        help="Browse his code repositories",
        use_container_width=True,
        icon=":material/code:",
    )

    st.divider()


selected_model = (
    f"{settings.default_llm_provider}/{settings.default_llm_model}"
    if not settings.allow_model_selection
    else st.sidebar.selectbox(
        "Choose a model:",
        options=settings.allowed_models,
        help="Select the AI model to use for responses",
        format_func=lambda x: x.split("/")[1],
    )
)


llm = get_model(selected_model)

st.title("💬 Talk to my Portfolio")


config = RunnableConfig(configurable={"thread_id": st.session_state.thread_id})

graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)


@st.dialog("Feedback Details")
def feedback_dialog(feedback: Literal[0, 1], message_id: str) -> None:
    """Dialog for collecting additional feedback details."""
    logger.debug(
        f"Opening feedback dialog for message_id={message_id}, feedback={feedback}"
    )
    st.write("Thank you for your feedback!")

    if feedback == 0:
        st.write("We'd appreciate more details to help us improve:")
    else:
        st.write("Any additional thoughts? (optional)")

    details = st.text_area(
        "Additional details:",
        placeholder="Tell us more about your experience..."
        if feedback == 0
        else "What did you like?",
    )

    if st.button("Submit", type="primary", use_container_width=True):
        logger.info(f"Feedback submit button pressed for message_id={message_id}")
        # Store feedback in session state
        if "feedback_log" not in st.session_state:
            logger.debug("Initializing feedback_log in session state.")
            st.session_state.feedback_log = []

        feedback_entry = {
            "message_id": message_id,
            "thread_id": st.session_state.thread_id,
            "type": feedback,
            "details": details,
            "timestamp": datetime.now(UTC),
        }

        logger.debug(f"Appending feedback_entry: {feedback_entry}")
        st.session_state.feedback_log.append(feedback_entry)

        # Log the feedback
        logger.info(
            f"Feedback received - Type: {feedback}, Message ID: {message_id}, Details: {details}"
        )

        try:
            client = Client()
            logger.debug("Creating feedback in LangSmith.")
            ls_result = client.create_feedback(
                key="thumb",
                score=feedback,
                trace_id=st.session_state["msg_id_to_trace_id"][message_id],
                comment=details or None,
            )
            logger.info(f"LangSmith feedback created: {ls_result}")
        except Exception as e:
            logger.error(f"Error creating feedback in LangSmith: {e}")

        st.session_state[f"feedback_{message_id}"] = feedback

        logger.debug(
            f"Set session_state feedback_{message_id} to {feedback}. Rerunning app."
        )
        st.rerun()


def handle_feedback(
    message_id: str,
) -> None:
    last_feedback = st.session_state.get(f"feedback_{message_id}")

    new_feedback = st.feedback(
        options="thumbs",
        key=f"feedback_{message_id}_buttons",
    )

    st.session_state[f"feedback_{message_id}"] = new_feedback

    if new_feedback is not None and new_feedback != last_feedback:
        feedback_dialog(new_feedback, message_id=message_id)


def render_message_history(state: StateSnapshot) -> None:
    logger.debug("Rendering message history.")

    with st.chat_message("assistant"):
        st.markdown(settings.welcome_message)

    if "messages" not in state.values:
        logger.debug("No messages in state.values.")
        return

    logger.debug(f"Rendering {len(state.values['messages'])} messages.")
    for msg in state.values["messages"]:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    with st.expander(
                        f"🔧 Tool call: {tool_call['name'].capitalize()}",
                        expanded=True,
                    ):
                        st.json(tool_call["args"])
            if msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

                    assert msg.id is not None
                    handle_feedback(message_id=msg.id)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                assert isinstance(msg.content, str)
                cleaned_content = hide_documents_from_user_message(msg.content)
                st.markdown(cleaned_content)
                if documents := st.session_state["msg_id_to_documents"].get(msg.id, []):
                    logger.debug(
                        f"Found {len(documents)} documents for message {msg.id}"
                    )
                    for document in documents:
                        st.badge(
                            truncate_display_filename(document["filename"]), icon="📄"
                        )
                else:
                    logger.debug(f"No documents found for message {msg.id}")
        elif isinstance(msg, ToolMessage):
            if msg.status == "error":
                with st.expander(f"🔧 Tool call failed: {msg.name}", expanded=False):
                    st.error(f"{msg.content}")
            elif msg.status == "success":
                with st.expander(f"🔧 Tool call succeeded: {msg.name}", expanded=False):
                    st.success(f"{msg.content}")
            else:
                with st.expander(f"🔧 Tool call: {msg.name}", expanded=False):
                    st.json(msg)
        else:
            logger.warning(f"Unexpected message type: {type(msg)}")
            continue


# Display chat history
state = graph.get_state(config)
render_message_history(state)


def show_starters(questions: list[str]) -> str | None:
    logger.debug(f"Showing starters: {questions}")
    if not questions:
        return

    questions = list(set(questions))

    assert len(questions) <= 4
    for col, question in zip(
        st.columns(
            len(questions),
            border=True,
        ),
        questions,
        strict=True,
    ):
        if col.button(
            question,
            use_container_width=True,
            disabled=question in st.session_state["used_starters"],
            type="tertiary",
        ):
            st.session_state["used_starters"].add(question)
            return question


selected_starter = show_starters(settings.starter_questions)

if settings.enable_voice_input and not selected_starter:
    with st.sidebar:
        logger.debug("Showing voice input.")
        if voice_message := st.audio_input(
            "You can also speak to me !", key="audio_input"
        ):
            logger.info("Voice message received")
            if (
                "last_audio_input" in st.session_state
                and voice_message == st.session_state["last_audio_input"]
            ):
                logger.info("Voice message is the same as the last one")
                # Do nothing. This is useful to avoid re-transcribing the same message on every rerun.
            else:
                st.session_state["last_audio_input"] = voice_message
                with st.spinner("Transcribing voice message..."):
                    ls_attachment = Attachment(
                        mime_type="audio/wav",
                        data=voice_message.getvalue(),
                    )
                    result = transcribe_audio_with_mistral(audio_file=ls_attachment)
                    st.session_state["chat_input"] = result
                    logger.info(f"Transcription result: {result}")
        st.divider()

#  Add reset chat button to sidebar
if st.sidebar.button(
    "🗑️ Reset Chat",
    help="Clear conversation history and start a new chat",
    use_container_width=True,
    disabled="messages" not in state.values or len(state.values["messages"]) == 0,
):
    # Clear session state
    if "thread_id" in st.session_state:
        del st.session_state.thread_id
    if "used_starters" in st.session_state:
        del st.session_state.used_starters
    if "msg_id_to_documents" in st.session_state:
        del st.session_state.msg_id_to_documents
    if "msg_id_to_trace_id" in st.session_state:
        del st.session_state.msg_id_to_trace_id
    if "feedback_log" in st.session_state:
        del st.session_state.feedback_log
    if "chat_input" in st.session_state:
        del st.session_state.chat_input
    if "last_audio_input" in st.session_state:
        # Don't delete this, or otherwise the previous
        # audio input will still be displayed
        pass

    logger.info("Chat reset.")

    st.rerun()


class UploadedFile(BaseModel):
    file_name: str
    size_bytes: int
    content: bytes
    type: str
    sha256: str


class UserChatInput(BaseModel):
    text: str | None = None
    files: list[UploadedFile] = Field(default_factory=list)

    def __repr__(self) -> str:
        files_info = (
            [
                f"{f.file_name} (sha256: {f.sha256}) ({f.size_bytes / 1024 / 1024:.2f} MB)"
                for f in self.files
            ]
            if self.files
            else None
        )
        return f"UserChatInput(text={self.text!r}, files={files_info})"

    def __str__(self) -> str:
        files_info = (
            [
                f"{f.file_name} (sha256: {f.sha256}) ({f.size_bytes / 1024 / 1024:.2f} MB)"
                for f in self.files
            ]
            if self.files
            else None
        )
        return f"UserChatInput(text={self.text!r}, files={files_info})"


def render_chat_input(*, force_return_text: str | None = None) -> UserChatInput | None:
    logger.debug(f"Rendering chat input. {force_return_text=}")
    if settings.enable_file_upload:
        prompt_data = st.chat_input(
            "Type your message or upload a file...",
            accept_file="multiple",
            file_type=settings.allowed_file_types,
            key="chat_input",
            disabled=bool(force_return_text),
        )
        if force_return_text:
            return UserChatInput(text=force_return_text)
        if prompt_data:
            return UserChatInput(
                text=prompt_data.get("text", None),
                files=[
                    UploadedFile(
                        file_name=file.name,
                        size_bytes=file.size,
                        content=file.getvalue(),
                        type=file.type,
                        sha256=sha256(file.getvalue()).hexdigest(),
                    )
                    for file in prompt_data.get("files", [])
                ],
            )
    else:
        text_only_input = st.chat_input(
            "Type your message here...",
            key="chat_input",
            disabled=bool(force_return_text),
        )
        if force_return_text:
            return UserChatInput(text=force_return_text, files=[])
        if text_only_input:
            return UserChatInput(text=text_only_input)


user_chat_input = render_chat_input(force_return_text=selected_starter)
logger.debug(f"user_chat_input: {user_chat_input}")

if user_chat_input and user_chat_input.files:
    logger.debug(
        f"User uploaded {len(user_chat_input.files)} files, processing them..."
    )
    uploaded_files = user_chat_input.files
    # TODO: validate number of files per session
    # TODO: validate file size
    pass

    for uploaded_file in uploaded_files:
        if uploaded_file.size_bytes > settings.max_file_size_mb * 1024 * 1024:
            st.error(
                f"File '{uploaded_file.file_name}' is too large (max {settings.max_file_size_mb}MB)."
            )
            st.stop()
        try:
            file_bytes = uploaded_file.content
            markdown_content = cached_ocr_document_with_mistral(
                file_content=file_bytes, filename=uploaded_file.file_name
            )
            if not markdown_content:
                st.error(f"File '{uploaded_file.file_name}' looks empty.")
                st.stop()

            st.session_state["file_sha256_to_markdown"][uploaded_file.sha256] = (
                markdown_content
            )

        except Exception as e:
            logger.error(f"Failed to process file {uploaded_file.file_name}: {e}")
            st.error(
                f"Sorry, there was an error processing '{uploaded_file.file_name}'."
            )
            st.stop()


if user_chat_input:
    trace_id = uuid.uuid4()

    config = RunnableConfig(
        run_id=trace_id,
        configurable={"thread_id": st.session_state.thread_id},
    )
    logger.debug(f"Config: {config}")

    logger.debug(f"Final chat input: '{user_chat_input}'")
    documents = []
    logger.debug(
        f"user_chat_input.files: {[file.file_name for file in user_chat_input.files]}"
    )
    for file in user_chat_input.files:
        if file.sha256 in st.session_state["file_sha256_to_markdown"]:
            document = {
                "filename": file.file_name,
                "text": st.session_state["file_sha256_to_markdown"][file.sha256],
            }
            documents.append(document)
        else:
            logger.warning(
                f"File {file.file_name} not found in st.session_state['file_sha256_to_markdown']"
            )
            continue
    if not user_chat_input.text and not documents:
        logger.warning("Empty input. Stopping.")
        st.stop()

    final_input = ""
    if documents:
        for document in documents:
            final_input += f'<user-uploaded-document filename="{document["filename"]}">\n{document["text"]}\n</user-uploaded-document>\n\n'
    if user_chat_input.text:
        final_input += "\n\n" + user_chat_input.text

    with st.chat_message("user"):
        cleaned_final_input = hide_documents_from_user_message(final_input)
        st.markdown(cleaned_final_input)
        if documents:
            for document in documents:
                st.badge(truncate_display_filename(document["filename"]), icon="📄")

    # Create placeholder for streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_message_id = None

        try:
            # Stream tokens from LangGraph
            logger.debug("Calling graph.stream()")
            human_message_id = str(uuid.uuid4())
            st.session_state["msg_id_to_documents"][human_message_id] = documents
            for token, _ in graph.stream(
                {"messages": [HumanMessage(content=final_input, id=human_message_id)]},
                config,
                stream_mode="messages",
            ):
                if isinstance(token, ToolMessage):
                    logger.info(f"Tool message: {token.content}")
                    continue
                if not isinstance(token, AIMessageChunk):
                    logger.info(f"Unexpected token type: {type(token)}")
                    continue

                assistant_message_id = token.id
                if assistant_message_id is not None and token.id:
                    st.session_state["msg_id_to_trace_id"][assistant_message_id] = (
                        trace_id
                    )

                # Process AIMessageChunk tokens safely
                token_content = getattr(token, "content", None)
                if token_content:
                    full_response += token_content
                    # Update the placeholder with accumulated response and cursor
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except httpx.RequestError as e:
            logger.error(f"Network error during streaming: {e}")
            st.error(
                "🌐 Network error. Please check your internet connection and try again."
            )
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}")
            st.error("😔 Something went wrong. Please try again or refresh the page.")

    st.rerun()
