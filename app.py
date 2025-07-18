import logging
import uuid
from datetime import UTC, datetime
from textwrap import dedent
from typing import Literal, Self

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
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langsmith import Client
from pydantic import BaseModel, Field, model_validator

from src.settings import settings

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Samuel's AI Portfolio Chatbot", layout="wide")


@st.cache_data
def load_portfolio() -> str:
    with httpx.Client() as client:
        response = client.get(str(settings.portfolio_content_url), timeout=10)
        response.raise_for_status()
        return response.text


@st.cache_data
def load_system_prompt(portfolio_content: str) -> SystemMessage:
    """Load system prompt from LangSmith and format it with portfolio content."""
    try:
        client = Client()
        prompt_template = client.pull_prompt(
            settings.langsmith_prompt_reference,
        )

        # Format the prompt with the portfolio content
        formatted_prompt = prompt_template.invoke(
            {"portfolio_content": portfolio_content}
        )

        # Extract just the system message
        messages = formatted_prompt.to_messages()
        assert len(messages) == 1
        system_message = messages[0]
        assert isinstance(system_message, SystemMessage)

        return system_message
    except Exception as e:
        logger.error(f"Error loading prompt from LangSmith: {e}")
        raise e


portfolio_content: str = load_portfolio()

# Load system prompt from LangSmith
system_prompt = load_system_prompt(portfolio_content)


class Sender(BaseModel):
    """Sender of the message. At least one of 'name' or 'email' must be provided."""

    name: str | None = Field(default=None, description="Name of the sender")
    email: str | None = Field(default=None, description="Email of the sender")
    company: str | None = Field(default=None, description="Company of the sender")

    @model_validator(mode="after")
    def name_or_email_required(self) -> Self:
        if not self.name and not self.email:
            raise ValueError("At least one of 'name' or 'email' must be provided.")
        return self


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

    # Send the HTTP request
    with httpx.Client() as client:
        response = client.post(url, json=payload, timeout=10)
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

    if not subject:
        subject = "[No subject]"
    if not content:
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

        return "Success: Message sent."

    except Exception as e:
        logger.error(f"Error in contact function: {e}")
        return "Error: Could not send message."


tools = [contact]


@st.cache_resource
def get_memory() -> MemorySaver:
    return MemorySaver()


memory = get_memory()


@st.cache_resource
def get_model(model_string: str) -> BaseChatModel:
    assert "/" in model_string

    provider, model_name = model_string.split("/", 1)

    if provider == "google_genai":
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


#
llm = get_model(selected_model)

st.title("Welcome to Samuel's AI Portfolio Chatbot")

# Generate and store thread_id for this conversation session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    logger.info(f"Generated thread_id: {st.session_state.thread_id}")

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
        # Store feedback in session state
        if "feedback_log" not in st.session_state:
            st.session_state.feedback_log = []

        feedback_entry = {
            "message_id": message_id,
            "thread_id": st.session_state.thread_id,
            "type": feedback,
            "details": details,
            "timestamp": datetime.now(UTC),
        }

        st.session_state.feedback_log.append(feedback_entry)

        # Log the feedback
        logger.info(
            f"Feedback received - Type: {feedback}, Message ID: {message_id}, Details: {details}"
        )

        client = Client()
        ls_result = client.create_feedback(
            key="thumb",
            score=feedback,
            trace_id=st.session_state["msg_id_to_trace_id"][message_id],
            comment=details or None,
        )
        logger.info(f"LangSmith feedback created: {ls_result}")

        st.session_state[f"feedback_{message_id}"] = feedback

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


state = graph.get_state(config)
# Display chat history
with st.chat_message("assistant"):
    st.markdown(settings.welcome_message)
if "messages" in state.values:
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
                st.markdown(msg.content)

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

    # Generate new thread_id
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.used_starters = set()

    logger.info(f"Chat reset - new thread_id: {st.session_state.thread_id}")
    st.rerun()


if "used_starters" not in st.session_state:
    st.session_state["used_starters"] = set()


def show_starters(questions: list[str]) -> str | None:
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

user_input = st.chat_input("Type your message here...") or selected_starter

if "msg_id_to_trace_id" not in st.session_state:
    st.session_state["msg_id_to_trace_id"] = {}

if user_input:
    trace_id = uuid.uuid4()

    config = RunnableConfig(
        run_id=trace_id,
        configurable={"thread_id": st.session_state.thread_id},
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Create placeholder for streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_message_id = None

        try:
            # Stream tokens from LangGraph
            for token, _ in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="messages",
            ):
                assert isinstance(token, AIMessageChunk)
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
