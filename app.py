import logging
import uuid
from typing import Self

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
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


@st.cache_data
def load_portfolio() -> str:
    response = requests.get(str(settings.portfolio_content_url), timeout=10)
    response.raise_for_status()
    return response.text


@st.cache_data
def load_system_prompt(portfolio_content: str) -> str:
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

        return str(formatted_prompt)
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

    try:
        name = sender.name
        email = sender.email
        company = sender.company
        # For MVP, simulate sending by logging to console (replace with real email/telegram in production)
        logger.info(
            f"Contact request received - Name: {name}, Email: {email}, Company: {company}, Subject: {subject}, Content: {content}"
        )

        # Simulate success (for demo; in real, handle errors)
        return "Success: Message sent to Samuel. He will respond soon."
    except Exception as e:
        logger.error(f"Error in contact function: {e}")
        return "Error: Could not send message."


@st.cache_resource
def get_memory() -> MemorySaver:
    return MemorySaver()


llm = init_chat_model(
    model=settings.default_llm_model,
    model_provider=settings.default_llm_provider,
)
memory = get_memory()
tools = [contact]
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

# Streamlit app
st.set_page_config(page_title="Samuel's AI Portfolio Chatbot", layout="wide")
st.title("Welcome to Samuel's AI Portfolio Chatbot")

# Generate and store thread_id for this conversation session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    logger.info(f"Generated thread_id: {st.session_state.thread_id}")

config = RunnableConfig(configurable={"thread_id": st.session_state.thread_id})

state = graph.get_state(config)

# Display chat history
with st.chat_message("assistant"):
    welcome_msg = "Hello! I'm Samuel's AI portfolio assistant. Ask me about his projects, skills, or how to contact him. Use the starters above for ideas."
    st.markdown(welcome_msg)
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
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)


if user_input := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    # Create placeholder for streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Stream tokens from LangGraph
            for token, _ in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="messages",
            ):
                # Process AIMessageChunk tokens safely
                token_content = getattr(token, "content", None)
                if token_content:
                    full_response += token_content
                    # Update the placeholder with accumulated response and cursor
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during streaming: {e}")
            st.error(
                "🌐 Network error. Please check your internet connection and try again."
            )
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}")
            st.error("😔 Something went wrong. Please try again or refresh the page.")

    st.rerun()
