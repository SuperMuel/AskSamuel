import logging
import operator
from typing import Annotated, Self, TypedDict

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
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


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


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


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Create react agent with memory
tools = [contact]
agent_executor = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)

# Streamlit app
st.set_page_config(page_title="Samuel's AI Portfolio Chatbot", layout="wide")
st.title("Welcome to Samuel's AI Portfolio Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial welcome message
    welcome_msg = "Hello! I'm Samuel's AI portfolio assistant. Ask me about his projects, skills, or how to contact him. Use the starters above for ideas."
    st.session_state.messages.append(AIMessage(content=welcome_msg))


# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage) and not msg.tool_calls:
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)


user_input = st.chat_input("Type your message here...") or st.session_state.get(
    "user_input", None
)
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"messages": st.session_state.messages})
        st.session_state.messages = response["messages"]
        ai_response = st.session_state.messages[-1]

    if ai_response.content:
        with st.chat_message("assistant"):
            st.markdown(ai_response.content)

    # Clear temp input
    if "user_input" in st.session_state:
        del st.session_state.user_input
