import re


def hide_documents_from_user_message(message: str) -> str:
    """
    Remove all <user-uploaded-document> tags and their content from a message.

    Args:
        message: The input message that may contain document tags

    Returns:
        The message with all document tags and their content removed
    """
    # Pattern to match <user-uploaded-document filename="...">content</user-uploaded-document>
    # Using re.DOTALL to make . match newlines as well
    pattern = r"<user-uploaded-document[^>]*>.*?</user-uploaded-document>\s*"

    # Remove all matches and any trailing whitespace/newlines
    cleaned_message = re.sub(pattern, "", message, flags=re.DOTALL)

    # Clean up any extra newlines that might be left behind
    cleaned_message = re.sub(r"\n{3,}", "\n\n", cleaned_message)

    return cleaned_message.strip()


def truncate_display_filename(filename: str, max_length: int = 30) -> str:
    """
    Truncate a filename to a maximum length and add ellipsis if it exceeds the length.
    """
    if len(filename) <= max_length:
        return filename
    return filename[:max_length] + "..."
