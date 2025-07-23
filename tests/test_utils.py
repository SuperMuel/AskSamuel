from src.utils import hide_documents_from_user_message


class TestHideDocumentsFromUserMessage:
    """Test cases for the hide_documents_from_user_message function."""

    def test_message_without_documents(self) -> None:
        """Test that messages without documents are returned unchanged."""
        message = "Hello, this is a simple message without any documents."
        result = hide_documents_from_user_message(message)
        assert result == message

    def test_message_with_single_document(self) -> None:
        """Test removing a single document from a message."""
        message = """Hello, here is my question:

<user-uploaded-document filename="test.txt">
This is the content of the document.
It has multiple lines.
</user-uploaded-document>

Please analyze this document."""

        expected = """Hello, here is my question:

Please analyze this document."""

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_message_with_multiple_documents(self) -> None:
        """Test removing multiple documents from a message."""
        message = """Please compare these documents:

<user-uploaded-document filename="doc1.txt">
Content of first document.
</user-uploaded-document>

Some text between documents.

<user-uploaded-document filename="doc2.pdf">
Content of second document.
With multiple lines.
</user-uploaded-document>

What are the differences?"""

        expected = """Please compare these documents:

Some text between documents.

What are the differences?"""

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_document_with_complex_filename(self) -> None:
        """Test removing documents with complex filenames."""
        message = """<user-uploaded-document filename="my-complex_file name (1).txt">
Document content here.
</user-uploaded-document>

Analysis please."""

        expected = "Analysis please."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_document_with_nested_content(self) -> None:
        """Test removing documents that contain XML-like content."""
        message = """<user-uploaded-document filename="config.xml">
<configuration>
    <setting name="value">test</setting>
</configuration>
</user-uploaded-document>

Please review this config."""

        expected = "Please review this config."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_document_with_special_characters(self) -> None:
        """Test removing documents with special characters in content."""
        message = """<user-uploaded-document filename="special.txt">
Content with special chars: !@#$%^&*()
And some unicode: 中文, émojis 🎉
</user-uploaded-document>

Handle special characters."""

        expected = "Handle special characters."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_empty_document(self) -> None:
        """Test removing empty documents."""
        message = """<user-uploaded-document filename="empty.txt">
</user-uploaded-document>

This document was empty."""

        expected = "This document was empty."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_document_with_only_newlines(self) -> None:
        """Test removing documents that contain only whitespace."""
        message = """<user-uploaded-document filename="whitespace.txt">


</user-uploaded-document>

Just whitespace above."""

        expected = "Just whitespace above."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_multiple_documents_consecutive(self) -> None:
        """Test removing multiple consecutive documents."""
        message = """<user-uploaded-document filename="first.txt">
First document content.
</user-uploaded-document>

<user-uploaded-document filename="second.txt">
Second document content.
</user-uploaded-document>

<user-uploaded-document filename="third.txt">
Third document content.
</user-uploaded-document>

All documents above should be removed."""

        expected = "All documents above should be removed."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_document_at_beginning_of_message(self) -> None:
        """Test removing document at the start of the message."""
        message = """<user-uploaded-document filename="start.txt">
Document at the beginning.
</user-uploaded-document>

This text should remain."""

        expected = "This text should remain."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_document_at_end_of_message(self) -> None:
        """Test removing document at the end of the message."""
        message = """This text should remain.

<user-uploaded-document filename="end.txt">
Document at the end.
</user-uploaded-document>"""

        expected = "This text should remain."

        result = hide_documents_from_user_message(message)
        assert result == expected

    def test_empty_message(self) -> None:
        """Test handling empty message."""
        message = ""
        result = hide_documents_from_user_message(message)
        assert result == ""

    def test_message_only_documents(self) -> None:
        """Test message containing only documents."""
        message = """<user-uploaded-document filename="only.txt">
Only document content.
</user-uploaded-document>"""

        result = hide_documents_from_user_message(message)
        assert result == ""
