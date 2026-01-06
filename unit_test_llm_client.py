#!/usr/bin/env python3
"""
Unit tests for LLM Client
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llm_client import generate_response


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response object"""
    mock_response = Mock()
    mock_response.output_text = "The Apollo 11 mission landed on the Moon on July 20, 1969. (Source: transcript_pao)"
    return mock_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client"""
    mock_client = Mock()
    mock_client.responses.create.return_value = mock_openai_response
    return mock_client


@pytest.fixture
def sample_context():
    """Sample context from RAG retrieval"""
    return """<context>
<document index="1">
<header>[Index: 1] Mission: Apollo 11 | Category: Public affairs | Source: transcript_pao</header>
<content>
The Eagle has landed. Neil Armstrong and Buzz Aldrin have successfully landed on the Moon.
</content>
</document>
</context>"""


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history"""
    return [
        {"role": "user", "content": "Tell me about NASA missions"},
        {"role": "assistant", "content": "NASA has conducted many missions including Apollo, Space Shuttle, and more."}
    ]


@pytest.fixture
def empty_conversation_history():
    """Empty conversation history for new conversations"""
    return []


# ============================================================================
# Tests: generate_response
# ============================================================================

class TestGenerateResponse:
    """Tests for generate_response function"""

    def test_generate_response_basic(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test basic response generation"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="When did Apollo 11 land on the Moon?",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0

    def test_generate_response_returns_output_text(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that response returns output_text from OpenAI"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="Test question",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            assert "Apollo 11" in response
            assert "1969" in response

    def test_generate_response_creates_client_with_key(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that OpenAI client is created with correct API key"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client) as mock_openai_class:
            generate_response(
                openai_key="my-secret-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            mock_openai_class.assert_called_once_with(api_key="my-secret-key")

    def test_generate_response_uses_default_model(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that default model is gpt-3.5-turbo"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            assert call_args.kwargs['model'] == "gpt-3.5-turbo"

    def test_generate_response_uses_custom_model(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that custom model can be specified"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history,
                model="gpt-4"
            )

            call_args = mock_openai_client.responses.create.call_args
            assert call_args.kwargs['model'] == "gpt-4"

    def test_generate_response_includes_context_in_prompt(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that context is included in the user prompt"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="What happened?",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            messages = call_args.kwargs['input']

            # Find the user message that contains the context
            user_messages = [m for m in messages if m.get('role') == 'user']
            assert len(user_messages) > 0

            # Check that context is in one of the user messages
            last_user_message = user_messages[-1]['content']
            assert "Eagle has landed" in last_user_message or "context" in last_user_message.lower()

    def test_generate_response_includes_question_in_prompt(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that user question is included in the prompt"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="When did they land?",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            messages = call_args.kwargs['input']

            # Find the last user message
            user_messages = [m for m in messages if m.get('role') == 'user']
            last_user_message = user_messages[-1]['content']

            assert "When did they land?" in last_user_message

    def test_generate_response_includes_conversation_history(self, mock_openai_client, sample_context, sample_conversation_history):
        """Test that conversation history is included"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="New question",
                context=sample_context,
                conversation_history=sample_conversation_history.copy()
            )

            call_args = mock_openai_client.responses.create.call_args
            messages = call_args.kwargs['input']

            # Should have history plus new message
            assert len(messages) >= 3  # 2 from history + 1 new

    def test_generate_response_has_system_instructions(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that system instructions are provided"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            instructions = call_args.kwargs['instructions']

            assert "NASA" in instructions
            assert len(instructions) > 0

    def test_generate_response_sets_temperature(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that temperature is set to low value for consistency"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            temperature = call_args.kwargs['temperature']

            assert temperature == 0.1

    def test_generate_response_sets_max_tokens(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that max_output_tokens is set"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            max_tokens = call_args.kwargs['max_output_tokens']

            assert max_tokens == 200


# ============================================================================
# Tests: Prompt Construction
# ============================================================================

class TestPromptConstruction:
    """Tests for prompt construction in generate_response"""

    def test_prompt_contains_output_format(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that prompt specifies output format"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test question",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            messages = call_args.kwargs['input']

            user_messages = [m for m in messages if m.get('role') == 'user']
            last_user_message = user_messages[-1]['content']

            assert "Output Format" in last_user_message or "Answer:" in last_user_message

    def test_prompt_requests_source_citation(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that prompt requests source citation"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test question",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            messages = call_args.kwargs['input']

            user_messages = [m for m in messages if m.get('role') == 'user']
            last_user_message = user_messages[-1]['content']

            assert "source" in last_user_message.lower()

    def test_system_prompt_mentions_rag(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test that system prompt mentions RAG context"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            generate_response(
                openai_key="test-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            call_args = mock_openai_client.responses.create.call_args
            instructions = call_args.kwargs['instructions']

            assert "RAG" in instructions or "context" in instructions.lower()


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases"""

    def test_empty_context(self, mock_openai_client, empty_conversation_history):
        """Test with empty context"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="What is NASA?",
                context="",
                conversation_history=empty_conversation_history
            )

            assert response is not None

    def test_empty_user_message(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test with empty user message"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            assert response is not None

    def test_long_context(self, mock_openai_client, empty_conversation_history):
        """Test with very long context"""
        long_context = "Document content. " * 1000

        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="Summarize this",
                context=long_context,
                conversation_history=empty_conversation_history
            )

            assert response is not None

    def test_special_characters_in_question(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test with special characters in question"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="What about <special> & 'chars'?",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            assert response is not None

    def test_unicode_in_question(self, mock_openai_client, sample_context, empty_conversation_history):
        """Test with unicode characters in question"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="Tell me about the 🚀 mission",
                context=sample_context,
                conversation_history=empty_conversation_history
            )

            assert response is not None

    def test_long_conversation_history(self, mock_openai_client, sample_context):
        """Test with long conversation history"""
        long_history = []
        for i in range(20):
            long_history.append({"role": "user", "content": f"Question {i}"})
            long_history.append({"role": "assistant", "content": f"Answer {i}"})

        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="Final question",
                context=sample_context,
                conversation_history=long_history
            )

            assert response is not None


# ============================================================================
# Tests: API Error Handling
# ============================================================================

class TestAPIErrorHandling:
    """Tests for API error scenarios"""

    def test_api_error_propagates(self, sample_context, empty_conversation_history):
        """Test that API errors propagate to caller"""
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("API Error")

        with patch('llm_client.OpenAI', return_value=mock_client):
            with pytest.raises(Exception) as exc_info:
                generate_response(
                    openai_key="test-key",
                    user_message="Test",
                    context=sample_context,
                    conversation_history=empty_conversation_history
                )

            assert "API Error" in str(exc_info.value)

    def test_invalid_api_key_error(self, sample_context, empty_conversation_history):
        """Test handling of invalid API key"""
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("Invalid API key")

        with patch('llm_client.OpenAI', return_value=mock_client):
            with pytest.raises(Exception) as exc_info:
                generate_response(
                    openai_key="invalid-key",
                    user_message="Test",
                    context=sample_context,
                    conversation_history=empty_conversation_history
                )

            assert "Invalid API key" in str(exc_info.value)

    def test_rate_limit_error(self, sample_context, empty_conversation_history):
        """Test handling of rate limit errors"""
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("Rate limit exceeded")

        with patch('llm_client.OpenAI', return_value=mock_client):
            with pytest.raises(Exception) as exc_info:
                generate_response(
                    openai_key="test-key",
                    user_message="Test",
                    context=sample_context,
                    conversation_history=empty_conversation_history
                )

            assert "Rate limit" in str(exc_info.value)


# ============================================================================
# Tests: Different Models
# ============================================================================

class TestDifferentModels:
    """Tests for different model configurations"""

    @pytest.mark.parametrize("model", [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ])
    def test_various_models(self, mock_openai_client, sample_context, empty_conversation_history, model):
        """Test that various models can be specified"""
        with patch('llm_client.OpenAI', return_value=mock_openai_client):
            response = generate_response(
                openai_key="test-key",
                user_message="Test",
                context=sample_context,
                conversation_history=empty_conversation_history,
                model=model
            )

            call_args = mock_openai_client.responses.create.call_args
            assert call_args.kwargs['model'] == model
            assert response is not None


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
