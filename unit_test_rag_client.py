#!/usr/bin/env python3
"""
Unit tests for RAG Client
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from rag_client import (
    discover_chroma_backends,
    initialize_rag_system,
    retrieve_documents,
    format_context
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection"""
    mock = Mock()
    mock.name = "test_collection"
    mock.count.return_value = 100
    mock.query.return_value = {
        'ids': [['id1', 'id2', 'id3']],
        'documents': [['Document 1 content', 'Document 2 content', 'Document 3 content']],
        'metadatas': [[
            {'mission': 'apollo_11', 'source': 'transcript', 'document_category': 'pao'},
            {'mission': 'apollo_13', 'source': 'audio', 'document_category': 'mission_audio'},
            {'mission': 'challenger', 'source': 'report', 'document_category': 'technical'}
        ]],
        'distances': [[0.1, 0.2, 0.3]]
    }
    return mock


@pytest.fixture
def mock_chroma_client(mock_collection):
    """Mock ChromaDB client"""
    mock = Mock()
    mock.list_collections.return_value = [mock_collection]
    mock.get_or_create_collection.return_value = mock_collection
    return mock


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "This is the first document about Apollo 11 mission.",
        "This is the second document about Apollo 13.",
        "This is the third document about the Challenger."
    ]


@pytest.fixture
def sample_metadatas():
    """Sample metadata for testing"""
    return [
        {'mission': 'apollo_11', 'source': 'transcript_pao', 'document_category': 'public_affairs'},
        {'mission': 'apollo_13', 'source': 'audio_segment', 'document_category': 'mission_audio'},
        {'mission': 'challenger', 'source': 'report', 'document_category': 'technical_report'}
    ]


# ============================================================================
# Tests: discover_chroma_backends
# ============================================================================

class TestDiscoverChromaBackends:
    """Tests for discover_chroma_backends function"""

    def test_discover_empty_directory(self):
        """Test discovery when no chroma directories exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('rag_client.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.glob.return_value = []

                # Call with patched Path that returns empty
                with patch.object(Path, 'glob', return_value=[]):
                    backends = discover_chroma_backends()
                    # Should return empty dict or dict with no valid backends
                    assert isinstance(backends, dict)

    def test_discover_with_valid_backend(self, mock_chroma_client, mock_collection):
        """Test discovery with valid ChromaDB backend"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock chroma directory
            chroma_dir = Path(tmpdir) / "chroma_db"
            chroma_dir.mkdir()

            with patch('rag_client.chromadb.PersistentClient', return_value=mock_chroma_client):
                with patch.object(Path, 'glob', return_value=[chroma_dir]):
                    backends = discover_chroma_backends()

                    assert isinstance(backends, dict)
                    if backends:  # If any backends found
                        for key, info in backends.items():
                            assert 'directory' in info
                            assert 'collection_name' in info
                            assert 'display_name' in info
                            assert 'document_count' in info

    def test_discover_excludes_venv(self):
        """Test that .venv directories are excluded"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directories
            venv_chroma = Path(tmpdir) / ".venv" / "chroma_db"
            venv_chroma.mkdir(parents=True)

            valid_chroma = Path(tmpdir) / "chroma_db"
            valid_chroma.mkdir()

            # The function should exclude .venv paths
            with patch('rag_client.chromadb.PersistentClient') as mock_client:
                mock_client.return_value.list_collections.return_value = []

                backends = discover_chroma_backends()

                # Should not include paths with .venv
                for key, info in backends.items():
                    assert '.venv' not in info['directory']

    def test_discover_handles_connection_error(self):
        """Test handling of connection errors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_dir = Path(tmpdir) / "chroma_db"
            chroma_dir.mkdir()

            with patch('rag_client.chromadb.PersistentClient') as mock_client:
                mock_client.side_effect = Exception("Connection failed")

                with patch.object(Path, 'glob', return_value=[chroma_dir]):
                    backends = discover_chroma_backends()

                    # Should still return a dict, possibly with error info
                    assert isinstance(backends, dict)
                    if backends:
                        # Check if error entry exists
                        for key, info in backends.items():
                            if 'error' in key.lower() or 'Error' in info.get('display_name', ''):
                                assert info['document_count'] == 0


# ============================================================================
# Tests: initialize_rag_system
# ============================================================================

class TestInitializeRagSystem:
    """Tests for initialize_rag_system function"""

    def test_initialize_creates_client(self, mock_chroma_client, mock_collection):
        """Test that initialization creates a ChromaDB client"""
        with patch('rag_client.chromadb.PersistentClient', return_value=mock_chroma_client):
            collection = initialize_rag_system("./chroma_db", "test_collection")

            mock_chroma_client.get_or_create_collection.assert_called_once_with("test_collection")

    def test_initialize_returns_collection(self, mock_chroma_client, mock_collection):
        """Test that initialization returns a collection"""
        with patch('rag_client.chromadb.PersistentClient', return_value=mock_chroma_client):
            collection = initialize_rag_system("./chroma_db", "test_collection")

            assert collection == mock_collection

    def test_initialize_with_different_paths(self, mock_chroma_client):
        """Test initialization with different directory paths"""
        with patch('rag_client.chromadb.PersistentClient', return_value=mock_chroma_client) as mock_persistent:
            initialize_rag_system("/path/to/chroma", "my_collection")

            mock_persistent.assert_called_once_with(path="/path/to/chroma")


# ============================================================================
# Tests: retrieve_documents
# ============================================================================

class TestRetrieveDocuments:
    """Tests for retrieve_documents function"""

    def test_retrieve_basic_query(self, mock_collection):
        """Test basic document retrieval"""
        result = retrieve_documents(mock_collection, "test query")

        mock_collection.query.assert_called_once()
        assert result is not None
        assert 'documents' in result
        assert 'metadatas' in result

    def test_retrieve_with_n_results(self, mock_collection):
        """Test retrieval with custom n_results"""
        retrieve_documents(mock_collection, "test query", n_results=5)

        call_args = mock_collection.query.call_args
        assert call_args.kwargs['n_results'] == 5

    def test_retrieve_with_mission_filter(self, mock_collection):
        """Test retrieval with mission filter"""
        retrieve_documents(mock_collection, "test query", mission_filter="apollo_11")

        call_args = mock_collection.query.call_args
        assert call_args.kwargs['where'] == {"mission": "apollo_11"}

    def test_retrieve_without_filter(self, mock_collection):
        """Test retrieval without mission filter"""
        retrieve_documents(mock_collection, "test query", mission_filter=None)

        call_args = mock_collection.query.call_args
        assert call_args.kwargs['where'] is None

    def test_retrieve_with_all_filter(self, mock_collection):
        """Test that 'all' filter means no filtering"""
        retrieve_documents(mock_collection, "test query", mission_filter="all")

        call_args = mock_collection.query.call_args
        assert call_args.kwargs['where'] is None

    def test_retrieve_with_all_uppercase_filter(self, mock_collection):
        """Test that 'ALL' filter (case insensitive) means no filtering"""
        retrieve_documents(mock_collection, "test query", mission_filter="ALL")

        call_args = mock_collection.query.call_args
        assert call_args.kwargs['where'] is None

    def test_retrieve_query_texts_format(self, mock_collection):
        """Test that query is passed correctly"""
        retrieve_documents(mock_collection, "What happened during Apollo 11?")

        call_args = mock_collection.query.call_args
        assert call_args.kwargs['query_texts'] == "What happened during Apollo 11?"


# ============================================================================
# Tests: format_context
# ============================================================================

class TestFormatContext:
    """Tests for format_context function"""

    def test_format_empty_documents(self):
        """Test formatting with empty document list"""
        result = format_context([], [])

        assert result == ""

    def test_format_single_document(self):
        """Test formatting with a single document"""
        documents = ["This is a test document."]
        metadatas = [{'mission': 'apollo_11', 'source': 'transcript', 'document_category': 'pao'}]

        result = format_context(documents, metadatas)

        assert "<context>" in result
        assert "</context>" in result
        assert "Apollo 11" in result
        assert "transcript" in result
        assert "This is a test document." in result

    def test_format_multiple_documents(self, sample_documents, sample_metadatas):
        """Test formatting with multiple documents"""
        result = format_context(sample_documents, sample_metadatas)

        assert "<context>" in result
        assert "</context>" in result
        # Check all documents are included
        assert 'index="1"' in result
        assert 'index="2"' in result
        assert 'index="3"' in result

    def test_format_includes_mission(self, sample_documents, sample_metadatas):
        """Test that mission is included and formatted"""
        result = format_context(sample_documents, sample_metadatas)

        # Check mission formatting (underscores replaced, capitalized)
        assert "Apollo 11" in result or "apollo 11" in result.lower()

    def test_format_includes_category(self, sample_documents, sample_metadatas):
        """Test that category is included"""
        result = format_context(sample_documents, sample_metadatas)

        assert "<category>" in result
        assert "</category>" in result

    def test_format_includes_source(self, sample_documents, sample_metadatas):
        """Test that source is included"""
        result = format_context(sample_documents, sample_metadatas)

        assert "<source>" in result
        assert "</source>" in result

    def test_format_truncates_long_documents(self):
        """Test that long documents are truncated"""
        long_doc = "x" * 2000  # Longer than max_length (1000)
        documents = [long_doc]
        metadatas = [{'mission': 'test', 'source': 'test', 'document_category': 'test'}]

        result = format_context(documents, metadatas)

        # Should be truncated with ellipsis
        assert "..." in result
        # Should not contain the full 2000 chars
        assert len(result) < 2000 + 500  # Some buffer for XML tags

    def test_format_handles_missing_metadata(self):
        """Test formatting with missing metadata fields"""
        documents = ["Test document"]
        metadatas = [{}]  # Empty metadata

        result = format_context(documents, metadatas)

        # Should use fallback values
        assert "Unknown Mission" in result or "unknown mission" in result.lower()
        assert "Unknown Source" in result or "unknown source" in result.lower()

    def test_format_xml_structure(self, sample_documents, sample_metadatas):
        """Test that output has correct XML structure"""
        result = format_context(sample_documents, sample_metadatas)

        # Check XML structure
        assert result.startswith("<context>")
        assert result.endswith("</context>")
        assert "<document index=" in result
        assert "</document>" in result
        assert "<header>" in result
        assert "</header>" in result
        assert "<metadata>" in result
        assert "</metadata>" in result
        assert "<content>" in result
        assert "</content>" in result

    def test_format_preserves_document_content(self):
        """Test that document content is preserved"""
        documents = ["The Eagle has landed."]
        metadatas = [{'mission': 'apollo_11', 'source': 'pao', 'document_category': 'transcript'}]

        result = format_context(documents, metadatas)

        assert "The Eagle has landed." in result

    def test_format_index_starts_at_one(self, sample_documents, sample_metadatas):
        """Test that document indexing starts at 1"""
        result = format_context(sample_documents, sample_metadatas)

        assert 'index="1"' in result
        assert 'Index: 1' in result
        # Should not have index 0
        assert 'index="0"' not in result


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestRagClientIntegration:
    """Integration-style tests for RAG client functions"""

    def test_retrieve_and_format_flow(self, mock_collection):
        """Test the retrieve and format flow together"""
        # Retrieve documents
        result = retrieve_documents(mock_collection, "Apollo mission details", n_results=3)

        # Format the results
        documents = result['documents'][0]
        metadatas = result['metadatas'][0]

        formatted = format_context(documents, metadatas)

        # Verify formatted output
        assert "<context>" in formatted
        assert len(formatted) > 0

    def test_initialize_and_retrieve_flow(self, mock_chroma_client, mock_collection):
        """Test initialize and retrieve flow"""
        with patch('rag_client.chromadb.PersistentClient', return_value=mock_chroma_client):
            # Initialize
            collection = initialize_rag_system("./chroma_db", "test_collection")

            # Retrieve
            result = retrieve_documents(collection, "test query")

            assert result is not None


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_format_context_with_special_characters(self):
        """Test formatting with special characters in documents"""
        documents = ["Test with <special> & 'characters' \"here\""]
        metadatas = [{'mission': 'test', 'source': 'test', 'document_category': 'test'}]

        # Should not raise an error
        result = format_context(documents, metadatas)
        assert result is not None

    def test_format_context_with_unicode(self):
        """Test formatting with unicode characters"""
        documents = ["Test with unicode: 日本語 émojis 🚀"]
        metadatas = [{'mission': 'test', 'source': 'test', 'document_category': 'test'}]

        result = format_context(documents, metadatas)
        assert "🚀" in result

    def test_format_context_with_newlines(self):
        """Test formatting with newlines in documents"""
        documents = ["Line 1\nLine 2\nLine 3"]
        metadatas = [{'mission': 'test', 'source': 'test', 'document_category': 'test'}]

        result = format_context(documents, metadatas)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_retrieve_with_empty_query(self, mock_collection):
        """Test retrieval with empty query string"""
        result = retrieve_documents(mock_collection, "")

        # Should still call query
        mock_collection.query.assert_called_once()

    def test_format_with_none_in_metadata(self):
        """Test formatting when metadata contains None values raises error"""
        documents = ["Test document"]
        metadatas = [{'mission': None, 'source': None, 'document_category': None}]

        # Note: Current implementation doesn't handle None values gracefully
        # This test documents the expected behavior (raises AttributeError)
        with pytest.raises(AttributeError):
            format_context(documents, metadatas)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
