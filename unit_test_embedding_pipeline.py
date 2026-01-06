#!/usr/bin/env python3
"""
Unit tests for ChromaDB Embedding Pipeline
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile
import os

from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock = Mock()
    mock_embedding = Mock()
    mock_embedding.embedding = [0.1] * 1536  # text-embedding-3-small dimension
    mock_response = Mock()
    mock_response.data = [mock_embedding]
    mock.embeddings.create.return_value = mock_response
    return mock


@pytest.fixture
def mock_collection():
    """Mock ChromaDB collection"""
    mock = Mock()
    mock.name = "test_collection"
    mock.count.return_value = 0
    mock.metadata = {"description": "Test collection"}
    mock.get.return_value = {'ids': [], 'metadatas': [], 'documents': []}
    mock.query.return_value = {
        'ids': [['id1', 'id2']],
        'documents': [['doc1', 'doc2']],
        'metadatas': [[{'source': 'test'}, {'source': 'test'}]],
        'distances': [[0.1, 0.2]]
    }
    return mock


@pytest.fixture
def mock_chroma_client(mock_collection):
    """Mock ChromaDB client"""
    mock = Mock()
    mock.get_or_create_collection.return_value = mock_collection
    return mock


@pytest.fixture
def pipeline(mock_openai_client, mock_chroma_client):
    """Create pipeline with mocked dependencies"""
    with patch('embedding_pipeline.OpenAI', return_value=mock_openai_client), \
         patch('embedding_pipeline.chromadb.PersistentClient', return_value=mock_chroma_client), \
         patch('embedding_pipeline.OpenAIEmbeddingFunction'):

        pipeline = ChromaEmbeddingPipelineTextOnly(
            openai_api_key="test-key",
            chroma_persist_directory="./test_chroma",
            collection_name="test_collection",
            chunk_size=100,
            chunk_overlap=20
        )
        pipeline.openai_client = mock_openai_client
        return pipeline


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document. It has multiple sentences. "
                "We use it for testing the pipeline.")
        temp_path = f.name
    yield Path(temp_path)
    os.unlink(temp_path)


@pytest.fixture
def temp_data_directory():
    """Create temporary data directory structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create apollo11 directory with text files
        apollo11_dir = Path(tmpdir) / "apollo11"
        apollo11_dir.mkdir()

        (apollo11_dir / "transcript_pao.txt").write_text(
            "Apollo 11 PAO transcript. The eagle has landed."
        )
        (apollo11_dir / "cm_transcript.txt").write_text(
            "Command module transcript content."
        )

        # Create challenger directory
        challenger_dir = Path(tmpdir) / "challenger"
        challenger_dir.mkdir()

        (challenger_dir / "mission_audio_01.txt").write_text(
            "Challenger mission audio transcript."
        )

        yield tmpdir


# ============================================================================
# Tests: Initialization
# ============================================================================

class TestInitialization:
    """Tests for pipeline initialization"""

    def test_init_stores_config(self, pipeline):
        """Test that initialization stores configuration correctly"""
        assert pipeline.api_key == "test-key"
        assert pipeline.chroma_persist_directory == "./test_chroma"
        assert pipeline.collection_name == "test_collection"
        assert pipeline.chunk_size == 100
        assert pipeline.chunk_overlap == 20

    def test_init_creates_collection(self, mock_chroma_client, mock_openai_client):
        """Test that initialization creates/gets collection"""
        with patch('embedding_pipeline.OpenAI', return_value=mock_openai_client), \
             patch('embedding_pipeline.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('embedding_pipeline.OpenAIEmbeddingFunction'):

            ChromaEmbeddingPipelineTextOnly(
                openai_api_key="test-key",
                collection_name="my_collection"
            )

            mock_chroma_client.get_or_create_collection.assert_called_once()


# ============================================================================
# Tests: Text Chunking
# ============================================================================

class TestChunkText:
    """Tests for chunk_text method"""

    def test_short_text_no_chunking(self, pipeline):
        """Test that short text is not chunked"""
        text = "Short text."
        metadata = {'source': 'test'}

        chunks = pipeline.chunk_text(text, metadata)

        assert len(chunks) == 1
        assert chunks[0][0] == text
        assert chunks[0][1]['chunk_index'] == 0
        assert chunks[0][1]['chunk_count'] == 1

    def test_long_text_chunking(self, pipeline):
        """Test that long text is properly chunked"""
        # Create text longer than chunk_size (100)
        text = "This is sentence one. " * 10  # ~220 chars
        metadata = {'source': 'test'}

        chunks = pipeline.chunk_text(text, metadata)

        assert len(chunks) > 1
        for i, (chunk_text, chunk_meta) in enumerate(chunks):
            assert chunk_meta['chunk_index'] == i
            assert chunk_meta['chunk_count'] == len(chunks)

    def test_chunk_metadata_preserved(self, pipeline):
        """Test that original metadata is preserved in chunks"""
        text = "Test sentence. " * 20
        metadata = {'source': 'test_file', 'mission': 'apollo_11'}

        chunks = pipeline.chunk_text(text, metadata)

        for _, chunk_meta in chunks:
            assert chunk_meta['source'] == 'test_file'
            assert chunk_meta['mission'] == 'apollo_11'


# ============================================================================
# Tests: Document ID Generation
# ============================================================================

class TestGenerateDocumentId:
    """Tests for generate_document_id method"""

    def test_id_format(self, pipeline):
        """Test document ID format"""
        file_path = Path("/data/apollo11/transcript.txt")
        metadata = {
            'mission': 'apollo_11',
            'source': 'transcript',
            'chunk_index': 5
        }

        doc_id = pipeline.generate_document_id(file_path, metadata)

        assert doc_id == "apollo_11_transcript_chunk_0005"

    def test_id_defaults(self, pipeline):
        """Test document ID with missing metadata"""
        file_path = Path("/data/unknown.txt")
        metadata = {}

        doc_id = pipeline.generate_document_id(file_path, metadata)

        assert doc_id == "unknown_unknown_chunk_0000"

    def test_id_consistency(self, pipeline):
        """Test that same inputs produce same ID"""
        file_path = Path("/data/test.txt")
        metadata = {'mission': 'apollo_11', 'source': 'test', 'chunk_index': 1}

        id1 = pipeline.generate_document_id(file_path, metadata)
        id2 = pipeline.generate_document_id(file_path, metadata)

        assert id1 == id2


# ============================================================================
# Tests: Mission Extraction
# ============================================================================

class TestExtractMissionFromPath:
    """Tests for extract_mission_from_path method"""

    def test_apollo11_detection(self, pipeline):
        """Test Apollo 11 mission detection"""
        paths = [
            Path("/data/apollo11/transcript.txt"),
            Path("/data/Apollo11/doc.txt"),
            Path("/data/apollo_11/file.txt"),
        ]

        for path in paths:
            assert pipeline.extract_mission_from_path(path) == 'apollo_11'

    def test_apollo13_detection(self, pipeline):
        """Test Apollo 13 mission detection"""
        paths = [
            Path("/data/apollo13/transcript.txt"),
            Path("/data/Apollo13/doc.txt"),
            Path("/data/apollo_13/file.txt"),
        ]

        for path in paths:
            assert pipeline.extract_mission_from_path(path) == 'apollo_13'

    def test_challenger_detection(self, pipeline):
        """Test Challenger mission detection"""
        path = Path("/data/challenger/audio.txt")
        assert pipeline.extract_mission_from_path(path) == 'challenger'

    def test_unknown_mission(self, pipeline):
        """Test unknown mission fallback"""
        path = Path("/data/gemini/transcript.txt")
        assert pipeline.extract_mission_from_path(path) == 'unknown'


# ============================================================================
# Tests: Data Type Extraction
# ============================================================================

class TestExtractDataTypeFromPath:
    """Tests for extract_data_type_from_path method"""

    def test_transcript_type(self, pipeline):
        """Test transcript type detection"""
        path = Path("/data/apollo11/transcript_pao.txt")
        assert pipeline.extract_data_type_from_path(path) == 'transcript'

    def test_textract_type(self, pipeline):
        """Test textract type detection"""
        path = Path("/data/apollo11_textract/doc.txt")
        assert pipeline.extract_data_type_from_path(path) == 'textract_extracted'

    def test_audio_type(self, pipeline):
        """Test audio type detection"""
        path = Path("/data/challenger/audio_segment.txt")
        assert pipeline.extract_data_type_from_path(path) == 'audio_transcript'

    def test_flight_plan_type(self, pipeline):
        """Test flight plan type detection"""
        path = Path("/data/apollo11/flight_plan.txt")
        assert pipeline.extract_data_type_from_path(path) == 'flight_plan'

    def test_default_document_type(self, pipeline):
        """Test default document type"""
        path = Path("/data/apollo11/notes.txt")
        assert pipeline.extract_data_type_from_path(path) == 'document'


# ============================================================================
# Tests: Document Category Extraction
# ============================================================================

class TestExtractDocumentCategoryFromFilename:
    """Tests for extract_document_category_from_filename method"""

    def test_pao_category(self, pipeline):
        """Test PAO category detection"""
        assert pipeline.extract_document_category_from_filename("pao_transcript.txt") == 'public_affairs_officer'

    def test_cm_category(self, pipeline):
        """Test command module category detection"""
        assert pipeline.extract_document_category_from_filename("cm_transcript.txt") == 'command_module'

    def test_technical_category(self, pipeline):
        """Test technical category detection"""
        assert pipeline.extract_document_category_from_filename("tec_report.txt") == 'technical'

    def test_mission_audio_category(self, pipeline):
        """Test mission audio category detection"""
        assert pipeline.extract_document_category_from_filename("mission_audio_01.txt") == 'mission_audio'

    def test_general_document_category(self, pipeline):
        """Test general document fallback"""
        assert pipeline.extract_document_category_from_filename("random_file.txt") == 'general_document'


# ============================================================================
# Tests: Document Existence Check
# ============================================================================

class TestCheckDocumentExists:
    """Tests for check_document_exists method"""

    def test_document_exists(self, pipeline, mock_collection):
        """Test when document exists"""
        mock_collection.get.return_value = {'ids': ['doc_id_1']}

        result = pipeline.check_document_exists('doc_id_1')

        assert result is True
        mock_collection.get.assert_called_with(ids=['doc_id_1'])

    def test_document_not_exists(self, pipeline, mock_collection):
        """Test when document does not exist"""
        mock_collection.get.return_value = {'ids': []}

        result = pipeline.check_document_exists('nonexistent_id')

        assert result is False


# ============================================================================
# Tests: Get Embedding
# ============================================================================

class TestGetEmbedding:
    """Tests for get_embedding method"""

    def test_get_embedding_returns_vector(self, pipeline, mock_openai_client):
        """Test that get_embedding returns a vector"""
        embedding = pipeline.get_embedding("test text")

        assert embedding is not None
        assert len(embedding) == 1536
        mock_openai_client.embeddings.create.assert_called_once()

    def test_get_embedding_uses_correct_model(self, pipeline, mock_openai_client):
        """Test that correct embedding model is used"""
        pipeline.get_embedding("test text")

        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args.kwargs['model'] == 'text-embedding-3-small'


# ============================================================================
# Tests: Process Text File
# ============================================================================

class TestProcessTextFile:
    """Tests for process_text_file method"""

    def test_process_valid_file(self, pipeline, temp_text_file):
        """Test processing a valid text file"""
        documents = pipeline.process_text_file(temp_text_file)

        assert len(documents) > 0
        text, metadata = documents[0]
        assert 'source' in metadata
        assert 'mission' in metadata
        assert 'file_type' in metadata
        assert metadata['file_type'] == 'text'

    def test_process_empty_file(self, pipeline):
        """Test processing an empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            documents = pipeline.process_text_file(Path(temp_path))
            assert documents == []
        finally:
            os.unlink(temp_path)

    def test_process_nonexistent_file(self, pipeline):
        """Test processing a nonexistent file"""
        documents = pipeline.process_text_file(Path("/nonexistent/file.txt"))
        assert documents == []


# ============================================================================
# Tests: Scan Text Files
# ============================================================================

class TestScanTextFilesOnly:
    """Tests for scan_text_files_only method"""

    def test_scan_finds_txt_files(self, pipeline, temp_data_directory):
        """Test that scan finds .txt files"""
        files = pipeline.scan_text_files_only(temp_data_directory)

        assert len(files) == 3
        for f in files:
            assert f.suffix == '.txt'

    def test_scan_excludes_summary_files(self, pipeline, temp_data_directory):
        """Test that summary files are excluded"""
        # Add a summary file
        summary_file = Path(temp_data_directory) / "apollo11" / "summary.txt"
        summary_file.write_text("Summary content")

        files = pipeline.scan_text_files_only(temp_data_directory)

        assert not any('summary' in f.name.lower() for f in files)

    def test_scan_excludes_hidden_files(self, pipeline, temp_data_directory):
        """Test that hidden files are excluded"""
        # Add a hidden file
        hidden_file = Path(temp_data_directory) / "apollo11" / ".hidden.txt"
        hidden_file.write_text("Hidden content")

        files = pipeline.scan_text_files_only(temp_data_directory)

        assert not any(f.name.startswith('.') for f in files)


# ============================================================================
# Tests: Update Document
# ============================================================================

class TestUpdateDocument:
    """Tests for update_document method"""

    def test_update_document_success(self, pipeline, mock_collection, mock_openai_client):
        """Test successful document update"""
        result = pipeline.update_document(
            doc_id='test_id',
            text='Updated text',
            metadata={'source': 'test'}
        )

        assert result is True
        mock_collection.update.assert_called_once()

    def test_update_document_failure(self, pipeline, mock_collection):
        """Test document update failure"""
        mock_collection.update.side_effect = Exception("Update failed")

        result = pipeline.update_document(
            doc_id='test_id',
            text='Updated text',
            metadata={'source': 'test'}
        )

        assert result is False


# ============================================================================
# Tests: Delete Documents by Source
# ============================================================================

class TestDeleteDocumentsBySource:
    """Tests for delete_documents_by_source method"""

    def test_delete_matching_documents(self, pipeline, mock_collection):
        """Test deleting documents by source pattern"""
        mock_collection.get.return_value = {
            'ids': ['id1', 'id2', 'id3'],
            'metadatas': [
                {'source': 'apollo11_transcript'},
                {'source': 'apollo11_pao'},
                {'source': 'challenger_audio'}
            ]
        }

        count = pipeline.delete_documents_by_source('apollo11')

        assert count == 2
        mock_collection.delete.assert_called_once()

    def test_delete_no_matching_documents(self, pipeline, mock_collection):
        """Test when no documents match pattern"""
        mock_collection.get.return_value = {
            'ids': ['id1'],
            'metadatas': [{'source': 'challenger_audio'}]
        }

        count = pipeline.delete_documents_by_source('gemini')

        assert count == 0
        mock_collection.delete.assert_not_called()


# ============================================================================
# Tests: Get File Documents
# ============================================================================

class TestGetFileDocuments:
    """Tests for get_file_documents method"""

    def test_get_file_documents(self, pipeline, mock_collection):
        """Test getting documents for a file"""
        mock_collection.get.return_value = {
            'ids': ['id1', 'id2', 'id3'],
            'metadatas': [
                {'source': 'transcript', 'mission': 'apollo_11'},
                {'source': 'transcript', 'mission': 'apollo_11'},
                {'source': 'other', 'mission': 'apollo_13'}
            ]
        }

        file_path = Path("/data/apollo11/transcript.txt")
        doc_ids = pipeline.get_file_documents(file_path)

        assert len(doc_ids) == 2
        assert 'id1' in doc_ids
        assert 'id2' in doc_ids


# ============================================================================
# Tests: Add Documents to Collection
# ============================================================================

class TestAddDocumentsToCollection:
    """Tests for add_documents_to_collection method"""

    def test_add_empty_documents(self, pipeline):
        """Test adding empty document list"""
        result = pipeline.add_documents_to_collection(
            documents=[],
            file_path=Path("/test.txt")
        )

        assert result == {'added': 0, 'updated': 0, 'skipped': 0}

    def test_add_documents_skip_mode(self, pipeline, mock_collection, mock_openai_client):
        """Test adding documents in skip mode"""
        mock_collection.get.return_value = {'ids': []}

        documents = [
            ("Test text 1", {'source': 'test', 'chunk_index': 0, 'mission': 'apollo_11'}),
            ("Test text 2", {'source': 'test', 'chunk_index': 1, 'mission': 'apollo_11'})
        ]

        result = pipeline.add_documents_to_collection(
            documents=documents,
            file_path=Path("/apollo11/test.txt"),
            update_mode='skip'
        )

        assert result['added'] == 2
        assert result['skipped'] == 0
        mock_collection.add.assert_called_once()

    def test_add_documents_skip_existing(self, pipeline, mock_collection, mock_openai_client):
        """Test that existing documents are skipped"""
        # First document already exists
        mock_collection.get.return_value = {'ids': ['apollo_11_test_chunk_0000']}

        documents = [
            ("Test text 1", {'source': 'test', 'chunk_index': 0, 'mission': 'apollo_11'}),
            ("Test text 2", {'source': 'test', 'chunk_index': 1, 'mission': 'apollo_11'})
        ]

        result = pipeline.add_documents_to_collection(
            documents=documents,
            file_path=Path("/apollo11/test.txt"),
            update_mode='skip'
        )

        assert result['skipped'] == 1
        assert result['added'] == 1

    def test_add_documents_update_mode(self, pipeline, mock_collection, mock_openai_client):
        """Test adding documents in update mode using upsert"""
        mock_collection.get.return_value = {'ids': ['apollo_11_test_chunk_0000']}

        documents = [
            ("Test text 1", {'source': 'test', 'chunk_index': 0, 'mission': 'apollo_11'}),
            ("Test text 2", {'source': 'test', 'chunk_index': 1, 'mission': 'apollo_11'})
        ]

        result = pipeline.add_documents_to_collection(
            documents=documents,
            file_path=Path("/apollo11/test.txt"),
            update_mode='update'
        )

        assert result['updated'] == 1
        assert result['added'] == 1
        mock_collection.upsert.assert_called_once()

    def test_add_documents_replace_mode(self, pipeline, mock_collection, mock_openai_client):
        """Test adding documents in replace mode"""
        # Mock get_file_documents to return existing IDs
        mock_collection.get.return_value = {
            'ids': ['old_id_1', 'old_id_2'],
            'metadatas': [
                {'source': 'test', 'mission': 'apollo_11'},
                {'source': 'test', 'mission': 'apollo_11'}
            ]
        }

        documents = [
            ("New text", {'source': 'test', 'chunk_index': 0, 'mission': 'apollo_11'})
        ]

        result = pipeline.add_documents_to_collection(
            documents=documents,
            file_path=Path("/apollo11/test.txt"),
            update_mode='replace'
        )

        # Should delete old docs and add new ones
        mock_collection.delete.assert_called()
        mock_collection.add.assert_called()


# ============================================================================
# Tests: Process All Text Data
# ============================================================================

class TestProcessAllTextData:
    """Tests for process_all_text_data method"""

    def test_process_all_returns_stats(self, pipeline, temp_data_directory, mock_collection, mock_openai_client):
        """Test that process_all_text_data returns statistics"""
        mock_collection.get.return_value = {'ids': []}

        stats = pipeline.process_all_text_data(temp_data_directory)

        assert 'files_processed' in stats
        assert 'documents_added' in stats
        assert 'documents_updated' in stats
        assert 'documents_skipped' in stats
        assert 'errors' in stats
        assert 'missions' in stats

    def test_process_all_with_update_mode(self, pipeline, temp_data_directory, mock_collection, mock_openai_client):
        """Test processing with different update modes"""
        mock_collection.get.return_value = {'ids': []}

        # Test skip mode
        stats_skip = pipeline.process_all_text_data(temp_data_directory, update_mode='skip')
        assert stats_skip['files_processed'] > 0

        # Test update mode
        stats_update = pipeline.process_all_text_data(temp_data_directory, update_mode='update')
        assert stats_update['files_processed'] > 0


# ============================================================================
# Tests: Get Collection Info
# ============================================================================

class TestGetCollectionInfo:
    """Tests for get_collection_info method"""

    def test_get_collection_info(self, pipeline, mock_collection):
        """Test getting collection info"""
        mock_collection.count.return_value = 100

        info = pipeline.get_collection_info()

        assert info['name'] == 'test_collection'
        assert info['count'] == 100
        assert 'metadata' in info


# ============================================================================
# Tests: Query Collection
# ============================================================================

class TestQueryCollection:
    """Tests for query_collection method"""

    def test_query_collection(self, pipeline, mock_collection, mock_openai_client):
        """Test querying the collection"""
        results = pipeline.query_collection("test query", n_results=5)

        assert 'documents' in results
        assert 'metadatas' in results
        assert 'distances' in results
        mock_collection.query.assert_called_once()

    def test_query_uses_embedding(self, pipeline, mock_collection, mock_openai_client):
        """Test that query uses embedding"""
        pipeline.query_collection("test query")

        # Verify embedding was generated
        mock_openai_client.embeddings.create.assert_called()


# ============================================================================
# Tests: Get Collection Stats
# ============================================================================

class TestGetCollectionStats:
    """Tests for get_collection_stats method"""

    def test_get_collection_stats(self, pipeline, mock_collection):
        """Test getting collection statistics"""
        mock_collection.get.return_value = {
            'ids': ['id1', 'id2'],
            'metadatas': [
                {'mission': 'apollo_11', 'data_type': 'transcript',
                 'document_category': 'pao', 'file_type': 'text'},
                {'mission': 'challenger', 'data_type': 'audio',
                 'document_category': 'mission_audio', 'file_type': 'text'}
            ]
        }

        stats = pipeline.get_collection_stats()

        assert stats['total_documents'] == 2
        assert 'missions' in stats
        assert 'data_types' in stats
        assert 'document_categories' in stats
        assert stats['missions']['apollo_11'] == 1
        assert stats['missions']['challenger'] == 1

    def test_get_collection_stats_empty(self, pipeline, mock_collection):
        """Test stats on empty collection"""
        mock_collection.get.return_value = {'ids': [], 'metadatas': []}

        stats = pipeline.get_collection_stats()

        assert 'error' in stats


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
