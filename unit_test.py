import os
from dotenv import load_dotenv
load_dotenv()

# Initialize pipeline with test configuration
openai_key = os.environ.get('OPENAI_API_KEY')
# from llm_client import generate_response
# response = generate_response(os.environ['OPENAI_API_KEY'], "What was Apollo 11?", "", [])
# print(response)

# from rag_client import initialize_rag_system
# chroma_dir = "chroma_db_test"
# chroma_collection = "test123"
# backends = initialize_rag_system(chroma_dir,chroma_collection)
# print(backends)

# from rag_client import discover_chroma_backends
# backends = discover_chroma_backends()
# print(backends)

# from rag_client import format_context
# # Dummy documents (content)
# documents = [
#     "The Apollo 11 mission successfully landed the first humans on the Moon on July 20, 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle.",
#     "Apollo 13 was the seventh crewed mission in the Apollo space program. The craft was launched from Kennedy Space Center on April 11, 1970, but the lunar landing was aborted after an oxygen tank in the service module exploded.",
#     "The Space Shuttle Challenger disaster was a fatal incident on January 28, 1986. The spacecraft broke apart 73 seconds into its flight, killing all seven crew members aboard."
# ]

# # Dummy metadata matching your schema
# metadatas = [
#     {
#         "mission": "apollo_11",
#         "source": "apollo11_landing_report.txt",
#         "document_category": "mission_overview",
#         "file_type": "text"
#     },
#     {
#         "mission": "apollo_13", 
#         "source": "apollo13_incident.txt",
#         "document_category": "emergency_procedures",
#         "file_type": "text"
#     },
#     {
#         "mission": "challenger",
#         "source": "challenger_report.txt", 
#         "document_category": "disaster_analysis",
#         "file_type": "text"
#     }
# ]
# context = format_context(documents, metadatas)
# print(context)

# from llm_client import generate_response
# response = generate_response(os.environ['OPENAI_API_KEY'], "What was Apollo 13?", context, [])
# print(response)

# Test chunk_text method
from embedding_pipeline import ChromaEmbeddingPipelineTextOnly


if openai_key:
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=openai_key,
        chroma_persist_directory="./chroma_db_test",
        collection_name="test_chunks",
        chunk_size=100,
        chunk_overlap=20
    )
    
    # Test 1: Short text (no chunking needed)
    print("Test 1: Short text")
    short_text = "This is a short sentence."
    metadata = {
        "mission": "test_mission",
        "source": "test_source.txt",
        "document_category": "test"
    }
    result = pipeline.chunk_text(short_text, metadata)
    print(f"Input length: {len(short_text)}")
    print(f"Number of chunks: {len(result)}")
    print(f"Chunk 0: {result[0][0]}")
    print(f"Metadata: {result[0][1]}\n")
    
    # Test 2: Long text with multiple sentences
    print("Test 2: Long text with multiple sentences")
    long_text = "The Apollo 11 mission was historic. It landed on the Moon. Neil Armstrong walked on the lunar surface. Buzz Aldrin joined him. Michael Collins orbited above. This was a tremendous achievement for humanity."
    result = pipeline.chunk_text(long_text, metadata)
    print(f"Input length: {len(long_text)}")
    print(f"Number of chunks: {len(result)}")
    for i, (chunk_text, chunk_meta) in enumerate(result):
        print(f"Chunk {i}: {chunk_text}")
        print(f"  Size: {chunk_meta['chunk_size']}, Index: {chunk_meta['chunk_index']}, Count: {chunk_meta['chunk_count']}")
    print()
    
    # Test 3: Verify overlap between chunks
    print("Test 3: Verify overlap between chunks")
    if len(result) > 1:
        chunk_0_text = result[0][0]
        chunk_1_text = result[1][0]
        
        # Split into sentences and check overlap
        chunk_0_sentences = chunk_0_text.split('. ')
        chunk_1_sentences = chunk_1_text.split('. ')
        
        # Check if last sentence of chunk 0 appears in chunk 1
        if chunk_0_sentences and chunk_1_sentences:
            last_sent_chunk_0 = chunk_0_sentences[-1]
            overlap_found = last_sent_chunk_0 in chunk_1_text
            print(f"Last sentence of Chunk 0: {last_sent_chunk_0}")
            print(f"First sentence of Chunk 1: {chunk_1_sentences[0]}")
            print(f"Overlap detected: {overlap_found}")
            
            # Show actual overlap
            for sent in chunk_0_sentences:
                if sent in chunk_1_text:
                    print(f"  Overlapping sentence: {sent}")
else:
    print("OPENAI_API_KEY not found in environment")

