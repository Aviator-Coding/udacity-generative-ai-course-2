import os
from dotenv import load_dotenv
load_dotenv()

# from llm_client import generate_response
# response = generate_response(os.environ['OPENAI_API_KEY'], "What was Apollo 11?", "", [])
# print(response)

from rag_client import initialize_rag_system
chroma_dir = "chroma_db_test"
chroma_collection = "test123"
backends = initialize_rag_system(chroma_dir,chroma_collection)
print(backends)

from rag_client import discover_chroma_backends
backends = discover_chroma_backends()
print(backends)

from rag_client import format_context
# Dummy documents (content)
documents = [
    "The Apollo 11 mission successfully landed the first humans on the Moon on July 20, 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle.",
    "Apollo 13 was the seventh crewed mission in the Apollo space program. The craft was launched from Kennedy Space Center on April 11, 1970, but the lunar landing was aborted after an oxygen tank in the service module exploded.",
    "The Space Shuttle Challenger disaster was a fatal incident on January 28, 1986. The spacecraft broke apart 73 seconds into its flight, killing all seven crew members aboard."
]

# Dummy metadata matching your schema
metadatas = [
    {
        "mission": "apollo_11",
        "source": "apollo11_landing_report.txt",
        "document_category": "mission_overview",
        "file_type": "text"
    },
    {
        "mission": "apollo_13", 
        "source": "apollo13_incident.txt",
        "document_category": "emergency_procedures",
        "file_type": "text"
    },
    {
        "mission": "challenger",
        "source": "challenger_report.txt", 
        "document_category": "disaster_analysis",
        "file_type": "text"
    }
]
context = format_context(documents, metadatas)
print(context)

from llm_client import generate_response
response = generate_response(os.environ['OPENAI_API_KEY'], "What was Apollo 13?", context, [])
print(response)