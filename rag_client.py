import chromadb
from chromadb.config import Settings
from chromadb import Collection,QueryResult,Where
from typing import Dict, List, Optional, TypedDict, Literal
from pathlib import Path
from datatypes import DbBuildInformation


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    chromadb_search_criteria = ("**/chroma*")
    exclude_dirs = {".venv", "__pycache__", ".git"}
    chroma_dirs = [
        path for path in current_dir.glob(chromadb_search_criteria)
        if path.is_dir()
        and not any(excluded in path.parts for excluded in exclude_dirs)
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(path=chroma_dir)
            db_collections = client.list_collections()
            for db_collection in db_collections:
                unique_identifier = f"{chroma_dir.name}-{db_collection}"
                try:
                    doc_count = db_collection.count()
                except:
                    doc_count = 0
                build_information: DbBuildInformation = {
                    "directory": str(chroma_dir),
                    "collection_name": db_collection.name,
                    "display_name": f"{chroma_dir.name}/{db_collection.name}",
                    "document_count": doc_count
                }

                backends[unique_identifier] = build_information
        except Exception as e:
            unique_identifier = f"{chroma_dir.name}-error"
            error_msg = str(e)[:50]
            build_information: DbBuildInformation = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (Error: {error_msg})",
                "document_count": 0,
            }
            backends[unique_identifier] = build_information
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_or_create_collection(collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection:Collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None,
                       similarity_threshold: Optional[float] = None) -> Optional[QueryResult]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    filter:Optional[Where] = None
    if mission_filter and mission_filter.lower() != "all":
        filter= {"mission":mission_filter}
    result = collection.query(
        query_texts=query,
        where=filter,
        n_results=n_results,
        # Include distances for similarity threshold filtering
        include=["documents", "metadatas", "distances"]
    )

    # Filter by similarity threshold if provided
    distances = result.get("distances")
    documents = result.get("documents")
    metadatas = result.get("metadatas")
    if similarity_threshold is not None and distances and documents and metadatas:
        # ChromaDB returns distances where lower = more similar
        # Filter out documents with distance > threshold
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []

        for i, distance in enumerate(distances[0]):
            if distance <= similarity_threshold:
                filtered_docs.append(documents[0][i])
                filtered_metadatas.append(metadatas[0][i])
                filtered_distances.append(distance)

        result["documents"] = [filtered_docs]
        result["metadatas"] = [filtered_metadatas]
        result["distances"] = [filtered_distances]

    return result


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""


    context_part: List[str] = ["<context>"]
    for idx, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        mission = metadata.get("mission", "Unknown Mission")
        mission = mission.replace("_", " ").capitalize()
        source = metadata.get("source", "Unknown Source")
        category =  metadata.get("document_category", "Unknown Category")
        category = category.replace("_", " ").capitalize()
        header = f"[Index: {idx}] Mission: {mission} | Category: {category} | Source: {source}"
        max_length = 1000
        if len(doc) > max_length:
            doc = doc[:max_length] + "..."

        # -> See implementation Notes why this has been choosen
        context_part.append(f"""<document index="{idx}">
<header>{header}</header>
<metadata>
<mission>{mission}</mission>
<category>{category}</category>
<source>{source}</source>
</metadata>
<content>
{doc}
</content>
</document>""")

    context_part.append("</context>")
    return "\n".join(context_part)
