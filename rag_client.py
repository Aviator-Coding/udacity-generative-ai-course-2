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
    # DONE: Create list of directories that match specific criteria (directory type and name pattern)
    chromadb_search_criteria = ("**/chroma*")
    exclude_dirs = {".venv", "__pycache__", ".git"}
    chroma_dirs = [
        path for path in current_dir.glob(chromadb_search_criteria)
        if path.is_dir()
        and not any(excluded in path.parts for excluded in exclude_dirs)
    ]

    # DONE: Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # DONE: Wrap connection attempt in try-except block for error handling
        try:
            # DONE: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=chroma_dir)
            # DONE: Retrieve list of available collections from the database
            db_collections = client.list_collections()
            # DONE: Loop through each collection found
            for db_collection in db_collections:
                # DONE: Create unique identifier key combining directory and collection names
                unique_identifier = f"{chroma_dir.name}-{db_collection}"
                # DONE: Get document count with fallback for unsupported operations
                try:
                    doc_count = db_collection.count()
                except:
                    doc_count = 0
                # DONE: Build information dictionary containing:
                build_information: DbBuildInformation = {
                    # DONE: Store directory path as string
                    "directory": str(chroma_dir),
                    # DONE: Store collection name
                    "collection_name": db_collection.name,
                    # DONE: Create user-friendly display name
                    "display_name": f"{chroma_dir.name}/{db_collection.name}",
                    # DONE: Get document count
                    "document_count": doc_count
                }

                # DONE: Add collection information to backends dictionary
                backends[unique_identifier] = build_information
        # DONE: Handle connection or access errors gracefully
        except Exception as e:
            # DONE: Create fallback entry for inaccessible directories
            unique_identifier = f"{chroma_dir.name}-error"
            # DONE: Include error information in display name with truncation
            error_msg = str(e)[:50]
            # DONE: Set appropriate fallback values for missing information
            build_information: DbBuildInformation = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (Error: {error_msg})",
                "document_count": 0,
            }
            # DONE: Add Error information to backends dictionary
            backends[unique_identifier] = build_information
    # DONE: Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # DONE: Create a chomadb persistentclient
    client = chromadb.PersistentClient(path=chroma_dir)
    # DONE: Return the collection with the collection_name
    return client.get_or_create_collection(collection_name)


def retrieve_documents(collection:Collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[QueryResult]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # DONE: Initialize filter variable to None (represents no filtering)
    filter:Optional[Where] = None
    # DONE: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
    # DONE: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        filter= {"mission":mission_filter}
    # DONE: Execute database query with the following parameters:
    result = collection.query(
         # DONE: Pass search query in the required format
        query_texts=query,
        # DONE: Apply conditional filter (None for no filtering, dictionary for specific filtering)
        where=filter,
        # DONE: Set maximum number of results to return
        n_results=n_results
    )
    # DONE: Return query results to caller
    return result

    
def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""


    # DONE: Initialize list with header text for context section
    context_part: List[str] = ["<context>"]
    # DONE: Loop through paired documents and their metadata using enumeration
    for idx, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        # DONE: Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "Unknown Mission")
        # DONE: Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").capitalize()
        # DONE: Extract source information from metadata with fallback value
        source = metadata.get("source", "Unknown Source")
        # DONE: Extract category information from metadata with fallback value
        category =  metadata.get("document_category", "Unknown Category")
        # DONE: Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").capitalize()
        # DONE: Create formatted source header with index number and extracted information
        header = f"[Index: {idx}] Mission: {mission} | Category: {category} | Source: {source}"
        # DONE: Check document length and truncate if necessary
        max_length = 1000
        if len(doc) > max_length:
            doc = doc[:max_length] + "..."

        # DONE: Add source header to context parts list 
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

        # DONE: Add truncated or full document content to context parts list
    context_part.append("</context>")
    # DONE: Join all context parts with newlines and return formatted string
    return "\n".join(context_part)
