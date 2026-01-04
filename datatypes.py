from typing import Dict, List, Optional,TypedDict
from pydantic import BaseModel

class DbBuildInformation(TypedDict):
  # DONE: Store directory path as string
  directory:str
  # DONE: Store collection name
  collection_name:str
  # DONE: Create user-friendly display name
  display_name:str
  # DONE: Get document count with fallback for unsupported operations
  document_count:int 

                    
                    