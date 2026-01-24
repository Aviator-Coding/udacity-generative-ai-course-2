from typing import Dict, List, Optional,TypedDict
from pydantic import BaseModel

class DbBuildInformation(TypedDict):
  directory:str
  collection_name:str
  display_name:str
  document_count:int



