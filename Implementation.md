
llm_client.py:
- I decided to use the repsonse this is a more simplified API and gives us more control over the model and teh behaviour the documentation is here https://platform.openai.com/docs/guides/migrate-to-responses?update-item-definitions=responses&update-multiturn=chat-completions

rag_client.py
- decided to use for the path detection unix style path detection (easier to adjust pattern) https://docs.python.org/3/library/glob.html

- for the format format_context i decided to use xml i create a test prompt which is in the notes.
The note instructs the LLM to evaluate the parsing methods and attach a number how certian it was.
XML has performed the best across the multiple LLMs and provider. This extra tokens should be well spend.
I remove the indention as it is just to make it more human redable but it saves token the LLM parses the schematic structure it should be fine.

https://community.openai.com/t/providing-context-to-the-chat-api-before-a-conversation/195853/6

rag pipeline:
- used the documentation here https://docs.trychroma.com/integrations/embedding-models/openai in order to implement the embedding
- also use the settings to disable telemtry https://docs.trychroma.com/docs/overview/telemetry no nasa system want to leaks docs
- for the sentence splitting i decided to use regex (?<=[.!?]) (https://stackoverflow.com/questions/2973436/regex-lookahead-lookbehind-and-atomic-groups) this splits the text at .!? it  may not be precise.
- embeddings https://www.pinecone.io/learn/vector-search-filtering/