# NASA Intelligence Chat System

A complete Retrieval-Augmented Generation (RAG) system for querying NASA's historic space mission documents.

## Project Overview

Build a Q&A system that answers questions about NASA's most historic space missions using actual mission transcripts and technical documents from **Apollo 11**, **Apollo 13**, and the **Challenger** missions.

> *Example: Ask "What problems did Apollo 13 encounter?" and get an accurate, detailed answer sourced directly from NASA's archives.*

---

## Project Architecture

| Component | File | Description |
|-----------|------|-------------|
| **Embedding Pipeline** | `embedding_pipeline.py` | Process NASA text files into chunks and store as embeddings in ChromaDB |
| **RAG Client** | `rag_client.py` | Search ChromaDB for relevant document chunks and format as context |
| **LLM Client** | `llm_client.py` | Connect to OpenAI API and generate answers using retrieved context |
| **RAGAS Evaluator** | `ragas_evaluator.py` | Real-time evaluation of response quality (faithfulness, relevancy) |
| **Chat Application** | `chat.py` | Interactive Streamlit interface bringing all components together |

---

## Skills Demonstrated

- Building an end-to-end RAG system
- Using vector databases (ChromaDB) for semantic search
- Integrating and prompting large language models
- Evaluating AI system performance with modern tools

---

## Implementation Guide

### Phase 1: Core Infrastructure

#### 1. LLM Client (`llm_client.py`)

Connect to the OpenAI API and create a NASA expert persona.

**Tasks:**
- Define a system prompt that tells the model to act as a NASA expert
- Manage conversation history so the model can remember previous turns
- Write the function that sends requests to OpenAI and returns responses

---

#### 2. RAG Client (`rag_client.py`)

Build the retrieval system that searches for relevant document chunks.

**Tasks:**
- Connect to the ChromaDB backend
- Implement semantic search to find the best matching document chunks
- Format retrieved documents into a clean context string for the LLM

---

#### 3. Embedding Pipeline (`embedding_pipeline.py`)

Process NASA text files and create the vector database.

**Tasks:**
- Implement a text chunking strategy with overlap
- Use OpenAI API to generate embeddings for each chunk
- Manage ChromaDB collection creation and population
- Build a command-line interface for running the pipeline

---

### Phase 2: Evaluation and Interface

#### 4. RAGAS Evaluator (`ragas_evaluator.py`)

Implement real-time quality scoring for RAG responses.

**Tasks:**
- Integrate the RAGAS framework
- Define evaluation metrics (faithfulness, answer relevancy, context precision)
- Write the function that returns quality scores for each response

---

#### 5. Chat Application (`chat.py`)

Create the interactive Streamlit interface.

**Tasks:**
- Build the chat interface for user questions
- Integrate all components (RAG, LLM, evaluation)
- Display real-time quality metrics in the interface

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# 3. Process documents (create embeddings)
python embedding_pipeline.py --openai-key $OPENAI_API_KEY --data-path ./data

# 3.1 Test with existing collection stats 
python embedding_pipeline.py --openai-key $OPENAI_API_KEY --stats-only  

# 3.2  Test a query to verify embeddings work 
python embedding_pipeline.py --openai-key $OPENAI_API_KEY --stats-only --test-query "Apollo 11 mission" 

# 4. Launch the chat interface
streamlit run chat.py
```

---

## Submission Checklist

- [ ] All TODO items implemented in all Python files
- [ ] End-to-end testing completed (embedding → chat → evaluation)
- [ ] Sample questions provided in `evaluation_dataset.txt`
- [ ] All code is clean and documented
- [ ] Project files zipped into a single archive

---

## Implementation Notes

### LLM Client (`llm_client.py`)
- I decided to use the Response API as it provides a more simplified interface and gives us more control over the model behavior
- Documentation: https://platform.openai.com/docs/guides/migrate-to-responses?update-item-definitions=responses&update-multiturn=chat-completions

### RAG Client (`rag_client.py`)
- Decided to use Unix-style path detection with glob patterns (easier to adjust patterns)
- Documentation: https://docs.python.org/3/library/glob.html

- For `format_context`, I decided to use XML formatting. I created a test prompt (LLM Delimiter Format Benchmark) which instructs the LLM to evaluate different parsing methods and rate how certain it was about each format.
- XML performed the best across multiple LLMs and providers. The extra tokens should be well spent.
- I removed the indentation as it's just for human readability but saves tokens - the LLM parses the schematic structure fine without it.
- Reference: https://community.openai.com/t/providing-context-to-the-chat-api-before-a-conversation/195853/6

### Embedding Pipeline (`embedding_pipeline.py`)
- Used the ChromaDB OpenAI integration documentation: https://docs.trychroma.com/integrations/embedding-models/openai
- Disabled telemetry via settings: https://docs.trychroma.com/docs/overview/telemetry (no NASA system wants to leak docs)
- For sentence splitting, I used regex `(?<=[.!?])` - this splits text at `.!?` characters. It may not be precise for all cases but works well for NASA documents.
- Reference: https://stackoverflow.com/questions/2973436/regex-lookahead-lookbehind-and-atomic-groups
- Vector search filtering concepts: https://www.pinecone.io/learn/vector-search-filtering/
- Another lesson i leraned was to batch requests, i made a misstake resulting in 1000 embedding requests rather then batching them, i was
really lucky that this ony cost a few cent. 1000 request took like 15 minutes to create the embeddings.I created a new batch method which does it in about 50 requests.


### RAGAS Evaluator (`ragas_evaluator.py`)
- The RAGAS implementation was challenging at the beginning, but it provided a deeper understanding through experimentation.
- Fun fact: At the beginning I got perfect answers and realized the model was hallucinating everything because the ChromaDB was missing data - this was a "shocking" discovery!
- I learned a lot about these concepts:
  - **Faithfulness** - Measures if response claims are supported by the retrieved context
  - **ResponseRelevancy** - Measures if the answer addresses the user's question
  - **LLMContextPrecisionWithoutReference** - Evaluates context relevance without needing ground truth
- Documentation: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

---

## LLM Delimiter Format Benchmark

This benchmark tests how well different LLMs can parse and extract data from various delimiter formats. The same prompt was sent to three models to compare their parsing confidence and recommendations.

### The Benchmark Prompt

```
You are participating in a benchmark test to evaluate how well you can parse and extract data from different delimiter formats.

For each of the 10 test cases below, you must:
1. Extract these three fields: NAME, ROLE, TASK
2. Rate your parsing confidence from 1-5 (5 = completely unambiguous, 1 = had to guess)
3. Note any ambiguities or parsing challenges

Respond in this exact format for each test:

TEST [N] [FORMAT]:
- Name: [extracted value]
- Role: [extracted value]
- Task: [extracted value]
- Confidence: [1-5]/5
- Notes: [any parsing challenges]
```

<details>
<summary><strong>View All 10 Test Cases</strong></summary>

#### TEST 1: XML TAGS
```xml
<user>
  <name>Alice Chen</name>
  <role>Senior DevOps Engineer</role>
  <task>Deploy Kubernetes cluster with Rook-Ceph storage</task>
</user>
```

#### TEST 2: TRIPLE HASHES
```
###NAME###
Bob Martinez
###ROLE###
Database Administrator
###TASK###
Migrate PostgreSQL 14 to PostgreSQL 17 with pgvector
###END###
```

#### TEST 3: JSON FORMAT
```json
{
  "name": "Carol Wu",
  "role": "ML Engineer",
  "task": "Build MCP server for AWS Athena integration"
}
```

#### TEST 4: MARKDOWN HEADERS
```markdown
## Name
David Park

## Role
Platform Engineer

## Task
Configure FluxCD GitOps pipeline for production
```

#### TEST 5: TEXT LABELS WITH COLONS
```
NAME: Elena Volkov
ROLE: Security Architect
TASK: Implement zero-trust network architecture
```

#### TEST 6: TRIPLE DASHES (YAML-STYLE)
```yaml
---
name: Frank Okonkwo
---
role: Data Engineer
---
task: Deploy TiDB Operator on Kubernetes cluster
---
```

#### TEST 7: TRIPLE QUOTES
```
"""
Name: Grace Kim
Role: SRE Lead
Task: Debug Redis cluster failover issues
"""
```

#### TEST 8: NESTED XML (COMPLEX)
```xml
<request>
  <context>
    <environment>Production</environment>
    <priority>High</priority>
  </context>
  <user>
    <name>Henry Liu</name>
    <role>Infrastructure Lead</role>
  </user>
  <task>
    <description>Troubleshoot load balancer configuration</description>
  </task>
</request>
```

#### TEST 9: PIPE-SEPARATED
```
|NAME|ROLE|TASK|
|Ivan Petrov|Backend Developer|Optimize API response times|
```

#### TEST 10: NATURAL LANGUAGE (STRESS TEST)
```
The engineer named Julia Santos, who currently works as a Cloud Architect, has been assigned to design the multi-region disaster recovery system for our infrastructure.
```

</details>

---

### Model Responses

#### Claude Opus Results

| Rank | Delimiter Type | Confidence | Best For | Limitations |
|------|---------------|------------|----------|-------------|
| 1 | **XML Tags** | 5.0/5 | Claude, complex/nested data | Verbose syntax |
| 1 | **JSON** | 5.0/5 | Structured output, API integration | Less human-readable |
| 3 | **Markdown Headers** | 4.0/5 | Documentation, human-readable | Can conflict with doc structure |
| 3 | **Triple Hashes** | 4.0/5 | Section separation, simple prompts | No native nesting |
| 5 | **Triple Dashes** | 3.5/5 | YAML-style, block separation | Can confuse with YAML frontmatter |
| 5 | **Text Labels** | 3.5/5 | Simple, quick prompts | Weak boundaries |
| 7 | **Triple Quotes** | 3.0/5 | Enclosing content blocks | No internal structure |
| 8 | **Pipe-separated** | 3.0/5 | Tabular data only | Poor for text with pipes |
| 9 | **Natural Language** | 2.0/5 | Quick informal prompts | Highly ambiguous |

**Claude's Recommendation:** XML tags are the clear winner for Claude - Anthropic specifically trained Claude to recognize XML as a prompt organizing mechanism.

---

#### OpenAI GPT-4 Results

| Test | Format | Confidence | Notes |
|------|--------|------------|-------|
| 1 | XML Tags | 5/5 | Clear XML tags; no ambiguity |
| 2 | Triple Hashes | 5/5 | Explicit section delimiters |
| 3 | JSON | 5/5 | Valid JSON with clear keys |
| 4 | Markdown Headers | 5/5 | Headers clearly label each field |
| 5 | Text Labels | 5/5 | Colon-separated labels are explicit |
| 6 | YAML-Style | 5/5 | Clear despite repeated separators |
| 7 | Triple Quotes | 5/5 | Labels inside quoted block remove ambiguity |
| 8 | Nested XML | 4/5 | Required extracting nested element |
| 9 | Pipe-Separated | 4/5 | Depends on header-to-column alignment |
| 10 | Natural Language | 3/5 | Required semantic inference |

---

#### OpenAI GPT-3.5-Turbo Results

| Test | Format | Confidence | Notes |
|------|--------|------------|-------|
| 1 | XML Tags | 5/5 | Straightforward and unambiguous |
| 2 | Triple Hashes | 4/5 | Slight ambiguity with hash characters |
| 3 | JSON | 5/5 | Clear structure |
| 4 | Markdown Headers | 5/5 | Clear separation |
| 5 | Text Labels | 5/5 | Easy field identification |
| 6 | YAML-Style | 3/5 | Ambiguity between field and delimiter |
| 7 | Triple Quotes | 4/5 | Required attention to enclosing quotes |
| 8 | Nested XML | 4/5 | Added complexity but manageable |
| 9 | Pipe-Separated | 5/5 | Easy extraction |
| 10 | Natural Language | 3/5 | Ambiguity in free-form text |

---

### Comparison Summary

| Format | Claude Opus | GPT-4 | GPT-3.5 | Average |
|--------|-------------|-------|---------|---------|
| XML Tags | 5.0 | 5.0 | 5.0 | **5.0** |
| JSON | 5.0 | 5.0 | 5.0 | **5.0** |
| Markdown Headers | 4.0 | 5.0 | 5.0 | 4.7 |
| Text Labels | 3.5 | 5.0 | 5.0 | 4.5 |
| Triple Hashes | 4.0 | 5.0 | 4.0 | 4.3 |
| Nested XML | 5.0 | 4.0 | 4.0 | 4.3 |
| Pipe-Separated | 3.0 | 4.0 | 5.0 | 4.0 |
| Triple Quotes | 3.0 | 5.0 | 4.0 | 4.0 |
| YAML-Style | 3.5 | 5.0 | 3.0 | 3.8 |
| Natural Language | 2.0 | 3.0 | 3.0 | 2.7 |

---

### Benchmark Conclusion

**XML and JSON are universally the most reliable formats across all models.**

- Use **XML** for Claude-specific prompts and complex nested structures
- Use **JSON** when output needs to be machine-parsed
- Use **Text Labels with Colons** for simple, quick prompts
- **Avoid Natural Language** for structured data extraction

*Benchmark conducted: January 2026*