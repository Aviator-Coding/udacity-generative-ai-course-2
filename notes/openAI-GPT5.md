TEST 1:
- Format: XML (Simple Tags)
- Name: Alice Chen
- Role: Senior DevOps Engineer
- Task: Deploy Kubernetes cluster with Rook-Ceph storage
- Confidence: 5/5
- Notes: Clear XML tags with one-to-one mapping; no ambiguity.

TEST 2:
- Format: Custom Delimiters (Triple Hashes)
- Name: Bob Martinez
- Role: Database Administrator
- Task: Migrate PostgreSQL 14 to PostgreSQL 17 with pgvector
- Confidence: 5/5
- Notes: Explicit section delimiters; straightforward parsing.

TEST 3:
- Format: JSON
- Name: Carol Wu
- Role: ML Engineer
- Task: Build MCP server for AWS Athena integration
- Confidence: 5/5
- Notes: Valid JSON with clear keys; unambiguous.

TEST 4:
- Format: Markdown Headers
- Name: David Park
- Role: Platform Engineer
- Task: Configure FluxCD GitOps pipeline for production
- Confidence: 5/5
- Notes: Markdown headers clearly label each field.

TEST 5:
- Format: Labeled Text (Colon-Separated)
- Name: Elena Volkov
- Role: Security Architect
- Task: Implement zero-trust network architecture
- Confidence: 5/5
- Notes: Colon-separated labels are explicit and consistent.

TEST 6:
- Format: YAML-Style Key/Value with Separators
- Name: Frank Okonkwo
- Role: Data Engineer
- Task: Deploy TiDB Operator on Kubernetes cluster
- Confidence: 5/5
- Notes: YAML-like structure is clear despite repeated separators.

TEST 7:
- Format: Quoted Labeled Text Block
- Name: Grace Kim
- Role: SRE Lead
- Task: Debug Redis cluster failover issues
- Confidence: 5/5
- Notes: Labels inside quoted block remove ambiguity.

TEST 8:
- Format: XML (Nested / Complex)
- Name: Henry Liu
- Role: Infrastructure Lead
- Task: Troubleshoot load balancer configuration
- Confidence: 4/5
- Notes: Task required extracting nested `<description>` element rather than a direct tag.

TEST 9:
- Format: Pipe-Separated Table
- Name: Ivan Petrov
- Role: Backend Developer
- Task: Optimize API response times
- Confidence: 4/5
- Notes: Correct parsing depends on trusting header-to-column alignment.

TEST 10:
- Format: Natural Language (Free-Form Text)
- Name: Julia Santos
- Role: Cloud Architect
- Task: Design the multi-region disaster recovery system for our infrastructure
- Confidence: 3/5
- Notes: Required semantic inference and normalization of phrasing.
