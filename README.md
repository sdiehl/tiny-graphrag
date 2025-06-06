<p align="center">
  <img src=".github/logo.jpeg" alt="Tiny GraphRAG Logo" width="512"/>
</p>

# Tiny GraphRAG

A tiny 1000 line implementation of the GraphRAG algorithm using only language
models that run locally. This implementation is designed to be easy to be
easily understandable, hackable and forkable and not dependent on any
framework.

Notably it does not use OpenAI or any commercial LLM providers and can be run
locally.

| Component         | Implementation                                    |
|------------------|--------------------------------------------------|
| Vector Database  | [pgvector](https://github.com/pgvector/pgvector) |
| Embedding Model  | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) |
| Language Model   | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Meta-Llama-3.2-3B) |
| Entity Extractor | [GLiNER](https://github.com/urchade/GLiNER)     |
| Relation Extract | [GLiREL](https://github.com/jackboyla/GLiREL)   |
| Graph Database   | [networkx](https://github.com/networkx/networkx) |
| Inference        | [llama-cpp](https://github.com/abetlen/llama-cpp-python) |

## Usage

First clone the repository:

```shell
git clone https://github.com/sdiehl/tiny-graphrag.git
cd tiny-graphrag
```

To install the dependencies run:

```shell
poetry install
```

To setup the local Postgres vector database in a docker container run:

```shell
docker-compose up -d
```

Then create the database tables with:

```shell
poetry run graphrag init
```

To build the graph and embeddings for a document run:

```shell
poetry run graphrag build data/Barack_Obama.txt
```

This will run for some time and when complete with the metadata.

```shell
Success! Document stored with ID: 1
Graph saved to: graphs/1_graph.pkl
```

Then we can query the database with the following modes:

- **Local Search**: Uses the graph structure to find relevant entities and their relationships, providing context-aware results based on the document's knowledge graph.
- **Global Search**: Analyzes document communities and their summaries to provide high-level insights and thematic understanding.
- **Naive RAG**: Combines vector similarity and keyword matching using Reciprocal Rank Fusion to find relevant text chunks directly.

Choose the search mode based on your needs:

- Use `local` for specific factual queries where relationships between entities matter
- Use `global` for thematic questions or high-level document understanding
- Use `naive` for simple queries where direct text chunk matching is sufficient

```shell
# Local search
poetry run graphrag query local --graph graphs/1_graph.pkl "What did Barack Obama study at Columbia University?"

# Global search
poetry run graphrag query global --graph graphs/1_graph.pkl  --doc-id 1 "What are the main themes of this document?"

# Naive RAG
poetry run graphrag query naive "What efforts did Barack Obama due to combat climate change?"
```

## Performance

The pipeline performance can be improved by using either Apple Silicon or a
Nvidia GPU to drive the embedding and language model inference. This will fallback
to CPU if no other options are available but the addition of dedicated hardware
will improve the performance.

**Apple Silicon**

To run with acceleration you'll need the `apple` extras with `thinc-apple-ops`
installed.

```bash
poetry install --with apple
```

**Nvidia GPU**

To run with acceleration you'll need the `nvidia` extras to run `transformers`
and `flash-attn`.

```bash
poetry install --with nvidia
```

## Library

You can also this library directly.

```python
from tiny_graphrag import store_document, QueryEngine, init_db

# Initialize the database first
engine = init_db("postgresql://admin:admin@localhost:5432/tiny-graphrag")

# Process and store a document
doc_id, graph_path = store_document(
    filepath="data/Barack_Obama.txt",
    title="Barack Obama Wikipedia",
    engine=engine
)

# Create query engine with the database connection
query_engine = QueryEngine(engine)

# Local search
result = query_engine.local_search(
    query="What did Barack Obama study at Columbia University?",
    graph_path=graph_path
)

# Global search
result = query_engine.global_search(
    query="What are the main themes of this document?",
    doc_id=doc_id
)

# Naive RAG
result = query_engine.naive_search(
    query="What efforts did Barack Obama due to combat climate change?"
)
```

## Custom Entities and Relations


You can customize the entity types and relation types when storing documents. Here's an example using geographic entities:

```python
# Define custom entity types
geography_entity_types = [
    "City",
    "Country",
    "Landmark",
    "River"
]

# Define custom relation types with constraints
geography_relation_types = {
    "capital of": {
        "allowed_head": ["City"],
        "allowed_tail": ["Country"]
    },
    "flows through": {
        "allowed_head": ["River"],
        "allowed_tail": ["City", "Country"]
    },
    "located in": {
        "allowed_head": ["City", "Landmark"],
        "allowed_tail": ["Country"]
    }
}

# Store document with custom types
doc_id, graph_path = store_document(
    filepath="data/geography.txt",
    title="World Geography",
    engine=engine,
    entity_types=geography_entity_types,
    relation_types=geography_relation_types,
)
```

The constraints ensure that relationships make semantic sense, e.g., only cities can be capitals of countries.

## Modules

* `main.py` - CLI entry point and command handlers
* `build.py` - Core document processing pipeline for constructing knowledge graphs from text
* `query.py` - Query processing and response generation
* `chunking.py` - Text segmentation and preprocessing utilities
* `communities.py` - Graph community segmentation
* `config.py` - Configuration settings and environment variables
* `db.py` - Database connection and vector store management
* `entity_types.py` - Entity type definitions and classification
* `rel_types.py` - Relationship type definitions between entities
* `extract.py` - Named entity recognition and relation extraction
* `prompts.py` - LLM prompt templates
* `search.py` - Vector similarity search and ranking algorithms
* `visualize.py` - Graph and community visualization

License
-------

MIT License
