import json
import pickle
from typing import Any, List, Optional, Tuple

import networkx as nx
from llama_cpp import Llama
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from tiny_graphrag.chunking import chunk_document, model
from tiny_graphrag.communities import build_communities
from tiny_graphrag.config import MODEL_ID, MODEL_REPO
from tiny_graphrag.db import Community, Document, DocumentChunk
from tiny_graphrag.entity_types import MIN_ENTITY_TYPES
from tiny_graphrag.extract import extract_rels
from tiny_graphrag.prompts import COMMUNITY_COMBINE, COMMUNITY_SUMMARY
from tiny_graphrag.rel_types import DEFAULT_RELS_LIST
from tiny_graphrag.visualize import visualize, visualize_communities


def process_document(
    filepath: str,
    max_chunks: int = -1,
    entity_types: List[str] = MIN_ENTITY_TYPES,
    relation_types: List[str] = DEFAULT_RELS_LIST,
) -> Tuple[List[Tuple[str, Any]], nx.Graph]:
    """Process a document and return chunks and graph."""
    # Read and chunk document
    page_text = open(filepath).read()
    page_chunks = chunk_document(page_text)

    # Build graph
    g = nx.Graph()

    for chunk_text, _embedding in tqdm(page_chunks[:max_chunks]):
        extraction = extract_rels(chunk_text, entity_types, relation_types)

        for ent in extraction.entities:
            g.add_node(ent[0], label=ent[1])

        for rel in extraction.relations:
            g.add_edge(rel[0], rel[2], label=rel[1], source_chunk=chunk_text)

    return page_chunks, g


def generate_community_summary(
    llm: Llama,
    community: List[Tuple[str, str, str, str]],
    max_triples: int = 30,
    temperature: float = 0.2,
) -> str:
    """Generate a summary for a community, chunking if needed."""
    # Chunk the community into smaller pieces if too large
    chunks = [
        community[i : i + max_triples] for i in range(0, len(community), max_triples)
    ]

    summaries: List[str] = []
    for chunk in chunks:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": COMMUNITY_SUMMARY.format(community=chunk),
                }
            ],
            temperature=temperature,
        )
        summaries.append(response["choices"][0]["message"]["content"])  # type: ignore

    # If we had multiple chunks, combine them
    if len(summaries) > 1:
        combined = " ".join(summaries)
        # Generate a final summary of the combined text
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": COMMUNITY_COMBINE.format(combined=combined),
                }
            ],
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]  # type: ignore

    return summaries[0]


def store_document(
    filepath: str,
    engine: Engine,
    title: Optional[str] = None,
    max_chunks: int = -1,
    temperature: float = 0.2,
) -> Tuple[int, str]:
    """Store document in database and save graph."""
    # Initialize database and LLM
    session_local = sessionmaker(bind=engine)

    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO, filename=MODEL_ID, local_dir=".", verbose=False, n_ctx=4096
    )

    with session_local() as session:
        # Process document
        chunks, graph = process_document(filepath, max_chunks)

        # Store document
        doc = Document(content=open(filepath).read(), title=title or filepath)
        session.add(doc)
        session.flush()  # Get document ID

        # Store chunks
        for chunk_text, embedding in chunks:
            chunk = DocumentChunk(
                document_id=doc.id,
                content=chunk_text,
                embedding=embedding,
                chunk_index=0,
            )
            session.add(chunk)

        # Build and store communities
        community_result = build_communities(graph)

        # Generate and store community summaries
        for community in community_result.communities:
            # Generate summary with chunking
            summary = generate_community_summary(
                llm, community, temperature=temperature
            )

            # Get embedding for summary
            embedding = model.encode(
                summary, convert_to_numpy=True, show_progress_bar=False
            )

            # Store community
            community_obj = Community(
                document_id=doc.id,
                content=summary,
                embedding=embedding,
                nodes=json.dumps(
                    [
                        n
                        for n, c in community_result.node_community_map.items()
                        if c == community_result.communities.index(community)
                    ]
                ),
            )
            session.add(community_obj)

        # Save graph
        graph_path = f"graphs/{doc.id}_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)

        # Visualize graphs
        visualize(graph, f"graphs/{doc.id}_graph.png")
        visualize_communities(
            graph, community_result.communities, f"graphs/{doc.id}_communities.png"
        )

        session.commit()
        return doc.id, graph_path
