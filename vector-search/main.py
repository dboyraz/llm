from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

from embedder import Embedder

QUERY = "How does approximate nearest neighbor search work?"
TARGET = "02-vector-search/lessons/07-sqlitesearch-vector.md"

if __name__ == "__main__":
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]

    doc = next(d for d in documents if d["filename"] == TARGET)

    embedder = Embedder()
    q = embedder.encode(QUERY)            # (384,) normalized
    d = embedder.encode(doc["content"])   # (384,) normalized

    # Q1: first value of the query embedding
    print(f"v[0] = {q[0]}")

    # Q2: vectors are normalized, so cosine similarity is just the dot product
    similarity = float(q @ d)
    print(f"cosine similarity: {similarity}")

    # Q3: chunk the pages (size=2000, step=1000), embed every chunk's content into
    # a matrix X, score the Q1 query against all chunks, report the top chunk's file.
    chunks = chunk_documents(documents, size=2000, step=1000)
    X = embedder.encode_batch([c["content"] for c in chunks])  # (N, 384)
    scores = X.dot(q)                                          # cosine sim per chunk
    best = int(scores.argmax())
    print(f"highest-scoring file: {chunks[best]['filename']}  (score {float(scores[best]):.4f})")

    # Q4: index the chunk vectors with minsearch VectorSearch and run a new query.
    index = VectorSearch()
    index.fit(X, chunks)
    q4_query = "What metric do we use to evaluate a search engine?"
    results = index.search(embedder.encode(q4_query), num_results=5)
    print(f"first result file: {results[0]['filename']}")

    # Q5: text search vs vector search for the same query. Index the same chunks
    # with a text Index (content field) and compare top-5 filenames against vector.
    q5_query = "How do I store vectors in PostgreSQL?"

    vec_results = index.search(embedder.encode(q5_query), num_results=5)
    vec_files = [r["filename"] for r in vec_results]

    text_index = Index(text_fields=["content"])
    text_index.fit(chunks)
    text_results = text_index.search(q5_query, num_results=5)
    text_files = [r["filename"] for r in text_results]

    only_in_vector = [f for f in vec_files if f not in text_files]
    print("vector top5:", vec_files)
    print("text   top5:", text_files)
    print("in vector but not text:", only_in_vector)

    # Q6: hybrid search via Reciprocal Rank Fusion of the vector and text lists.
    def rrf(result_lists, k=60, num_results=5):
        scores = {}
        docs = {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                key = (doc["filename"], doc["start"])
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docs[key] for key in ranked[:num_results]]

    q6_query = "How do I give the model access to tools?"
    vector_results = index.search(embedder.encode(q6_query), num_results=5)
    text_results = text_index.search(q6_query, num_results=5)
    fused = rrf([vector_results, text_results])
    print("RRF first result:", fused[0]["filename"])
