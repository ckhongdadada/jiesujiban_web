from src.jsjb.retrieval.components import SemanticChunker


class DummyEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            if "垃圾" in text or "清运" in text:
                vectors.append([1.0, 0.0])
            elif "电梯" in text or "维保" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.7, 0.7])
        return vectors


def test_semantic_chunker_splits_on_embedding_boundary():
    doc = {
        "id": "doc-1",
        "title": "混合诉求案例",
        "content": "垃圾清运不及时，桶站满冒。居民反映异味明显。电梯近期频繁故障。维保单位已经安排检修。",
        "district": "朝阳区",
    }
    chunker = SemanticChunker(
        chunk_size=400,
        chunk_overlap=0,
        min_chunk_size=8,
        enable_semantic_chunking=True,
        semantic_threshold=0.8,
    )

    chunks = chunker.chunk_document(doc, embedding_model=DummyEmbeddingModel())

    assert len(chunks) >= 2
    assert "垃圾" in chunks[0].content
    assert any("电梯" in chunk.content for chunk in chunks[1:])
    assert all(chunk.start_char <= chunk.end_char for chunk in chunks)


def test_semantic_chunker_falls_back_without_embedding_model():
    doc = {
        "id": "doc-2",
        "title": "兜底案例",
        "content": "第一段说明垃圾清运问题。\n\n第二段说明物业维修问题。\n\n第三段说明后续办理结果。",
    }
    chunker = SemanticChunker(
        chunk_size=20,
        chunk_overlap=0,
        min_chunk_size=5,
        enable_semantic_chunking=True,
    )

    chunks = chunker.chunk_document(doc, embedding_model=None)

    assert chunks
    assert all(chunk.doc_id == "doc-2" for chunk in chunks)
