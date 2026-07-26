rank-profile bm25-only inherits base-features {
    first-phase {
        expression: bm25(title) + bm25(chunks)
    }
    summary-features {
        top_3_chunk_text_scores
    }
}
