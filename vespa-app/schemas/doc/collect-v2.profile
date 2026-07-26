rank-profile collect-v2 inherits base-features {
    rank-properties {
        freshness(modified_timestamp).maxAge: 94672800
    }
    function modified_freshness() {
        expression: freshness(modified_timestamp)
    }
    match-features {
        bm25(title)
        bm25(chunks)
        max_chunk_sim_scores
        max_chunk_text_scores
        avg_top_3_chunk_sim_scores
        avg_top_3_chunk_text_scores
        modified_freshness
    }
    first-phase {
        expression: bm25(title) + bm25(chunks) + max_chunk_sim_scores() + max_chunk_text_scores()
    }
}
