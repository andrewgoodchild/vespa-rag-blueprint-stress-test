rank-profile rrf-hybrid inherits base-features {
    function bm25sum() {
        expression: bm25(title) + bm25(chunks)
    }
    function semscore() {
        expression: max_chunk_sim_scores
    }
    first-phase {
        expression: max_chunk_sim_scores
    }
    # RRF inputs must be computed content-side and shipped as match-features;
    # evaluating chunk-tensor expressions in the container global-phase fails.
    match-features {
        bm25sum
        semscore
    }
    global-phase {
        expression: reciprocal_rank_fusion(bm25sum, semscore)
        rerank-count: 200
    }
}
