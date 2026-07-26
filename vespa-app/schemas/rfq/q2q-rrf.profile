rank-profile q2q-rrf inherits q2q-semantic {
    function title_bm25() {
        expression: bm25(title)
    }
    match-features {
        title_bm25
        title_sim
    }
    global-phase {
        expression: reciprocal_rank_fusion(title_bm25, title_sim)
        rerank-count: 200
    }
}
