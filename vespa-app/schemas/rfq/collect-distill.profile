rank-profile collect-distill inherits ce-hybrid {
    # Teacher = the cross-encoder (global-phase relevance, inherited from ce-hybrid).
    # match-features export the cheap student features for the returned hits.
    inputs {
        query(qt) tensor<float>(qt{}, x[128])
    }
    function title_vec() {
        expression: unpack_bits(attribute(title_embedding))
    }
    function title_sim() {
        expression: sum(query(float_embedding) * title_vec) / (sqrt(sum(pow(title_vec, 2), x)) * sqrt(sum(pow(query(float_embedding), 2), x)))
    }
    function colbert_unpack() {
        expression: unpack_bits(attribute(colbert))
    }
    function colbert_max_sim() {
        expression: sum(reduce(sum(query(qt) * colbert_unpack(), x), max, dt), qt)
    }
    match-features {
        bm25(title)
        bm25(chunks)
        max_chunk_sim_scores
        avg_top_3_chunk_sim_scores
        title_sim
        colbert_max_sim
    }
}
