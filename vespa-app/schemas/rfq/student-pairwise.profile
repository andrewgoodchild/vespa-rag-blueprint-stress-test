rank-profile student-pairwise inherits base-features {
    # Distilled pairwise-linear student deployed natively: cheap hybrid
    # first-phase selects the window, the linear student reranks in second-phase.
    # No cross-encoder anywhere in the serving path.
    inputs {
        query(embedding) tensor<int8>(x[96])
        query(float_embedding) tensor<float>(x[768])
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
    function bm25sum() {
        expression: bm25(title) + bm25(chunks)
    }
    first-phase {
        expression: bm25sum + 40 * max_chunk_sim_scores
    }
    second-phase {
        rerank-count: 50
        expression {
            0.353475 * bm25(title) +
            0.268666 * bm25(chunks) +
            3.698572 * max_chunk_sim_scores +
            3.698572 * avg_top_3_chunk_sim_scores +
            6.713463 * title_sim +
            0.249344 * colbert_max_sim
        }
    }
}
