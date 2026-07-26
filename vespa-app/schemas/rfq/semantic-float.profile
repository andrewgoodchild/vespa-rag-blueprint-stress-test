rank-profile semantic-float inherits base-features {
    # Full-precision float doc embeddings (vs the int8 binary in semantic-only).
    # Same float query; only the stored doc representation differs.
    function chunk_f_dot() {
        expression: reduce(query(float_embedding) * attribute(chunk_emb_f), sum, x)
    }
    function chunk_f_norm() {
        expression: sqrt(sum(pow(attribute(chunk_emb_f), 2), x))
    }
    function chunk_f_sim() {
        expression: chunk_f_dot() / (chunk_f_norm() * sqrt(sum(pow(query(float_embedding), 2), x)))
    }
    function max_chunk_f_sim() {
        expression: reduce(chunk_f_sim(), max, chunk)
    }
    first-phase {
        expression: max_chunk_f_sim()
    }
}
