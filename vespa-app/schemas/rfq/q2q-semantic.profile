rank-profile q2q-semantic inherits base-features {
    # Library-model retrieval: rank by similarity of the incoming question to
    # the STORED question (title), ignoring the answer text entirely.
    function title_vec() {
        expression: unpack_bits(attribute(title_embedding))
    }
    function title_sim() {
        expression: sum(query(float_embedding) * title_vec) / (sqrt(sum(pow(title_vec, 2), x)) * sqrt(sum(pow(query(float_embedding), 2), x)))
    }
    first-phase {
        expression: title_sim
    }
    match-features {
        title_sim
    }
}
