rank-profile q2q-sym {
    # Gap 7 treatment: identical to q2q-float except the stored question was
    # embedded with the QUERY prefix (title_emb_sym) — symmetric conditioning
    # for the symmetric question-vs-question task.
    inputs {
        query(float_embedding) tensor<float>(x[768])
    }
    function title_sim() {
        expression: sum(query(float_embedding) * attribute(title_emb_sym)) / (sqrt(sum(pow(attribute(title_emb_sym), 2), x)) * sqrt(sum(pow(query(float_embedding), 2), x)))
    }
    first-phase {
        expression: title_sim
    }
}
