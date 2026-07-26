rank-profile ce-q2q inherits ce-semantic {
    # Industry stack: library-model recall (question-to-question similarity)
    # feeding the cross-encoder rerank.
    function title_vec() {
        expression: unpack_bits(attribute(title_embedding))
    }
    function title_sim() {
        expression: sum(query(float_embedding) * title_vec) / (sqrt(sum(pow(title_vec, 2), x)) * sqrt(sum(pow(query(float_embedding), 2), x)))
    }
    first-phase {
        expression: title_sim
    }
    global-phase {
        rerank-count: 50
        expression: onnx(cross_encoder){d0:0,d1:0}
    }
}
