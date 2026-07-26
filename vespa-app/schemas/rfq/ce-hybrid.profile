rank-profile ce-hybrid inherits ce-semantic {
    function bm25sum() {
        expression: bm25(title) + bm25(chunks)
    }
    # crude hybrid first phase purely to select the rerank window:
    # bm25 spans ~0-40, cosine ~0-0.5, so scale sim into the same range
    first-phase {
        expression: bm25sum + 40 * max_chunk_sim_scores
    }
    global-phase {
        rerank-count: 50
        expression: onnx(cross_encoder){d0:0,d1:0}
    }
}
