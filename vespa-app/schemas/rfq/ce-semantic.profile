rank-profile ce-semantic inherits base-features {
    inputs {
        query(q_tokens) tensor<float>(d0[32])
    }
    onnx-model cross_encoder {
        file: models/cross_encoder.onnx
        input input_ids: my_input_ids
        input attention_mask: my_attention_mask
        input token_type_ids: my_token_type_ids
    }
    function my_input_ids() {
        expression: tokenInputIds(256, query(q_tokens), attribute(ce_tokens))
    }
    function my_token_type_ids() {
        expression: tokenTypeIds(256, query(q_tokens), attribute(ce_tokens))
    }
    function my_attention_mask() {
        expression: tokenAttentionMask(256, query(q_tokens), attribute(ce_tokens))
    }
    first-phase {
        expression: max_chunk_sim_scores
    }
    global-phase {
        rerank-count: 50
        expression: onnx(cross_encoder){d0:0,d1:0}
    }
}
