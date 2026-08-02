rank-profile q2q-ft {
    # Gap 5: the library model served with the contrastively fine-tuned MiniLM
    # (benchmark/gap5_export_minilm.py; trained on Gap 2's paraphrase pairs).
    inputs {
        query(mini_embedding) tensor<float>(x[384])
    }
    first-phase {
        expression: closeness(field, title_emb_ftmini)
    }
}
