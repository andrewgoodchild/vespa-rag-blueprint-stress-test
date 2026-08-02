rank-profile q2q-mini {
    # Gap 5: the library model served with base MiniLM instead of nomic.
    inputs {
        query(mini_embedding) tensor<float>(x[384])
    }
    first-phase {
        expression: closeness(field, title_emb_mini)
    }
}
