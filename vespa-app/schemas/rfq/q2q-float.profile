rank-profile q2q-float {
    # Gap 7 control: the library model on float title embeddings with the
    # standard asymmetric prefixes (search_query: vs search_document:).
    # Scores every doc in the (tenant-filtered) candidate set, so the only
    # difference from q2q-sym is the document-side prefix.
    inputs {
        query(float_embedding) tensor<float>(x[768])
    }
    function title_sim() {
        expression: sum(query(float_embedding) * attribute(title_emb_f)) / (sqrt(sum(pow(attribute(title_emb_f), 2), x)) * sqrt(sum(pow(query(float_embedding), 2), x)))
    }
    first-phase {
        expression: title_sim
    }
}
