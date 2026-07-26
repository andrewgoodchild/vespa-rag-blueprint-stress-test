rank-profile semantic-only inherits base-features {
    first-phase {
        expression: max_chunk_sim_scores()
    }
    summary-features {
        top_3_chunk_sim_scores
    }
}
