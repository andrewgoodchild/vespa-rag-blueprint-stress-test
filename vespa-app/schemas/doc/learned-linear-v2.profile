rank-profile learned-linear-v2 inherits base-features {
    inputs {
        query(embedding) tensor<int8>(x[96])
        query(float_embedding) tensor<float>(x[768])
        query(intercept) double
        query(avg_top_3_chunk_sim_scores_param) double
        query(avg_top_3_chunk_text_scores_param) double
        query(bm25_chunks_param) double
        query(bm25_title_param) double
        query(max_chunk_sim_scores_param) double
        query(max_chunk_text_scores_param) double
        query(modified_freshness_param) double
    }
    rank-properties {
        freshness(modified_timestamp).maxAge: 94672800
    }
    function modified_freshness() {
        expression: freshness(modified_timestamp)
    }
    first-phase {
        expression {
            query(intercept) +
            query(avg_top_3_chunk_sim_scores_param) * avg_top_3_chunk_sim_scores() +
            query(avg_top_3_chunk_text_scores_param) * avg_top_3_chunk_text_scores() +
            query(bm25_title_param) * bm25(title) +
            query(bm25_chunks_param) * bm25(chunks) +
            query(max_chunk_sim_scores_param) * max_chunk_sim_scores() +
            query(max_chunk_text_scores_param) * max_chunk_text_scores() +
            query(modified_freshness_param) * modified_freshness()
        }
    }
    summary-features {
        top_3_chunk_sim_scores
    }
}
