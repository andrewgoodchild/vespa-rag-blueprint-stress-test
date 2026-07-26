rank-profile colbert-maxsim {
    inputs {
        query(qt) tensor<float>(qt{}, x[128])
    }
    function unpack() {
        expression: unpack_bits(attribute(colbert))
    }
    function max_sim() {
        expression {
            sum(
                reduce(
                    sum(query(qt) * unpack(), x),
                    max, dt
                ),
                qt
            )
        }
    }
    first-phase {
        expression: max_sim
    }
}
