"""
Dummy file for RANK model to satisfy the PreprocessInputs.jl file check.
RANK does not require household argument packaging for Endogenous Grid Methods.
"""

function compute_args_hh_prob_ss(K, m_par, n_par)
    # Return an empty array. 
    args_hh_prob = Float64[]
    return args_hh_prob
end