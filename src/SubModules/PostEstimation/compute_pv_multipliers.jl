# Function to compute cumulative present-value multipliers
function compute_pv_multipliers(irf_matrix, IRFs_order, indexes, XSS, shock_var_name; max_horizon=80)
    
    # Find the index of the shock in the IRFs_order
    shock_idx = findfirst(==(shock_var_name), IRFs_order)
    if isnothing(shock_idx)
        error("Shock $shock_var_name not found in IRFs_order")
    end
    
    # Select the IRF matrix for this shock
    irf_matrix_shock = irf_matrix[:, :, shock_idx]
    
    # 1. Map the shock name to its respective IRF and Steady State indices
    if shock_var_name == :G
        irf_idx = indexes.G
        ss_idx = indexes.GSS
    elseif shock_var_name == :GI
        irf_idx = indexes.GI
        ss_idx = indexes.GISS
    else
        error("Unsupported shock variable.")
    end

    # 2. Extract Steady States to calculate the scaling ratio
    Y_SS = exp(XSS[indexes.YSS])
    Shock_SS = exp(XSS[ss_idx])
    ratio = Y_SS / Shock_SS

    # 3. Extract the steady-state real interest rate for discounting
    # Assuming RRL is your real liquid rate; subtract 1 for the net rate
    r_SS = exp(XSS[indexes.RRLSS]) - 1.0 

    # 4. Define the horizons you want to check (assuming quarterly data)
    horizons = [4, 8, 12, 20, 40, 80] # 1 Year, 2 Years, 3 Years, 5 Years, 10 Years
    results = Float64[]

    for h in horizons
        # Create a discounting vector: (1 + r)^(-t)
        discount = [1.0 / ((1.0 + r_SS)^(t-1)) for t in 1:h]

        # Calculate the cumulative discounted sum of the IRFs
        sum_Y = sum(irf_matrix_shock[indexes.Y, 1:h] .* discount)
        sum_Shock = sum(irf_matrix_shock[irf_idx, 1:h] .* discount)

        # Apply the scaling ratio to get the euro-for-euro multiplier
        mult = ratio * (sum_Y / sum_Shock)
        push!(results, mult)
    end

    # 5. Build and return the formatted table
    df = DataFrame(
        Horizon_Years = horizons ./ 4,
        Cumulative_Multiplier = round.(results, digits=2)
    )
    
    return df
end