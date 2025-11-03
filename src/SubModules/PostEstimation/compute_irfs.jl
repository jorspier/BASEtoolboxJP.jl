"""
    compute_irfs(
      exovars,
      gx,
      hx,
      XSS,
      ids;
      T = 1000,
      init_val = fill(0.01, length(exovars)),
      verbose = true,
      distribution = false,
      comp_ids = nothing,
      n_par = nothing,
    )

Computes impulse response functions (IRFs) for a given set of shocks to exogenous variables.

# Arguments

  - `exovars::Vector{Int64}`: Vector of positional indices of exogenous variables to which
    shocks are applied.
  - `gx::Matrix{Float64}`: Control matrix (states to controls equations)
  - `hx::Matrix{Float64}`: Transition matrix for states (state transition equations)
  - `XSS::Vector{Float64}`: Steady state values of the model variables.
  - `ids`: Indexes of the model variables.

# Keyword Arguments

  - `T::Int64`: Number of periods for which to compute IRFs.
  - `init_val::Vector{Float64}`: Initial value of the shock to each exogenous variable,
    defaults to 0.01 for all shocks.
  - `verbose::Bool`: Print progress to console.
  - `distribution::Bool`: Compute distributional IRFs, defaults to false.
  - `comp_ids`: Compression indices for the distribution, as created by
    `prepare_linearization`. Needed if `distribution` is true. Defaults to nothing.
  - `n_par`: Model parameters, needed for distributional IRFs. Defaults to nothing.

# Returns

  - `IRFs`, `IRFs_lvl`: 3D array of (level) IRFs for each exogenous variable, with
    dimensions:

      + 1: States and controls
      + 2: Time periods
      + 3: Exogenous variables

  - `IRFs_order`: Names of the exogenous variables and their indices in the IRFs array.
  - `IRFs_dist`: Dictionary containing the full distributional IRFs, if requested.
"""
function compute_irfs(
    exovars::Vector{Int64},
    gx::Matrix{Float64},
    hx::Matrix{Float64},
    XSS::Vector{Float64},
    ids;
    T::Int64 = 1000,
    init_val::Vector{Float64} = fill(0.01, length(exovars)),
    verbose::Bool = true,
    distribution::Bool = false,
    comp_ids = nothing,
    n_par = nothing,
)

    # If distributional IRFs are requested, check if all required arguments are provided
    if distribution && (isnothing(comp_ids) || isnothing(n_par))
        throw(
            ArgumentError(
                "Missing arguments: `comp_ids` and `n_par` are required for distributional IRFs.",
            ),
        )
    end

    # Compute the number of states and controls from gx and hx
    ncontrols = size(gx, 1)
    nstates = size(hx, 1)

    # Initialize IRFs for all selected exogenous variables
    IRFs = zeros(nstates + ncontrols, T, length(exovars))
    IRFs_lvl = similar(IRFs)
    if distribution
        IRFs_dist = initialize_distributional_dict(n_par, T, length(exovars))
    end

    # Store the shock names
    IRFs_order = [find_field_with_value(ids, exovars[i], false) for i = 1:length(exovars)]

    # Compute IRFs for each exogenous variable
    for (i, exovar) in enumerate(exovars)
        if verbose
            @printf "Computing IRFs for %s with index %d and initial condition of %f.\n" IRFs_order[i] exovar init_val[i]
        end
        if distribution
            IRFs[:, :, i], IRFs_lvl[:, :, i], dist_results = compute_irfs_inner(
                exovars[i],
                gx,
                hx,
                XSS,
                ids,
                T,
                nstates,
                ncontrols,
                init_val[i],
                distribution,
                comp_ids,
                n_par,
            )

            # Fill in here
            for (name, data) in pairs(dist_results)
                colon_tuple = (ntuple(_ -> Colon(), ndims(data))..., i)
                IRFs_dist[name][colon_tuple...] = data
            end

        else
            IRFs[:, :, i], IRFs_lvl[:, :, i] = compute_irfs_inner(
                exovars[i],
                gx,
                hx,
                XSS,
                ids,
                T,
                nstates,
                ncontrols,
                init_val[i],
                distribution,
                comp_ids,
                n_par,
            )
        end
    end

    if distribution
        IRFs_dist = Dict(
            key => convert(Array{Float64}, value) for
            (key, value) in IRFs_dist if value isa AbstractArray
        )
        return IRFs, IRFs_lvl, IRFs_order, IRFs_dist
    else
        return IRFs, IRFs_lvl, IRFs_order
    end
end

function initialize_distributional_dict(n_par, T, n_shocks)
    # Define the base dimensions (3D objects)
    dims_3d = (n_par.nb, n_par.nk, n_par.nh, T, n_shocks)

    # Define the 1D objects (e.g., PDF_b)
    dims_1d = Dict(
        "b" => (n_par.nb, T, n_shocks),
        "k" => (n_par.nk, T, n_shocks),
        "h" => (n_par.nh, T, n_shocks),
    )

    # Define the 2D objects (e.g., PDF_bk)
    dims_2d = Dict(
        "bk" => (n_par.nb, n_par.nk, T, n_shocks),
        "bh" => (n_par.nb, n_par.nh, T, n_shocks),
        "kh" => (n_par.nk, n_par.nh, T, n_shocks),
    )

    IRFs_dist = Dict{String,Any}()

    # Base 3D objects
    for name in ["Wb", "Wk", "PDF"]
        IRFs_dist[name] = zeros(dims_3d...)
    end

    # Marginal/Joint objects
    dist_to_dims = [
        ("PDF", dims_1d),
        ("Wb", dims_1d),
        ("Wk", dims_1d),
        ("PDF", dims_2d),
        ("Wb", dims_2d),
        ("Wk", dims_2d),
    ]
    for (key, dims_map) in dist_to_dims
        for (suffix, dims) in dims_map
            name = string(key, "_", suffix) # "PDF_b", "Wk_kh", etc.
            IRFs_dist[name] = zeros(dims...)
        end
    end

    return IRFs_dist
end

"""
    compute_irfs_inner(
        exovar,
        gx,
        hx,
        XSS,
        ids,
        T,
        nstates,
        ncontrols,
        init_val,
        distribution,
        comp_ids
    )

Computes impulse response functions (IRFs) for a given shock to a single exogenous variable.

See `compute_irfs` for arguments and return value descriptions.
"""
function compute_irfs_inner(
    exovar::Int64,
    gx::Matrix{Float64},
    hx::Matrix{Float64},
    XSS::Vector{Float64},
    ids,
    T::Int64,
    nstates::Int64,
    ncontrols::Int64,
    init_val::Float64,
    distribution::Bool,
    comp_ids,
    n_par,
)

    # Initialize matrices for states and controls
    S_t = zeros(nstates, T)
    C_t = zeros(ncontrols, T)

    # Initial conditions: states by assumption, controls as implied by gx and initial state
    S_t[exovar, 1] = init_val
    C_t[:, 1] = gx * S_t[:, 1]

    # Simulation: iterate forward
    for t = 2:T
        S_t[:, t] = hx * S_t[:, t - 1]
        C_t[:, t] = gx * S_t[:, t]
    end

    # Recompute levels for the original IRFs, as defined in macro @generate_equations
    original = [S_t; C_t]
    level = fill(NaN64, size(original))

    # Start with the aggregate variables
    idx = [getfield(ids, Symbol(j)) for j in aggr_names]
    idxSS = [getfield(ids, Symbol(j, "SS")) for j in aggr_names]
    level[idx, :] = exp.(XSS[idxSS] .+ original[idx, :])

    if distribution
        dist_results = compute_irfs_inner_distribution(original, ids, XSS, comp_ids, n_par)
        return original, level, dist_results
    else
        return original, level
    end
end

"""
    compute_irfs_inner_distribution(original, ids, XSS, comp_ids, n_par)

Reconstructs the full, uncompressed impulse responses for the marginal value functions (Wb,
Wk) and the joint probability distribution (PDF) from their compressed representations.

This function takes the 1D compressed impulse responses for all model variables and unpacks
them into their full-dimensional form for each period of the IRF. It follows the precise
logic of the `Fsys` function to combine steady-state values, transformation matrices (DCT
and Shuffle), and the IRF deviations to compute the dynamics of the entire state space.

# Arguments

  - `original::Array{Float64,2}`: A matrix `(n_vars x T)` containing the compressed IRFs for
    all variables.
  - `ids`: An object mapping variable names to their indices in `original` and `XSS`.
  - `XSS::Vector{Float64}`: The steady-state vector of the model.
  - `comp_ids`: An object containing the indices for compressed variables.
  - `n_par`: A struct with numerical parameters (e.g., grid sizes).

# Returns

  - `PDF::Array{Float64,4}`: The uncompressed IRF for the full joint probability
    distribution.
  - `Wb::Array{Float64,4}`, `Wk::Array{Float64,4}`: The uncompressed IRF for the marginal
    utility with respect to liquid/illiquid assets.
  - `PDF_b::Array{Float64,2}`, `PDF_k::Array{Float64,2}`, `PDF_h::Array{Float64,2}`: The
    uncompressed IRFs for the marginal PDFs of liquid assets, illiquid assets, and human
    capital, respectively.
  - `PDF_bk::Array{Float64,3}`, `PDF_bh::Array{Float64,3}`, `PDF_kh::Array{Float64,3}`: The
    uncompressed IRFs for the joint PDFs of liquid and illiquid assets, liquid assets and
    human capital, and illiquid assets and human capital, respectively.
  - `Wb_b::Array{Float64,2}`, `Wb_k::Array{Float64,2}`, `Wb_h::Array{Float64,2}`, `Wk_b::Array{Float64,2}`,
    `Wk_k::Array{Float64,2}`, `Wk_h::Array{Float64,2}`: The uncompressed IRFs for the
    marginal value functions with respect to liquid/illiquid assets, aggregated over the
    other two dimensions.
  - `Wb_bk::Array{Float64,3}`, `Wb_bh::Array{Float64,3}`, `Wb_kh::Array{Float64,3}`, `Wk_bk::Array{Float64,3}`,
    `Wk_bh::Array{Float64,3}`, `Wk_kh::Array{Float64,3}`: The uncompressed IRFs for the
    marginal value functions with respect to liquid/illiquid assets, aggregated over
    one dimension.
"""
function compute_irfs_inner_distribution(
    original::Array{Float64,2},
    ids,
    XSS::Vector{Float64},
    comp_ids,
    n_par,
)

    ## 1. Preamble & Setup

    # Unpack grid sizes and number of periods from parameters
    nb, nk, nh = n_par.nb, n_par.nk, n_par.nh
    nb_c, nk_c, nh_c = n_par.nb_copula, n_par.nk_copula, n_par.nh_copula
    T = size(original, 2)

    ## 2. Unpack Steady-State Values

    # Steady-state marginal value functions
    WbSS = XSS[ids.WbSS]
    WkSS = XSS[ids.WkSS]

    # Steady-state marginal probability density functions (PDFs)
    PDF_bSS = XSS[ids.distr_bSS]
    PDF_kSS = XSS[ids.distr_kSS]
    PDF_hSS = XSS[ids.distr_hSS]

    # Steady-state joint PDF
    PDFSS = reshape(XSS[ids.COPSS], (nb, nk, nh))

    # Steady-state marginal cumulative distribution functions (CDFs)
    CDF_bSS = cumsum(PDF_bSS[:])
    CDF_kSS = cumsum(PDF_kSS[:])
    CDF_hSS = cumsum(PDF_hSS[:])

    # Steady-state joint CDF
    CDFSS = pdf_to_cdf(PDFSS)

    # This is the uniform coordinate grid [0,1] for each dimension of the copula. The copula
    # deviation (COP_Dev) is defined on this grid.
    s_m_b = n_par.copula_marginal_b
    s_m_k = n_par.copula_marginal_k
    s_m_h = n_par.copula_marginal_h

    ## 3. Compute Transformation Matrices

    # DCT matrices for compressing/uncompressing the value functions
    DC = Array{Array{Float64,2},1}(undef, 3)
    DC[1] = mydctmx(nb)
    DC[2] = mydctmx(nk)
    DC[3] = mydctmx(nh)
    IDC = [DC[1]', DC[2]', DC[3]']

    # DCT matrices for compressing/uncompressing the copula
    DCD = Array{Array{Float64,2},1}(undef, 3)
    DCD[1] = mydctmx(nb_c)
    DCD[2] = mydctmx(nk_c)
    DCD[3] = mydctmx(nh_c)
    IDCD = [DCD[1]', DCD[2]', DCD[3]']

    # Shuffle matrix
    Γ = shuffleMatrix(PDFSS, nb, nk, nh)

    ## 4. Initialize Output Containers

    # Containers to store the full, uncompressed 4D IRF objects (state x state x state x
    # time)
    Wb = zeros(nb, nk, nh, T)
    Wk = zeros(nb, nk, nh, T)
    PDF = zeros(nb, nk, nh, T)

    ## 5. Main Loop: Reconstruct Full Dynamics for Each Period
    for t = 1:T

        ## a) Reconstruct Marginal Value Functions
        Wb[:, :, :, t] .= reshape(
            exp.(WbSS .+ uncompress(comp_ids[1], original[ids.Wb, t], DC, IDC)),
            (nb, nk, nh),
        )
        Wk[:, :, :, t] .= reshape(
            exp.(WkSS .+ uncompress(comp_ids[2], original[ids.Wk, t], DC, IDC)),
            (nb, nk, nh),
        )

        ## b) Reconstruct the Joint PDF

        # Step i: Uncompress the deviation of the copula from its steady state. This gives
        # the PDF of the deviation, defined on the coarse copula grid.
        θD = uncompress(comp_ids[3], original[ids.COP, t], DCD, IDCD)[:]
        COP_Dev_pdf = reshape(θD, (nb_c, nk_c, nh_c))

        # Step ii: Convert the deviation's PDF to a CDF for interpolation.
        COP_Dev_cdf = pdf_to_cdf(COP_Dev_pdf)

        # Step iii: Reconstruct the perturbed marginal PDFs and their corresponding CDFs for
        # period t.
        PDF_b = PDF_bSS .+ Γ[1] * original[ids.distr_b, t]
        CDF_b = cumsum(PDF_b)
        PDF_k = PDF_kSS .+ Γ[2] * original[ids.distr_k, t]
        CDF_k = cumsum(PDF_k)
        PDF_h = PDF_hSS .+ Γ[3] * original[ids.distr_h, t]
        CDF_h = cumsum(PDF_h)

        # Step iv: Construct the full joint CDF. This is done by taking the steady-state
        # joint CDF and adding the interpolated deviation. The result is evaluated at the
        # points defined by the perturbed marginal CDFs.
        Copula(x, y, z) =
            myinterpolate3(CDF_bSS, CDF_kSS, CDF_hSS, CDFSS, n_par.model, x, y, z) .+
            myinterpolate3(s_m_b, s_m_k, s_m_h, COP_Dev_cdf, n_par.model, x, y, z)
        CDF = Copula(CDF_b, CDF_k, CDF_h)

        # Step v: Convert the reconstructed joint CDF back to a PDF.
        PDF[:, :, :, t] .= cdf_to_pdf(CDF)
    end

    # Compute all marginals for Wb, Wk, and PDF
    all_marginals = Dict()

    # Function compute_all_marginals returns a Dict, which we merge with original dict
    merge!(all_marginals, compute_all_marginals(PDF, "PDF"))
    merge!(all_marginals, compute_all_marginals(Wb, "Wb"))
    merge!(all_marginals, compute_all_marginals(Wk, "Wk"))

    # Add the full 3D objects back in
    all_marginals["PDF"] = PDF
    all_marginals["Wb"] = Wb
    all_marginals["Wk"] = Wk

    return all_marginals
end

function compute_all_marginals(Base_Array::AbstractArray{Float64,4}, Base_Name::String)
    # Base_Array is a 4D array: (nb, nk, nh, T)

    # Store results in a NamedTuple or Dict to be merged later
    marginals = Dict{String,AbstractArray}()

    # Mapping from dimension set to the dimensions to sum over
    # (The dims in 'sum' are the dimensions to get rid of)
    dim_map = (
        b = (2, 3), # sum over k and h
        k = (1, 3), # sum over b and h
        h = (1, 2), # sum over b and k
        bk = (3,),  # sum over h
        bh = (2,),  # sum over k
        kh = (1,),  # sum over b
    )

    for (dim_symbol, dims_to_sum) in pairs(dim_map)
        # 1. Compute the marginal/joint sum
        marginal_array = dropdims(sum(Base_Array; dims = dims_to_sum); dims = dims_to_sum)

        # 2. Store with the appropriate symbol (e.g., "PDF_b", "Wk_kh")
        field_name = string(Base_Name, "_", dim_symbol)
        marginals[field_name] = marginal_array
    end

    return marginals
end
