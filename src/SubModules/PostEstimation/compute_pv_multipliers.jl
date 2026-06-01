"""
    compute_pv_multipliers(irf_matrix, IRFs_order, indexes, XSS, shock_var_name; ...)
    compute_pv_multipliers(irf_matrix, IRFs_order, ids, ss_dict, shock_var_name; ...)

Compute cumulative (undiscounted) fiscal multipliers for one or more response variables.

Returns a `DataFrame` with one row per horizon. Columns are named after each entry in
`response_vars`; an optional `Model` column is added when `model_name ≠ ""`.

The multiplier for variable X at horizon h is

    M_X(h) = (X_SS / GI_SS) × [Σ_{t=1}^{h} irf_X(t)] / [Σ_{t=1}^{h} irf_GI(t)]

For output/consumption variables this equals the standard Ramey-Zubairy (2018) cumulative
multiplier ΔX/ΔGI (same units). For Gini/share variables it gives absolute-unit change per
unit of GI spending; multiply by Y_SS/GI_SS to normalise to "per 1 % of GDP stimulus".

## Method 1 — use inside main.jl where `sr_full` is available

    compute_pv_multipliers(irf_matrix, IRFs_order, indexes, XSS, shock_var_name; ...)

- `indexes` : `IndexStruct` from `sr_full.indexes_r`; must have both plain and `*SS` fields.
- `XSS`     : full steady-state vector `sr_full.XSS`.

## Method 2 — use in standalone scripts that load from JLD2

    compute_pv_multipliers(irf_matrix, IRFs_order, ids, ss_dict, shock_var_name; ...)

- `ids`     : `NamedTuple` mapping variable names to row indices in `irf_matrix`
              (e.g. `ids_hank` from `IRF_comparison.jl`).
- `ss_dict` : `Dict{Symbol,Float64}` with at least `:GI` and one key per entry in
              `response_vars`.

## Keyword arguments

- `response_vars` : variables to compute multipliers for (default `[:Y, :C, :GiniW, :GiniI]`).
- `horizons`      : horizons in quarters (default `[4, 8, 12, 20, 40, 80]`).
- `model_name`    : label added as a `Model` column when non-empty (default `""`).
- `scale_by_ss`   : if `true` (default), multiply by `X_SS / GI_SS` to convert log-deviations
                    into level multipliers (ΔX/ΔGI, same units). Set to `false` to report
                    the pure elasticity `Σ irf_X / Σ irf_GI`, which is comparable across
                    specifications with different steady states.
"""
function compute_pv_multipliers(
    irf_matrix::Array{Float64,3},
    IRFs_order::Vector{Symbol},
    indexes,
    XSS::Vector{Float64},
    shock_var_name::Symbol;
    response_vars::Vector{Symbol} = [:Y, :C, :GiniW, :GiniI],
    horizons::Vector{Int}         = [4, 8, 12, 20, 40, 80],
    model_name::String            = "",
    scale_by_ss::Bool             = true,
)
    shock_ss_field = shock_var_name == :GI ? :GISS : :GSS
    SS_shock = exp(XSS[getfield(indexes, shock_ss_field)])

    ss_vals = Dict{Symbol,Float64}()
    for var in response_vars
        ss_field = Symbol(var, "SS")
        hasproperty(indexes, ss_field) ||
            error("Field :$ss_field not found in indexes; cannot look up SS value for :$var")
        ss_vals[var] = exp(XSS[getfield(indexes, ss_field)])
    end

    fiscal_row = getfield(indexes, shock_var_name)
    row_of(v)  = getfield(indexes, v)

    return _cumulative_multipliers(
        irf_matrix, IRFs_order, shock_var_name,
        fiscal_row, SS_shock, ss_vals, row_of,
        response_vars, horizons, model_name, scale_by_ss,
    )
end

# Method 2: ids (row indices only) + ss_dict (SS scalars); used in IRF_comparison.jl
function compute_pv_multipliers(
    irf_matrix::Array{Float64,3},
    IRFs_order::Vector{Symbol},
    ids,
    ss_dict::Dict{Symbol,Float64},
    shock_var_name::Symbol;
    response_vars::Vector{Symbol} = [:Y, :C, :GiniW, :GiniI],
    horizons::Vector{Int}         = [4, 8, 12, 20, 40, 80],
    model_name::String            = "",
    scale_by_ss::Bool             = true,
)
    haskey(ss_dict, :GI) || error("ss_dict must contain key :GI (SS level of government investment)")
    for var in response_vars
        haskey(ss_dict, var)    || error("ss_dict missing key :$var")
        hasproperty(ids, var)   || error("ids missing field :$var (row index in irf_matrix)")
    end

    SS_shock   = ss_dict[:GI]
    fiscal_row = getfield(ids, shock_var_name)
    row_of(v)  = getfield(ids, v)

    return _cumulative_multipliers(
        irf_matrix, IRFs_order, shock_var_name,
        fiscal_row, SS_shock, ss_dict, row_of,
        response_vars, horizons, model_name, scale_by_ss,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Shared inner implementation
# ──────────────────────────────────────────────────────────────────────────────
function _cumulative_multipliers(
    irf_matrix::Array{Float64,3},
    IRFs_order::Vector{Symbol},
    shock_var_name::Symbol,
    fiscal_row::Int,
    SS_shock::Float64,
    ss_vals,
    row_of::Function,
    response_vars::Vector{Symbol},
    horizons::Vector{Int},
    model_name::String,
    scale_by_ss::Bool,
)
    shock_col = findfirst(==(shock_var_name), IRFs_order)
    isnothing(shock_col) && error("Shock :$shock_var_name not found in IRFs_order")
    irf = view(irf_matrix, :, :, shock_col)

    results = Dict(v => Float64[] for v in response_vars)

    for h in horizons
        sum_fiscal = sum(irf[fiscal_row, 1:h])
        for var in response_vars
            sum_var = sum(irf[row_of(var), 1:h])
            scaling = scale_by_ss ? ss_vals[var] / SS_shock : 1.0
            push!(results[var], scaling * sum_var / sum_fiscal)
        end
    end

    df = DataFrame(:Horizon_Years => horizons ./ 4)
    for var in response_vars
        df[!, var] = round.(results[var]; digits = 4)
    end
    isempty(model_name) || (df[!, :Model] .= model_name)
    return df
end
