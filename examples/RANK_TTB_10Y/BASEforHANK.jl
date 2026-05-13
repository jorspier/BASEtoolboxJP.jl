module BASEforHANK

if !Sys.isapple() # issues encountered when using mkl with macos + more than 1 thread
    using MKL
end

using Plots, VegaLite, StatsPlots,  OrderedCollections, JLD2, FileIO, DataFrames, CSV, LaTeXStrings, JSON, CodecZlib, Parameters, Setfield, Flatten, FieldMetadata

using LinearAlgebra, SparseArrays, BlockDiagonals, CategoricalArrays, Random, MCMCChains, Distributions, Roots,  ForwardDiff, Optim, BenchmarkTools

using Statistics, PrettyTables, Colors, FiniteDiff, PositiveFactorizations

# using MatrixEquations, ProximalOperators,KrylovKit,SpecialFunctions,FFTW
using MatrixEquations: lyapd
using ProximalOperators: prox!, IndPSD
using KrylovKit: eigsolve
using SpecialFunctions: erf
using FFTW: dct, ifft

import Flatten: flattenable
export ModelParameters, NumericalParameters, EstimationSettings,
    SteadyResults, LinearResults, EstimResults,
    compute_steadystate, linearize_full_model, model_reduction, update_model,
    estimation_prep, montecarlo, mode, metaflatten, prior, compare_2_linearizations,
    reduction_quality, reduction_quality_seq, 
    compute_irfs_vardecomp, plot_irfs,  compute_hist_decomp, 
    plot_vardecomp, compute_bcfreq_vardecomp, compute_vardecomp_bounds, 
    @set!, jldsave, @load,
    @writeXSS, @make_fn, @make_fnaggr, @make_struct, @make_struct_aggr,
    @generate_equations

include("1_Model/input_aggregate_names.jl")

# ------------------------------------------------------------------------------
## Define Functions
# ------------------------------------------------------------------------------
include("1_Model/Parameters.jl")
include("3_NumericalBasics/Structs.jl")
include("6_Estimation/prior.jl")

e_set = EstimationSettings(shock_names=shock_names)
@make_struct IndexStruct
@make_struct_aggr IndexStructAggr

include("2_includeLists/include_NumericalBasics.jl")
include("2_includeLists/include_HetAgentsFcns.jl")
include("2_includeLists/include_LinearizationFunctions.jl")
include("2_includeLists/include_Estimation.jl")
include("2_includeLists/include_PostEstimation.jl")


@make_fn produce_indexes
@make_fnaggr produce_indexes_aggr


@doc raw"""
    compute_steadystate()

Compute steady state.

# Returns
`struct` `SteadyResults`, containing returns of [`find_SS()`](@ref)
"""
function compute_steadystate(m_par)
    #Calculate steady state capital stock
    KSS, n_par, m_par = find_steadystate(m_par)

    # Prepare steadys state information for linearization
    XSS, XSSaggr, indexes, indexes_r, indexes_aggr, n_par, m_par = 
        prepare_linearization(KSS, n_par, m_par)

    return SteadyResults(XSS, XSSaggr, indexes, indexes_r, indexes_aggr,
        n_par, m_par, state_names, control_names)
end

@doc raw"""
    linearize_full_model()

Linearize the full model (i.e. including idiosyncratic states and controls) around the steady state, and solves
using [`LinearSolution()`](@ref).

# Returns
`struct` `LinearResults`, containing
- `A::Array{Float64,2}`,`B::Array{Float64,2}`: first derivatives of [`Fsys()`](@ref) with respect to arguments `X` [`B`]
    and `XPrime` [`A`]
- `State2Control::Array{Float64,2}`: observation equation
- `LOMstate::Array{Float64,2}`: state transition equation
"""
function linearize_full_model(sr::SteadyResults, m_par::ModelParameters)
    A = zeros(sr.n_par.ntotal, sr.n_par.ntotal)
    B = zeros(sr.n_par.ntotal, sr.n_par.ntotal)

    if sr.n_par.verbose
        println("Initial linearization")
    end
    State2Control, LOMstate, SolutionError, nk, A, B = LinearSolution(sr, m_par, A, B; estim = false)

    return LinearResults(State2Control, LOMstate, A, B, SolutionError, nk)
end

@doc raw"""
    update_model()

Updates the linearized model (around the steady state, after parameter changes in the aggregate model) and solves,
using [`LinearSolution_estim()`](@ref). WARNING: The function is not threadsafe in the sense that calling it will alter the
input(!) lr.A/B across threads, if lr is not local to the thread.

# Returns
`struct` `LinearResults`, containing
- `A::Array{Float64,2}`,`B::Array{Float64,2}`: first derivatives of [`Fsys()`](@ref) with respect to arguments `X` [`B`]
    and `XPrime` [`A`]
- `State2Control::Array{Float64,2}`: observation equation
- `LOMstate::Array{Float64,2}`: state transition equation
"""
function update_model(sr::SteadyResults, lr::LinearResults, m_par::ModelParameters)

    if sr.n_par.verbose
        println("Updating linearization")
    end
    State2Control, LOMstate, SolutionError, nk, A, B = LinearSolution_estim(sr, m_par, lr.A, lr.B; estim = true)

    return LinearResults(State2Control, LOMstate, A, B, SolutionError, nk)
end

@doc raw"""
    model_reduction()

Produce Model Reduction based on Variance Covariance Matrix of States and Controls.

# Returns/ Updates
`struct` `SteadyResults`, containing returns of [`find_steadystate()`](@ref)
"""
function model_reduction(sr, lr, m_par)
    n_par = sr.n_par
    # Reduce further based on importance in dynamics at initial guess 
    if n_par.further_compress
        println("Reduction Step")
        indexes_r, n_par = compute_reduction(sr, lr, m_par, e_set.shock_names)

        println("Number of reduced model factors for DCTs for Vm & Vk:")
        println(length(indexes_r.Vm) + length(indexes_r.Vk))

        println("Number of reduced model factors for copula DCTs:")
        println(length(indexes_r.COP))
    else
        println("Further model reduction switched off --> reverting to full model")
        @set! n_par.PRightAll = float(I[1:n_par.ntotal, 1:n_par.ntotal])
        @set! n_par.PRightStates = float(I[1:n_par.nstates, 1:n_par.nstates])
        indexes_r = sr.indexes
        @set! n_par.nstates_r = n_par.nstates
        @set! n_par.ncontrols_r = n_par.ncontrols
        @set! n_par.ntotal_r = n_par.ntotal
    end

    return SteadyResults(sr.XSS, sr.XSSaggr, sr.indexes, indexes_r, sr.indexes_aggr, 
                            n_par, m_par, state_names, control_names)
end


@doc raw"""
    estimation_prep(sr, lr)

Prepare everything needed for MCMC estimation given linearized model `lr`.

# Arguments
- `sr::SteadyResults`
- `lr::LinearResults`

# Returns
`struct` `EstimResults`, containing all returns of [`mode_finding()`](@ref)
"""
function estimation_prep(sr::SteadyResults, lr::LinearResults, m_par::ModelParameters)
    
    # Load data
    Data_temp = DataFrame(CSV.File(e_set.data_file; missingstring = "NaN"))
    data_names_temp = propertynames(Data_temp)

    # Rename observables that do not have matching model names
    for i in data_names_temp
        name_temp = get(e_set.data_rename, i, :none)
        if name_temp != :none
        rename!(Data_temp, Dict(i => name_temp))
        end
    end

    # Identify missing observations
    observed_vars = e_set.observed_vars_input
    Data = Matrix(Data_temp[:, observed_vars])
    Data_missing = ismissing.(Data)

    # Built selection matrix
    H_sel = zeros(e_set.nobservables, sr.n_par.nstates_r + sr.n_par.ncontrols_r)
    for i in eachindex(observed_vars)
        H_sel[i, getfield(sr.indexes_r, (observed_vars[i]))] = 1.0
    end

    # get names of estimated parameters and add measurement error params
    parnames = collect(fieldnameflatten(m_par))
    if e_set.me_treatment != :fixed
        for i in eachindex(e_set.meas_error_input)
        push!(parnames, Symbol(:σ_me_, e_set.meas_error_input[i]))
        end
    end

    # Set up measurement error
    meas_error, meas_error_prior, meas_error_std = measurement_error(Data, observed_vars, e_set)

    # Prior specification
    priors = collect(metaflatten(m_par, prior)) # model parameters
    if e_set.me_treatment != :fixed
        append!(priors, meas_error_prior)          # add the meas. error priors
    end

    LL(pp)  = -likeli(pp, Data, Data_missing, H_sel, priors, meas_error, 
                meas_error_std, sr, lr, m_par, e_set)[3]

    @load BASEforHANK.e_set.estimation_start_file start_vector start_matrix
    posterior_start = -LL(start_vector)

    # Run Kalman smoother
    smoother_output = likeli(start_vector, Data, Data_missing, H_sel, priors,
        meas_error, meas_error_std, sr, lr, m_par, 
        e_set; smoother = true)

    er = EstimResults(start_vector, start_matrix, meas_error, meas_error_std, parnames, Data, Data_missing, H_sel, priors)

    return er, posterior_start, smoother_output, sr, lr, m_par
end



@doc raw"""
    montecarlo(mr,er;file=e_set.save_posterior_file)

Sample posterior of parameter vector with [`rwmh()`](@ref), take sample mean as
parameter estimate, and save all results in `file`.

# Arguments
- `sr::SteadyResults`
- `mr::LinearResults`
- `er::EstimResults`
"""
function montecarlo(sr::SteadyResults, lr::LinearResults, er::EstimResults, m_par::ModelParameters; file::String=e_set.save_posterior_file)
    hessian_sym = Symmetric(nearest_spd(inv(er.start_matrix)))
    if sr.n_par.verbose
        println("Started MCMC. This might take a while...")
    end
    if e_set.multi_chain_init == true
        init_draw, init_success = multi_chain_init(er.start_vector, hessian_sym, sr, lr, er, m_par, e_set)

        par_final = init_draw
        if init_success == false
            error("Couldn't find initial value that produces posterior")
        end
    else
        par_final = copy(er.start_vector)
    end

    draws_raw, posterior, accept_rate = rwmh(par_final, hessian_sym, sr, lr, er, m_par, e_set)

    ##
    parnames_ascii = collect(metaflatten(m_par, label))
    if e_set.me_treatment != :fixed
        for i in eachindex(e_set.meas_error_input)
            push!(parnames_ascii, string("sigma_me_", e_set.meas_error_input[i]))
        end
    end

    chn = Chains(reshape(draws_raw[e_set.burnin+1:end, :], (size(draws_raw[e_set.burnin+1:end, :])..., 1)),
        [string(parnames_ascii[i]) for i = 1:length(parnames_ascii)])
    chn_summary = summarize(chn)
    par_final = chn_summary[:, :mean]

    ##
    if e_set.me_treatment != :fixed
        m_par = Flatten.reconstruct(m_par, par_final[1:length(par_final)-length(er.meas_error)])
    else
        m_par = Flatten.reconstruct(m_par, par_final)
    end

    lr = update_model(sr, lr, m_par)

    smoother_output = likeli(par_final, sr, lr, er, m_par, e_set; smoother=true)

    if sr.n_par.verbose
        println("MCMC finished.")
    end
    return sr, lr, er, m_par, draws_raw, posterior, accept_rate, par_final, hessian_sym, smoother_output
end

end # module BASEforHANK
