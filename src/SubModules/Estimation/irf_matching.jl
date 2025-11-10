"""
We follow Christiano et al. (2010) in employing a Bayesian variant of the Christiano et al. (2005)-type impulse response matching approach.
"""
function irfmatch(
    par,
    IRFtargets,
    weights,
    shocks_selected,
    isstate,
    indexes_sel_vars,
    priors,
    sr,
    lr,
    m_par,
    e_set,
)
    par = remap_params!(par, priors)

    return irfmatch_backend(
        par,
        IRFtargets,
        weights,
        shocks_selected,
        isstate,
        indexes_sel_vars,
        priors,
        sr,
        lr,
        m_par,
        e_set,
    )
end

function irfmatch_backend(
    par,
    IRFtargets,
    weights,
    shocks_selected,
    isstate,
    indexes_sel_vars,
    priors,
    sr,
    lr,
    m_par,
    e_set,
)

    # check priors, abort if they are violated
    prior_like::eltype(par), alarm_prior::Bool = prioreval(Tuple(par), Tuple(priors))
    alarm = false
    if alarm_prior
        IRFdist = 9.e15
        alarm = true
        State2Control = zeros(sr.n_par.ncontrols_r, sr.n_par.nstates_r)
    else
        # replace estimated values in m_par by last candidate
        m_par = Flatten.reconstruct(m_par, par)

        # solve model using candidate parameters
        local State2Control, LOMstate, alarm_sgu
        @silent begin
            State2Control, LOMstate, alarm_sgu = LinearSolution_reduced_system(
                sr,
                m_par,
                lr.A,
                lr.B;
                allow_approx_sol = true,
            )
        end
        if alarm_sgu # abort if model doesn't solve
            IRFdist = 9.e15
            alarm = true
        else
            irf_horizon = e_set.irf_matching_dict["irf_horizon"]
            IRFs = compute_irfs(
                sr,
                State2Control,
                LOMstate,
                m_par,
                shocks_selected,
                indexes_sel_vars,
                isstate,
                irf_horizon,
            )
            IRFdist = (sum((IRFs[:] .- IRFtargets[:]) .^ 2 .* weights[:])) ./ 2
        end
    end

    prior_scale = 1.0 # could be adjusted to tune the relative weight of prior i.e., 0.0 is MLE

    return -IRFdist, prior_like, -IRFdist .+ prior_like * prior_scale, alarm
end

function compute_irfs(
    sr,
    State2Control,
    LOMstate,
    m_par,
    shocks_selected,
    indexes_sel_vars,
    isstate,
    irf_horizon,
)
    n_vars = length(indexes_sel_vars)
    n_shocks = length(shocks_selected)

    IRFs = Array{Float64}(undef, irf_horizon + 1, n_vars, n_shocks)
    IRFsout = Array{Float64}(undef, irf_horizon, n_vars, n_shocks)

    for (i, s) in enumerate(shocks_selected)
        x = zeros(size(LOMstate, 1))
        x[getfield(sr.indexes_r, s)] = getfield(m_par, Symbol("σ_", s))

        MX = [I; State2Control]
        for t = 1:(irf_horizon + 1)
            IRFs[t, :, i] = (MX[indexes_sel_vars, :] * x)'
            x[:] = LOMstate * x
        end
    end
    IRFsout[:, isstate, :] .= IRFs[2:end, isstate, :] # IRFs for state variables represent end-of-period values
    IRFsout[:, .~isstate, :] .= IRFs[1:(end - 1), .~isstate, :] # IRFs for state variables represent end-of-period values

    return IRFsout
end

softplus(x) = log(1 + exp(x))

function remap_params!(θ::AbstractVector, priors::AbstractVector; ϵ = 1e-9)
    @assert length(θ) == length(priors)
    for i in eachindex(θ)
        d = priors[i]
        if θ[i] ∉ Distributions.support(d)
            if d isa Gamma || d isa InverseGamma
                θ[i] = softplus(θ[i]) + ϵ                     # > 0

            elseif d isa Beta
                θ[i] = ϵ + (1 - 2ϵ) / (1 + exp(-θ[i]))        # ∈ (ϵ, 1-ϵ)

            elseif d isa Normal
                continue  # unbounded
            end
        end
    end

    return θ
end
