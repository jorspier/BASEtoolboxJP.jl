"""
Mainboard for the baseline example of the BASEforHANK package.
"""
global_start_time = time()

using PrettyTables, Printf, BenchmarkTools, LinearAlgebra;

## ------------------------------------------------------------------------------------------
## Header: set up paths, pre-process user inputs, load module
## ------------------------------------------------------------------------------------------

root_dir = replace(Base.current_project(), "Project.toml" => "");
cd(root_dir);

# set up paths for the project
paths = Dict(
    "root" => root_dir,
    "src" => joinpath(root_dir, "src"),
    "bld" => joinpath(root_dir, "bld"),
    "src_example" => @__DIR__,
    "bld_example" => replace(@__DIR__, "examples" => "bld") * "_estim",
);

# create bld directory for the current example
mkpath(paths["bld_example"]);

# pre-process user inputs for model setup
include(paths["src"] * "/Preprocessor/PreprocessInputs.jl")
include(paths["src"] * "/BASEforHANK.jl");
using .BASEforHANK;

# set BLAS threads to the number of Julia threads, prevents grabbing all
BASEforHANK.LinearAlgebra.BLAS.set_num_threads(Threads.nthreads());

## ------------------------------------------------------------------------------------------
## Initialize: set up model parameters, priors, and estimation settings
## ------------------------------------------------------------------------------------------

# model parameters and priors
m_par = ModelParameters();
priors = collect(metaflatten(m_par, prior));
par_prior = mode.(priors);
m_par = BASEforHANK.Flatten.reconstruct(m_par, par_prior);
e_set = BASEforHANK.e_set;

# set some paths
@set! e_set.save_mode_file = paths["bld_example"] * "/HANK_AR1_mode.jld2";
@set! e_set.save_posterior_file = paths["bld_example"] * "/HANK_AR1_chain.jld2";
# par_final_dict.txt contains θ_π = 0.768 (below Taylor-principle boundary of 1.0 → BK alarm).
# Start fresh from prior mode instead, with the corrected θ_π prior below.
# @set! e_set.mode_start_file = paths["src_example"] * "/Data/par_final_dict.txt";
@set! e_set.data_file = paths["src_example"] * "/Data/GER_growth.csv";

# fix seed for random number generation
BASEforHANK.Random.seed!(e_set.seed);

## ------------------------------------------------------------------------------------------
## Calculate Steady State and prepare linearization
## ------------------------------------------------------------------------------------------

# steady state at the prior mode
ss_full = call_find_steadystate(m_par);

# sparse DCT representation
sr_full = call_prepare_linearization(ss_full, m_par);

# save the steady state
jldsave(paths["bld_example"] * "/steadystate.jld2", true; sr_full);

# compute steady state moments
K = exp.(sr_full.XSS[sr_full.indexes.KSS]);
B = exp.(sr_full.XSS[sr_full.indexes.BSS]);
Bgov = exp.(sr_full.XSS[sr_full.indexes.BgovSS]);
Y = exp.(sr_full.XSS[sr_full.indexes.YSS]);
T10W = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS]);
B50W = exp(sr_full.XSS[sr_full.indexes.BOT50WshareSS]);
B50I = exp(sr_full.XSS[sr_full.indexes.BOT50IshareSS]);
G = exp.(sr_full.XSS[sr_full.indexes.GSS]);
fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0);

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "Liquid to Illiquid Assets Ratio" B/K
        "Capital to Output Ratio" K / Y/4.0
        "Government Debt to Output Ratio" Bgov / Y/4.0
        "Government Spending to Output Ratio" G/Y
        "TOP 10 Wealth Share" T10W
        "BOT 50 Wealth Share" B50W
        "BOT 50 Income Share" B50I
        "Fraction of Borrower" fr_borr
    ];
    header = ["Variable", "Value"],
    title = "Steady State Moments",
    formatters = ft_printf("%.4f"),
)

## ------------------------------------------------------------------------------------------
## Linearize the full model, find sparse state-space representation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ## ------------------------------------------------------------------------------------------

lr_full = linearize_full_model(sr_full, m_par);

# save the linearization
jldsave(paths["bld_example"] * "/linearresults.jld2", true; lr_full);

# sparse state-space representation
sr_reduc = model_reduction(sr_full, lr_full, m_par);
lr_reduc = update_model(sr_reduc, lr_full, m_par);

# save the reduction
jldsave(paths["bld_example"] * "/reduction.jld2", true; sr_reduc, lr_reduc);

# model timing
@printf "One model solution takes: \n"
@set! sr_reduc.n_par.verbose = false;
@btime lr_reduc = update_model(sr_reduc, lr_full, m_par);
@set! sr_reduc.n_par.verbose = true;

## ------------------------------------------------------------------------------------------
## Estimation
## ------------------------------------------------------------------------------------------

if e_set.estimate_model == true
    @printf "\n"
    @printf "Estimation...\n"

    # warning: estimation might take a long time!
    er_mode, posterior_mode, smoother_mode, sr_mode, lr_mode, m_par_mode =
        find_mode(sr_reduc, lr_reduc, m_par, e_set)

    # Save the Hessian from mode finding BEFORE overwriting it for MCMC proposals.
    # If compute_hessian = true: hessian_at_mode is the true numerical Hessian at the
    # posterior mode → use inv(hessian_at_mode) for Laplace-approximation posterior SDs.
    # If compute_hessian = false: this is just the identity (not useful for Laplace approx).
    hessian_at_mode = copy(er_mode.hessian_final)

    # Set MCMC proposal covariance.
    # If compute_hessian = true: use the true Hessian (captures parameter correlations,
    # keeps correlated proposals within BK-feasible region → better acceptance rate).
    # If compute_hessian = false: fall back to diagonal approximation.
    if e_set.compute_hessian
        # Build a usable proposal covariance from the finite-difference Hessian.
        #
        # Two sources of bad eigenvalues:
        #   (A) Alarm contamination: FD probes cross the BK boundary → entries O(9e23).
        #       These inflate max|λ|, making ε_floor = 1e-6*max ≈ 9e17, which blindly
        #       floors ALL genuine eigenvalues (O(1–1e4)) to that value → tiny proposals.
        #   (B) Genuine non-identification: likelihood flat in some direction → eigenvalue
        #       near zero but not alarm-contaminated → should get prior-scale proposal.
        #
        # Strategy:
        #   1. Separate alarm-contaminated eigenvalues (|λ| > 1e18) from genuine ones.
        #   2. Floor genuine eigenvalues with ε_floor based only on the genuine maximum.
        #   3. Replace alarm-contaminated eigenvalues with prior-scale curvature (= 100,
        #      i.e. 1/prior_std² for prior_std ≈ 0.1) so MCMC can still explore those
        #      directions at prior-reasonable step sizes.
        H_sym = Symmetric(er_mode.hessian_final)
        F = eigen(H_sym)

        # alarm/h² threshold: alarm ≈ 9e15, h = 1e-4  →  9e15/1e-8 = 9e23.
        # Use 1e18 as conservative boundary between genuine curvature and alarm artifact.
        alarm_ev_threshold = 1e18
        is_alarm = abs.(F.values) .> alarm_ev_threshold
        n_alarm  = sum(is_alarm)

        genuine_max = maximum(abs.(F.values[.!is_alarm]); init = 1.0)
        ε_floor = max(1e-4, 1e-6 * genuine_max)   # floor only genuine directions

        # Replace alarm directions with prior-scale curvature so proposals are bounded.
        prior_scale_ev = 100.0   # 1/0.1² — conservative: prior_std ≈ 0.1 for most params
        ev_proposal = copy(F.values)
        ev_proposal[is_alarm]  .= prior_scale_ev
        ev_proposal             .= max.(ev_proposal, ε_floor)

        n_floored = sum(F.values[.!is_alarm] .< ε_floor)
        if n_alarm > 0
            @printf "Hessian: %d alarm-contaminated direction(s) → replaced with prior-scale (%.1f).\n" n_alarm prior_scale_ev
        end
        if n_floored > 0
            @printf "Hessian: flooring %d near-zero eigenvalue(s) at %.2e.\n" n_floored ε_floor
        end
        H_pd = Symmetric(F.vectors * Diagonal(ev_proposal) * F.vectors')
        @set! er_mode.hessian_final = Matrix(H_pd)
        @printf "Using true Hessian from mode finding for MCMC proposals.\n"
    else
        # Diagonal fallback: 1% of parameter value, with 1e-4 floor
        # NOTE: breaks for near-zero parameters (γ_π, γ_Y, γ_Yτ) → see comments in
        # filter_smoother.jl for the step-size fix if MCMC AR collapses
        hank_start_vals = er_mode.par_final
        step_sizes = (abs.(hank_start_vals) .* 0.01) .+ 1e-4
        variances = step_sizes .^ 2
        hessian_diag = 1.0 ./ variances
        @set! er_mode.hessian_final = Matrix(Diagonal(hessian_diag))
    end

    # Only relevant output for later plotting will be saved.
    # If you require all smoother output including the variance estimates
    # over time, items 4 and 5, comment out the next line.
    # This increases the hard disk storage significantly.
    smoother_mode = (0.0, 0.0, smoother_mode[3], 0.0, 0.0, smoother_mode[6], 0.0)

    # Stores mode finding results in file e_set.save_mode_file
    jldsave(
        e_set.save_mode_file,
        true;
        posterior_mode,
        smoother_mode,
        sr_mode,
        lr_mode,
        er_mode,
        m_par_mode,
        e_set,
    )
    # !! warning: the provided mode file does not contain smoothed covars (smoother_mode[4] and [5])!!
    # @load BASEforHANK.e_set.save_mode_file posterior_mode sr_mode lr_mode er_mode m_par_mode smoother_mode

    sr_mc,
    lr_mc,
    er_mc,
    m_par_mc,
    draws_raw,
    posterior,
    accept_rate,
    par_final,
    hessian_sym,
    smoother_output = sample_posterior(sr_mode, lr_mode, er_mode, m_par_mode, e_set)

    # Only relevant output for later plotting will be saved.
    # If you want all smoother output including the variance estimates
    # over time, items 4 and 5, comment out the next line.
    # This increases the hard disk storage significantly.
    smoother_output = (0.0, 0.0, smoother_output[3], 0.0, 0.0, smoother_output[6], 0.0)

    # Stores mcmc results in file e_set.save_posterior_file
    jldsave(
        e_set.save_posterior_file,
        true;
        sr_mc,
        lr_mc,
        er_mc,
        m_par_mc,
        draws_raw,
        posterior,
        accept_rate,
        par_final,
        hessian_sym,
        smoother_output,
        e_set,
    )
    # !! The following file is not provided !!
    #      @load BASEforHANK.e_set.save_posterior_file sr_mc lr_mc er_mc  m_par_mc draws_raw posterior accept_rate par_final hessian_sym smoother_output e_set

    @printf "Estimation... Done. \n"
end

## ------------------------------------------------------------------------------------------
## Compute all IRFs, VDs, BCVDs, and historical decompositions
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Compute IRFs, VDs, and BCVDs...\n"

# Get indices of the shocks
exovars = [getfield(sr_mc.indexes_r, shock_names[i]) for i = 1:length(shock_names)];

# Get standard deviations of the shocks
stds_mc = [getfield(m_par_mc, Symbol("σ_", i)) for i in shock_names];
stds_mode = [getfield(m_par_mode, Symbol("σ_", i)) for i in shock_names];

# Compute IRFs
IRFs_mc, _, IRFs_order = compute_irfs(
    exovars,
    lr_mc.State2Control,
    lr_mc.LOMstate,
    sr_mc.XSS,
    sr_mc.indexes_r;
    init_val = stds_mc,
);
IRFs_mode, _, _, = compute_irfs(
    exovars,
    lr_mode.State2Control,
    lr_mode.LOMstate,
    sr_mode.XSS,
    sr_mc.indexes_r;
    init_val = stds_mode,
);

# Compute variance decomposition of IRFs
VDs_mc = compute_vardecomp(IRFs_mc);
VDs_mode = compute_vardecomp(IRFs_mode);

# Compute business cycle frequency variance decomposition
VDbcs_mc, UnconditionalVar_mc =
    compute_vardecomp_bcfreq(exovars, stds_mc, lr_mc.State2Control, lr_mc.LOMstate);
VDbcs_mode, UnconditionalVar_mode =
    compute_vardecomp_bcfreq(exovars, stds_mode, lr_mode.State2Control, lr_mode.LOMstate);

# Compute historical decompositions
ShockContr, ShockContr_order = compute_hist_decomp(
    exovars,
    lr_mc.State2Control,
    lr_mc.LOMstate,
    smoother_output,
    sr_mc.indexes_r,
);

## ------------------------------------------------------------------------------------------
## Graphical outputs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Plotting...\n"

horizon = 80
shocks_to_plot = [
    #(:TFP, "TFP"),
    #(:ZI, "Inv.-spec. tech."),
    #(:μ, "Price markup"),
    #(:μw, "Wage markup"),
    #(:A, "Risk premium"),
    #(:Rshock, "Mon. policy"),
    #(:Gshock, "Structural deficit"),
    (:GI, "Gov. Investment shock"),
    #(:Tprogshock, "Tax progr."),
    #(:Sshock, "Income risk"),
    ]
vars_to_plot = [
    (:Y, "Output"), # removed growth
    (:C, "Consumption"),
    (:Bgov, "Gov. Debt"),
    (:KG, "Public Capital"),    
    (:K, "Private Capital"),
    (:I, "Investment"),
    (:N, "Employment"),
    (:wF, "Wage"),
    (:π, "Inflation"),
    (:RB, "Nominal rate"),
    (:RRL, "Return on Bonds"),
    (:RK, "Return on Capital"),
    (:LPXA, "Ex-ante Liquidity Premium"),
    (:LP, "Ex-post Liquidity Premium"),
    #(:σ, "Income risk"),
    #(:Tprog, "Tax progressivity"),
    #(:TOP10Wshare, "Top 10 wealth share"),
    #(:TOP10Ishare, "Top 10 gross inc. share"),
    #(:TOP10Inetshare, "Top 10 net inc. share"),
    (:GiniW, "Wealth Gini"),
    (:GiniC, "Consumption Gini")
    ]


mkpath(paths["bld_example"] * "/IRFs");
plot_irfs(
    shocks_to_plot,
    vars_to_plot,
    [(IRFs_mc, "Posterior mean"), (IRFs_mode, "Mode")],
    IRFs_order,
    sr_mc.indexes_r;
    horizon,
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
)

mkpath(paths["bld_example"] * "/IRFs_cat");
plot_irfs_cat(
    Dict(
        #("Monetary", "mon") => [:Rshock, :A],
        ("Fiscal", "fis") => [:Gshock, :GI], # :Tprogshock,
        #("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
    ),
    vars_to_plot,
    IRFs_mc,
    IRFs_order,
    sr_mc.indexes_r;
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs_cat",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red, :green, :orange], linestyle = [:solid, :dash, :dot]),
)

# mkpath(paths["bld_example"] * "/VDs");
# plot_vardecomp(
#     vars_to_plot,
#     [(VDs_mc, "Posterior mean"), (VDs_mode, "Mode")],
#     IRFs_order,
#     sr_mc.indexes_r;
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDs",
# )

# mkpath(paths["bld_example"] * "/VDs_cat");
# plot_vardecomp(
#     vars_to_plot,
#     [(VDs_mc, "Posterior mean"), (VDs_mode, "Mode")],
#     IRFs_order,
#     sr_mc.indexes_r;
#     shock_categories = Dict(
#         ("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :GI], # :Tprogshock,
#         ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
#     ),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDs_cat",
# )

# mkpath(paths["bld_example"] * "/VDbcs");
# plot_vardecomp_bcfreq(
#     vars_to_plot,
#     [(VDbcs_mc, "Posterior mean"), (VDbcs_mode, "Mode")],
#     IRFs_order,
#     sr_mc.indexes_r;
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDbcs",
# )

# mkpath(paths["bld_example"] * "/VDbcs_cat");
# plot_vardecomp_bcfreq(
#     vars_to_plot,
#     [(VDbcs_mc, "Posterior mean"), (VDbcs_mode, "Mode")],
#     IRFs_order,
#     sr_mc.indexes_r;
#     shock_categories = Dict(
#         ("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :GI], # :Tprogshock,
#         ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
#     ),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDbcs_cat",
# )

# mkpath(paths["bld_example"] * "/HDs");
# plot_hist_decomp(
#     vars_to_plot,
#     ShockContr,
#     ShockContr_order,
#     sr_mc.indexes_r;
#     timeline = collect(1991.25:0.25:2025.75),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/HDs",
# );

# mkpath(paths["bld_example"] * "/HDs_cat");
# plot_hist_decomp(
#     vars_to_plot,
#     ShockContr,
#     ShockContr_order,
#     sr_mc.indexes_r;
#     shock_categories = Dict(
#         ("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :GI], # :Tprogshock,
#         ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
#     ),
#     timeline = collect(1991.25:0.25:2025.75),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/HDs_cat",
# );

# Print cumulative mutipliers
println("\n--- Cumulative PV Multipliers: Public Investment (AR1 - Model) ---")
table_GI = compute_pv_multipliers(IRFs_mode, IRFs_order, sr_mode.indexes_r, sr_mode.XSS, :GI; max_horizon = 80)
display(table_GI)

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")