"""
Mainboard for the baseline example of the BASEforHANK package, calibration.
"""
global_start_time = time()

using PrettyTables, Printf;

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
    "bld_example" => replace(@__DIR__, "examples" => "bld") * "_calib",
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
## Initialize: set up model parameters
## ------------------------------------------------------------------------------------------

m_par = ModelParameters();

# Replace estimated parameters with their prior modes — identical to what main.jl does.
# This ensures calibration runs at the same estimated-parameter values (δ_s, ϕ, etc.)
# as the estimation script, so BK is tested against the correct parameter vector.
priors = collect(metaflatten(m_par, prior));
par_prior = mode.(priors);
m_par = BASEforHANK.Flatten.reconstruct(m_par, par_prior);

# Starting point: Run 1 converged values (fitness 0.052), which also produced
# T10W≈0.56, B50W≈0.045 untargeted — close to PHF targets
#@set! m_par.β    = 0.9875734048672768
#@set! m_par.λ    = 0.04411244973163206
#@set! m_par.ζ    = 0.00043307074620808157
@set! m_par.ζ    = 0.001
#@set! m_par.σ_h  = 0.22251591981275654
@set! m_par.σ_h  = 0.135
@set! m_par.Rbar = 0.040518537810552924
@set! m_par.Tlev = 1.1006962925529395


## ------------------------------------------------------------------------------------------
## Preparing the calibration
## ------------------------------------------------------------------------------------------

# `moments_function`
function moments_function_example(m_par)
    ss_full = quiet_call(call_find_steadystate, m_par;
        n_par_kwargs = (nb = 60, nk = 60, ϵ = 1e-7))
    n_par = ss_full.n_par

    # Convert raw PDF distribution to CDF format (needed for Gini and fr_borr)
    distr_cdf = BASEforHANK.SteadyState.set_distribution(
        BASEforHANK.SteadyState.pdf_to_cdf(ss_full.distrSS),
        n_par.model, n_par.distribution_states, n_par.transition_type,
    )

    # Aggregate quantities from compute_args_hh_prob_ss
    args_hh_prob = BASEforHANK.IncomesETC.compute_args_hh_prob_ss(ss_full.KSS, m_par, n_par)
    BASEforHANK.Parsing.@read_args_hh_prob()

    # # Income distribution: gross income Gini and top-10% share
    # net_income, gross_income, _ = BASEforHANK.IncomesETC.incomes(n_par, m_par, args_hh_prob)
    # TOP10Ishare, _, _, GiniI, _ =
    #     BASEforHANK.SteadyState.distr_summaries_incomes(net_income, gross_income, distr_cdf, n_par)

    # Wealth distribution: sorted 1D CDF, then top-10% and bottom-50% shares
    wealth_grid = BASEforHANK.SteadyState.total_wealth_grid(1.0, n_par, n_par.model)
    wealth_pdf  = BASEforHANK.SteadyState.total_wealth_pdf(distr_cdf, n_par.model)
    IX = sortperm(wealth_grid)
    wealth_grid = wealth_grid[IX]
    wealth_pdf  = wealth_pdf[IX]
    wealth_cdf  = cumsum(wealth_pdf)
    TOP10Wshare = BASEforHANK.SteadyState.topXshare(wealth_grid, wealth_cdf, 10.0, n_par.transition_type)
    #B50Wshare   = 1.0 - BASEforHANK.SteadyState.topXshare(wealth_grid, wealth_cdf, 50.0, n_par.transition_type)

    # Fraction of borrowers
    fr_borr = BASEforHANK.eval_cdf(distr_cdf, :b, n_par, 0.0)

    # Aggregate quantities
    K = ss_full.KSS
    B = sum(ss_full.distrSS .* n_par.mesh_b)
    Y = BASEforHANK.IncomesETC.output(m_par.Z, K, N, m_par)

    # Fiscal aggregates
    BD = sum(ss_full.distrSS .* max.(.-n_par.mesh_b, 0.0))
    Π_F = (1.0 - 1.0/m_par.μ) * Y
    qΠ  = m_par.ωΠ * Π_F / (m_par.RRB - 1.0 + m_par.ιΠ) + 1.0
    Bgov = B - qΠ + 1.0
    distr_h = vec(sum(ss_full.distrSS, dims=(1,2)))
    TR   = BASEforHANK.IncomesETC.transfer_scheme(n_par, m_par, args_hh_prob; distr_h = distr_h)
    RK_before_taxes = (RK - 1.0) / (1.0 - (Tk - 1.0)) + 1.0
    income_taxes = (Tbar - 1.0) * (wH * N + Π_E + Π_U)
    capital_taxes = (Tk - 1.0) * (RK_before_taxes - 1.0) * K
    C = (Y - m_par.δ_0*K - m_par.Rbar*BD - income_taxes - capital_taxes + (TR-1.0)) / Tc
    T = income_taxes + (Tc - 1.0)*C + capital_taxes - (TR - 1.0)
    GI = m_par.GI_share * Y
    G  = T - GI

    return Dict(
        "K/Y"            => (K / Y) / 4.0,
        "Bgov/Y"         => Bgov / Y / 4.0,
        "G/Y"            => G / Y,
        "T10W"           => TOP10Wshare,
        #"B50W"           => B50Wshare,
        "Frac Borrowers" => fr_borr,
    )
end

# Generate dictionary for calibration
using Optim;

# 6 targets, 6 params.
# ι is included with a narrow range (0.04, 0.07) around its default (1/16 ≈ 0.0625)
# to jointly target T10W and B50W without risking the saving collapse seen with wider ranges.
# σ_h remains fixed at 0.135 (model default); it cannot generate German-level income Gini.
targets = Dict(
    "K/Y"            => 2.74,   # Capital-output ratio (annual, NFC)
    "Bgov/Y"         => 0.657,  # Government debt-to-output ratio
    "G/Y"            => 0.207,  # Government spending-to-output ratio
    "T10W"           => 0.576,  # Top 10% wealth share (PHF average)
    #"B50W"           => 0.038,  # Bottom 50% wealth share (PHF)
    "Frac Borrowers" => 0.15,   # Fraction of borrowers (Bundesbank PHF)
)

# Stage 1: BBO on coarse grid — global search
cal_dict_BBO = Dict(
    "params_to_calibrate" => [:β, :λ, :ζ, :σ_h, :Rbar, :Tlev],
    "target_moments" => targets,
    "opt_options" => (
        SearchRange=[
            (0.985, 0.996), # β
            (0.02, 0.08),   # λ
            #(0.04, 0.07),   # ι — narrow range around default 0.0625; lower ι → more top wealth concentration
            (0.0004, 0.0015),# ζ — lower bound prevents entrepreneur sector from vanishing
            (0.10, 0.24),    # σ_h
            (0.02, 0.06),   # Rbar
            (1.0, 1.35),     # Tlev
        ],
        Method=:adaptive_de_rand_1_bin_radiuslimited,
        MaxTime=2*60*60,
        TraceInterval=30,
        TraceMode=:compact,
        TargetFitness=1e-5,
    ),
);

# Stage 2: Nelder-Mead on fine grid — local refinement from BBO result
cal_dict = Dict(
    "params_to_calibrate" => [:β, :λ, :ζ, :σ_h, :Rbar, :Tlev],
    "target_moments" => targets,
    # Parameter-space bounds enforced via smooth transforms inside the objective.
    # Order must match params_to_calibrate exactly.
    "bounds" => [
        (0.97,  0.995),   # β:    unconstrained
        (0.02,  0.08),   # λ:    unconstrained
        (0.0004, 0.004),# ζ:    ≥ 0.0004 (prevents entrepreneur sector from vanishing)
        (0.10, 0.24),   # σ_h:  ≤ 0.25   (prevents implausibly high income risk)
        (0.02,  Inf),   # Rbar: unconstrained
        (1.1,  1.35),   # Tlev: ≥ 1.05   (prevents negative average tax rates)
    ],
    "opt_options" => Optim.Options(;
        time_limit = 7*60*60,
        show_trace = true,
        show_every = 5,
        f_reltol = 1e-5,
    ),
);


m_par = BASEforHANK.SteadyState.run_calibration(
    moments_function_example,
    cal_dict,
    m_par;
    solver = "NelderMead",
);


## ------------------------------------------------------------------------------------------
## Calculate Steady State and prepare linearization
## ------------------------------------------------------------------------------------------

# steady state at m_par
ss_full = call_find_steadystate(m_par);

# sparse DCT representation
sr_full = call_prepare_linearization(ss_full, m_par);

# compute steady state moments
K = exp.(sr_full.XSS[sr_full.indexes.KSS]);
B = exp.(sr_full.XSS[sr_full.indexes.BSS]);
Bgov = exp.(sr_full.XSS[sr_full.indexes.BgovSS]);
Y = exp.(sr_full.XSS[sr_full.indexes.YSS]);
T10W = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS]);
T10Ishare = exp(sr_full.XSS[sr_full.indexes.TOP10IshareSS]);
T10Inetshare = exp(sr_full.XSS[sr_full.indexes.TOP10InetshareSS]);
G = exp.(sr_full.XSS[sr_full.indexes.GSS]);
fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0);
GI = exp.(sr_full.XSS[sr_full.indexes.GISS]);
KG = exp.(sr_full.XSS[sr_full.indexes.KGSS]);
GiniW = exp.(sr_full.XSS[sr_full.indexes.GiniWSS])
GiniC = exp.(sr_full.XSS[sr_full.indexes.GiniCSS])
GiniI = exp.(sr_full.XSS[sr_full.indexes.GiniISS])
GiniInet = exp.(sr_full.XSS[sr_full.indexes.GiniInetSS])

# Bottom 50% wealth share — not stored in XSS, compute from raw PDF in ss_full
# (sr_full.distrSS is a compressed copula struct; ss_full.distrSS is the raw PDF array)
let
    distr_cdf_ss = BASEforHANK.SteadyState.set_distribution(
        BASEforHANK.SteadyState.pdf_to_cdf(ss_full.distrSS),
        sr_full.n_par.model, sr_full.n_par.distribution_states, sr_full.n_par.transition_type,
    )
    wg = BASEforHANK.SteadyState.total_wealth_grid(1.0, sr_full.n_par, sr_full.n_par.model)
    wp = BASEforHANK.SteadyState.total_wealth_pdf(distr_cdf_ss, sr_full.n_par.model)
    IX = sortperm(wg); wg = wg[IX]; wp = wp[IX]; wc = cumsum(wp)
    global B50W = 1.0 - BASEforHANK.SteadyState.topXshare(wg, wc, 50.0, sr_full.n_par.transition_type)
end

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "TOP 10 Wealth Share"                   T10W
        "Bottom 50 Wealth Share"                B50W
        "TOP 10 Gross Income Share"             T10Ishare
        "TOP 10 Net Income Share"               T10Inetshare
        "Wealth Gini"                           GiniW
        "Gross Income Gini"                     GiniI
        "Net Income Gini"                       GiniInet
        "Consumption Gini"                      GiniC
        "Fraction of Borrowers"                 fr_borr
        "Liquid to Illiquid Assets Ratio"       B/K
        "Private Capital to Output Ratio"       K / Y / 4.0
        "Government Debt to Output Ratio"       Bgov / Y / 4.0
        "Government Spending to Output Ratio"   G / Y
        "Government Investment to Output Ratio" GI / Y
        "Public Capital to Output Ratio"        KG / Y / 4.0
    ];
    header = ["Variable", "Value"],
    title = "Steady State Moments - HANK Calibration",
    formatters = ft_printf("%.4f"),
)

## ------------------------------------------------------------------------------------------
## Linearize the full model, find sparse state-space representation
## ------------------------------------------------------------------------------------------

lr_full = linearize_full_model(sr_full, m_par);

## ------------------------------------------------------------------------------------------
## Compute all IRFs, VDs, and BCVDs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Compute IRFs, VDs, and BCVDs...\n"

# Get indices of the shocks
exovars = [getfield(sr_full.indexes, shock_names[i]) for i = 1:length(shock_names)];

# Get standard deviations of the shocks
stds = [getfield(sr_full.m_par, Symbol("σ_", i)) for i in shock_names];

# Compute IRFs
transform_elements =
    transformation_elements(sr_full, sr_full.n_par.model, sr_full.n_par.distribution_states); # Γ, DC, IDC, DCD, IDCD

IRFs, _, IRFs_order, IRFs_dist = compute_irfs(
    exovars,
    lr_full.State2Control,
    lr_full.LOMstate,
    sr_full.XSS,
    sr_full.indexes;
    init_val = stds,
    distribution = true,
    comp_ids = sr_full.compressionIndexes,
    transform_elements = transform_elements,
    n_par = sr_full.n_par,
    m_par = sr_full.m_par,
);

# Compute variance decomposition of IRFs
#VDs = compute_vardecomp(IRFs);

# Compute business cycle frequency variance decomposition
#VDbcs, UnconditionalVar =
#    compute_vardecomp_bcfreq(exovars, stds, lr_full.State2Control, lr_full.LOMstate);

## ------------------------------------------------------------------------------------------
## Graphical outputs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Plotting...\n"

mkpath(paths["bld_example"] * "/IRFs");
plot_irfs(
    [
        #(:ZI, "Inv.-spec. tech."),
        #(:μ, "Price markup"),
        #(:μw, "Wage markup"),
        #(:A, "Risk premium"),
        #(:Rshock, "Mon. policy"),
        #(:Gshock, "Structural deficit"),
        (:GI, "Gov. Investment"),
        #(:Tprogshock, "Tax progr."),
        #(:Sshock, "Income risk"),
    ],
    [
        (:Ygrowth, "Output growth"),
        (:Cgrowth, "Consumption growth"),
        (:Igrowth, "Investment growth"),
        (:Bgov, "Gov. Debt"),
        (:KG, "Public Capital"),
        (:N, "Employment"),
        (:wgrowth, "Wage growth"),
        (:RB, "Nominal rate"),
        (:π, "Inflation"),
        #(:σ, "Income risk"),
        #(:Tprog, "Tax progressivity"),
        (:TOP10Wshare, "Top 10 wealth share"),
        (:TOP10Ishare, "Top 10 inc. share"),
        (:GiniW, "Wealth Gini"),
        (:GiniC, "Consumption Gini"),
    ],
    [(IRFs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    horizon = 80,
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
);


mkpath(paths["bld_example"] * "/IRFs_cat");
plot_irfs_cat(
    Dict(
        #("Monetary", "mon") => [:Rshock, :A],
        ("Fiscal", "fis") => [:Gshock, :GI],
        #("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
    ),
    [
        (:Ygrowth, "Output growth"),
        (:Cgrowth, "Consumption growth"),
        (:Igrowth, "Investment growth"),
        (:Bgov, "Gov. Debt"),
        (:KG, "Public Capital"),
        (:N, "Employment"),
        (:wgrowth, "Wage growth"),
        (:RB, "Nominal rate"),
        (:π, "Inflation"),
        #(:σ, "Income risk"),
        #(:Tprog, "Tax progressivity"),
        (:TOP10Wshare, "Top 10 wealth share"),
        (:TOP10Ishare, "Top 10 inc. share"),
        (:GiniW, "Wealth Gini"),
        (:GiniC, "Consumption Gini"),
    ],
    IRFs,
    IRFs_order,
    sr_full.indexes;
    horizon = 80,
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs_cat",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red, :green, :orange], linestyle = [:solid, :dash, :dot]),
);

# mkpath(paths["bld_example"] * "/IRFs_dist_dev");
# plot_distributional_irfs_deviation(
#     [   (:GI, "Gov. Investment")
#         ],
#     [   ("Wb_b", "Marginal Value of Bonds, over Bonds"),
#         ("Wk_k", "Marginal Value of Capital, over Capital"),
#         ("PDF_b", "Marginal PDF of Bonds"),
#         ("PDF_k", "Marginal PDF of Capital"),
#         ("PDF_bk", "Marginal PDF of Bonds and Capital"),
#         ("PDF_bh", "Marginal PDF of Bonds and Human Capital"),
#         ("PDF_kh", "Marginal PDF of Capital and Human Capital")
#         ],
#     IRFs_dist,
#     IRFs_order,
#     sr_full.n_par;
#     horizon = 80,
#     bounds = Dict(
#         "b" => (sr_full.n_par.grid_b[1], 100.0),
#         "k" => (sr_full.n_par.grid_k[1], 100.0),
#     ),
#     show_fig = false,
#     save_fig = true, 
#     path = paths["bld_example"] * "/IRFs_dist_dev"
# )

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")