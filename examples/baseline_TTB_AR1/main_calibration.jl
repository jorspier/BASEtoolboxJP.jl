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

@set! m_par.β = 0.9923121076496764
@set! m_par.λ = 0.08021000338907154
@set! m_par.ζ = 0.003149051319305283
@set! m_par.Rbar = 0.011993601601887893


## ------------------------------------------------------------------------------------------
## Preparing the calibration
## ------------------------------------------------------------------------------------------

# `moments_function`
function moments_function_example(m_par)
    # calculate the steady state associated with the current parameter vector
    ss_full = quiet_call(call_find_steadystate, m_par)
    sr_full = quiet_call(call_prepare_linearization, ss_full, m_par)
      
    K    = exp(sr_full.XSS[sr_full.indexes.KSS])
    B    = exp(sr_full.XSS[sr_full.indexes.BSS])
    Y    = exp(sr_full.XSS[sr_full.indexes.YSS])
    Bgov = exp(sr_full.XSS[sr_full.indexes.BgovSS])
    TOP10Wshare = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS])
    GiniW = exp(sr_full.XSS[sr_full.indexes.GiniWSS])

    #B_raw = B/K
    #BK_lo, BK_hi, BK_target = 0.35, 0.45, 0.40

    #G    = exp(sr_full.XSS[sr_full.indexes.GSS])
    #T    = exp(sr_full.XSS[sr_full.indexes.TSS])
    #TOP10Ishare = exp(sr_full.XSS[sr_full.indexes.TOP10IshareSS])
    #sdlogy = exp(sr_full.XSS[sr_full.indexes.sdlogySS])

    # Fraction of borrowers
    # fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0)

    # Top 10% wealth share 
    # n_par = ss_full.n_par
    # total_wealth = vec(n_par.mesh_b[:,:,1] .+ n_par.mesh_k[:,:,1])
    # dist_vec = vec(sum(ss_full.distrSS, dims=3))

    # IX = sortperm(total_wealth)
    # sorted_wealth = total_wealth[IX]
    # sorted_dist = dist_vec[IX]

    # total_wealth_w = sorted_wealth .* sorted_dist
    # wealthshares = cumsum(total_wealth_w) ./ sum(total_wealth_w)
    # cum_dist = cumsum(sorted_dist)

    # TOP10Wshare = 1.0 - BASEforHANK.Tools.mylinearinterpolate(cum_dist, wealthshares, [0.9])[1]
        
    if GiniW >= 1.0
        return Dict(k => 1e6 for k in keys(target_moments))
    end

    return Dict(
        "K/Y"            => (K / Y) / 4.0,
        "B/K"            => B/K, 
        "Bgov/Y"         => (Bgov / Y) / 4.0,
        "T10W"           => TOP10Wshare,
        "GiniW"          => GiniW,
        #"B/Y"            =>  (B / Y) / 4.0,
        #"G/Y"            =>  G / Y,
        #"Frac Borrowers" =>  fr_borr,
    )

    return model_moments
end;

# Generate dictionary for calibration
using Optim;

# For Nelder-Mead
cal_dict = Dict(
    "params_to_calibrate" => [  :β,     # discount factor
                                :λ,     # asset adjustement friction
                                #:Tlev,  # income tax level
                                :ζ,     # prob. to become entrepreneur
                                :Rbar, # borrowing wedge
                                #:δ_0,   # depreciation rate
                                #:ρ_h,   # persistence of income shock
                                #:σ_h,   # std. dev. of income shock
                            ],
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 3.2,  # Capital-output (quarterly) ratio
        "Bgov/Y" => 0.66, # debt to output ratio 
        #"G/Y" => 0.21,  # Gov. spending-output (annualy) ratio
        #"T/Y" => 0.24,  # Tax revenue to output ratio (annual)
        "B/K" => 0.37,  # Liquid to illiquid ratio
        "T10W" => 0.58,  # Top 10% wealth share
        "GiniW" => 0.74,
        #"T10I" => 0.32,  # Top 10% income share
        #"sdlog(y)" => 0.60, # standard deviation of log income (SOEP)
        #"Frac Borrowers" => 0.12,  # Fraction of borrowers
    ),
    # One must change options for their respective setting!
    "opt_options" => Optim.Options(;
        time_limit = 7200, # 10800 for 3h
        show_trace = true,
        show_every = 10, # iteration count
        f_reltol = 1e-5,   # stops if fitness ≤ tolerance
    ),
);
#

# For BBO
cal_dict_BBO = Dict(
    "params_to_calibrate" => [:β, :λ, :ζ, :Rbar, :σ_h],
    #"params_to_calibrate" => [:β, :λ, :Tlev, :Tprog, :δ_0, :ζ, :ρ_h, :σ_h],
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 3.2,       # Capital-output (quarterly) ratio
        "Bgov/Y" => 0.66,   # debt to output ratio (average)
        #"G/Y" => 0.21,     # Gov. spending-output (annual) ratio
        "B/K" => 0.37,      # Liquid to illiquid ratio (42 is the average, 37 the 2023 value)
        "T10W" => 0.58,     # Top 10% wealth share (average)
        #"GiniW" => 0.74,   # Wealth Gini (average)
        #"Frac Borrowers" => 0.18,  # Fraction of borrowers
    ),
    # One must change options for their respective setting!
    "opt_options" => (
        SearchRange=[
            (0.985, 0.9995), # β
            (0.02, 0.1), # λ
            #(1.1, 1.5), # Tlev
            #(1.1, 1.5), # Tprog
            #(0.01, 0.04), # δ_0
            (0.0007, 0.005), # ζ
            (0.009, 0.04), # Rbar
            #(0.95, 0.999), # ρ_h
            (0.05, 0.3), # σ_h
        ],
        Method=:adaptive_de_rand_1_bin_radiuslimited,
        MaxTime=2*60*60, 
        TraceInterval=30,
        TraceMode=:compact,
        TargetFitness=1e-5,   # stops if fitness ≤ tolerance
    ),
);


# Run calibration. Exports parameters
m_par = BASEforHANK.SteadyState.run_calibration(
    moments_function_example,
    cal_dict_BBO, # or cal_dict_BBO for BBO
    m_par;
    solver = "BBO", # "NelderMead" or "BBO"
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
G = exp.(sr_full.XSS[sr_full.indexes.GSS]);
fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0);
# new 
GI = exp.(sr_full.XSS[sr_full.indexes.GISS]);
KG = exp.(sr_full.XSS[sr_full.indexes.KGSS]);
GiniW = exp.(sr_full.XSS[sr_full.indexes.GiniWSS])


# Display steady state moments
@printf "\n"
pretty_table(
    [
        "TOP 10 Wealth Share" T10W
        "Fraction of Borrower" fr_borr
        "Liquid to Illiquid Assets Ratio" B/K
        "Private Capital to Output Ratio" K / Y/4.0
        "Government Debt to Output Ratio" Bgov / Y/4.0
        "Government Spending to Output Ratio" G/Y
        "Government Investment to Output Ratio" GI/Y
        "Public Capital to Output Ratio" KG/Y/4.0
        "Wealth Gini" GiniW
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
VDs = compute_vardecomp(IRFs);

# Compute business cycle frequency variance decomposition
VDbcs, UnconditionalVar =
    compute_vardecomp_bcfreq(exovars, stds, lr_full.State2Control, lr_full.LOMstate);

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
        (:σ, "Income risk"),
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

mkpath(paths["bld_example"] * "/IRFs_dist_dev");
plot_distributional_irfs_deviation(
    [   (:GI, "Gov. Investment")
        ],
    [   ("Wb_b", "Marginal Value of Bonds, over Bonds"),
        ("Wk_k", "Marginal Value of Capital, over Capital"),
        ("PDF_b", "Marginal PDF of Bonds"),
        ("PDF_k", "Marginal PDF of Capital"),
        ("PDF_bk", "Marginal PDF of Bonds and Capital"),
        ("PDF_bh", "Marginal PDF of Bonds and Human Capital"),
        ("PDF_kh", "Marginal PDF of Capital and Human Capital")
        ],
    IRFs_dist,
    IRFs_order,
    sr_full.n_par;
    horizon = 80,
    bounds = Dict(
        "b" => (sr_full.n_par.grid_b[1], 100.0),
        "k" => (sr_full.n_par.grid_k[1], 100.0),
    ),
    show_fig = false,
    save_fig = true, 
    path = paths["bld_example"] * "/IRFs_dist_dev"
)

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")