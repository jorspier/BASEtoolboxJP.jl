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

# calibration parameters
@set! m_par.β = 0.992;
@set! m_par.λ = 0.071;
@set! m_par.ζ = 0.00016; 
@set! m_par.Tlev = 1.145; 
@set! m_par.Rbar = 0.038; 

# reduced
@set! m_par.δ_0 = 0.02;
@set! m_par.μ = 1.05;
@set! m_par.μw = 1.05;
@set! m_par.Tprog = 1.0 + 0.28;

# rest
@set! m_par.ξ = 4.0;
@set! m_par.γ = 2.0;
@set! m_par.ρ_h = 0.9815;
@set! m_par.σ_h = 0.135;
@set! m_par.ι = 0.0625;
@set! m_par.α = 0.32;
@set! m_par.δ_s = 0.7055720197078786;
@set! m_par.ϕ = 1.9409223183717077;
@set! m_par.κ = 0.1456082664986374;
@set! m_par.κw = 0.23931075416274708;
@set! m_par.Tc = 1.19;
@set! m_par.Tk = 1.25;
@set! m_par.Ttr_1 = 1.5;
@set! m_par.Ttr_2 = 1.8;
@set! m_par.RRB = 1.0;
@set! m_par.ωΠ = 0.2;
@set! m_par.ιΠ = 0.016;
@set! m_par.shiftΠ = 0.7002848330469671;
@set! m_par.ρ_A = 0.9724112284399131;
@set! m_par.σ_A = 0.0015812471705012755;
@set! m_par.ρ_ZI = 0.7637111671257767;
@set! m_par.σ_ZI = 0.0721141538701523;
@set! m_par.ρ_μ = 0.903740078830077;
@set! m_par.σ_μ = 0.01350860622318172;
@set! m_par.ρ_μw = 0.9057892147641305;
@set! m_par.σ_μw = 0.035058308969408175;
@set! m_par.ρ_s = 0.84;
@set! m_par.σ_Sshock = 0.0;
@set! m_par.Σ_n = 0.0;
@set! m_par.ρ_R = 0.8030565250630299;
@set! m_par.σ_Rshock = 1.0e-8;
@set! m_par.θ_π = 1.25; # 2.0780841671981856
@set! m_par.θ_Y = 0.21872568927661648;
@set! m_par.γ_B = 0.020131162775595176;
@set! m_par.γ_π = -2.1737350397931947;
@set! m_par.γ_Y = -0.4363130165391906;
@set! m_par.ρ_Gshock = 0.9682224473297878;
@set! m_par.σ_Gshock = 0.003761816459554433;
@set! m_par.ρ_τ = 0.4926482696848203;
@set! m_par.γ_Bτ = 0.02 ; # 3.293063617271948
@set! m_par.γ_Yτ = -0.9207283604196101;
@set! m_par.ρ_P = 0.9194235885358465;
@set! m_par.σ_Tprogshock = 0.0;
@set! m_par.γ_BP = 0.0;
@set! m_par.γ_YP = 0.0;
@set! m_par.γ_WP = 0.0;
@set! m_par.ρ_Rshock = 1.0e-8;
@set! m_par.ρ_Tprogshock = 1.0e-8;
@set! m_par.ρ_Sshock = 1.0e-8;
@set! m_par.ρ_TFP = 0.5;     
@set! m_par.σ_TFP = 0.00033388842631140714;    

# new govt investment parameters
@set! m_par.γ_GI = 1.0;                     # Deficit reaction to GI (0 = tax financed, 1 = debt)
@set! m_par.GI_share = 0.028;                # Steady state share of govt investment
# @set! m_par.ϕ_GI = 1/4;                     # Pipeline efficiency (1/4 builds per quarter)
@set! m_par.δ_KG = 0.01;                    # Depreciation of public capital (1% per quarter - 4% per year)
@set! m_par.η_KG = 0.1;                    # Elasticity of output w.r.t public capital
@set! m_par.ρ_GI = 1.0e-8;                    # Persistence of GI shock
@set! m_par.σ_Auth = 0.0632512485920067  #m_par.σ_Gshock * (0.21 / m_par.GI_share) # scale investment shock to same absolute size as consumption shock


## ------------------------------------------------------------------------------------------
## Preparing the calibration
## ------------------------------------------------------------------------------------------

function moments_function_example(m_par)
    # Calculate the base steady state
    ss_full = quiet_call(call_find_steadystate, m_par)

    # Prepare the full state-space representation to get EXACT internal accounting
    sr_full = quiet_call(call_prepare_linearization, ss_full, m_par)

    # Extract exact variables directly from the log-linearized steady-state vector
    K = exp(sr_full.XSS[sr_full.indexes.KSS])
    B = exp(sr_full.XSS[sr_full.indexes.BSS])
    Bgov = exp(sr_full.XSS[sr_full.indexes.BgovSS])
    Y = exp(sr_full.XSS[sr_full.indexes.YSS])
    G = exp(sr_full.XSS[sr_full.indexes.GSS])
    TOP10Wshare = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS])
    fr_borr = sum(ss_full.distrSS[ss_full.n_par.mesh_b .<= 0.0])

    # Compute model moments matching the target dictionary
    model_moments = Dict(
        "K/Y" => (K / Y) / 4.0,
        #"B/K" => B / K,
        "Bgov/Y" => (Bgov / Y) / 4.0,
        "G/Y" => G / Y,
        "T10W" => TOP10Wshare,
        "Frac Borrowers" => fr_borr
    )

    return model_moments
end

# Generate dictionary for calibration
using Optim;

#= For Nelder-Mead
cal_dict = Dict(
    "params_to_calibrate" => [  :β,     # discount factor
                                :λ,     # asset adjustement friction
                                :Tlev,  # income tax level
                                :ζ,     # prob. to become entrepreneur
                                :Rbar], # borrowing wedge
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 12.4 / 4,  # Capital-output ratio (Bundesbank assumes 3-3.2)
        #"B/K" => 0.21,  # Liquid to illiquid ratio
        "Bgov/Y" => 0.60, # debt to output ratio (0.21*12.4/4)
        "G/Y" => 0.21,  # Gov. spending-output ratio (0.21 GER)
        "T10W" => 0.58,  # Top 10% wealth share
        "Frac Borrowers" => 0.12,  # Fraction of borrowers (10-15% HFCS)
    ),
    # One must change options for their respective setting!
    "opt_options" => Optim.Options(;
        time_limit = 60 * 60, # 10800 for 3h
        show_trace = true,
        show_every = 10, # iteration count
        f_reltol = 1e-3,   # stops if fitness ≤ tolerance
    ),
);
=#

# For BBO
cal_dict_BBO = Dict(
    "params_to_calibrate" => [:β, :λ, :Tlev, :ζ, :Rbar],
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 12.4 / 4,  # Capital-output (quarterly) ratio
        #"B/K" => 0.21,  # Liquid to illiquid ratio
        "Bgov/Y" => 0.60, # debt to output ratio 
        "G/Y" => 0.21,  # Gov. spending-output (annualy) ratio
        "T10W" => 0.58,  # Top 10% wealth share
        "Frac Borrowers" => 0.12,  # Fraction of borrowers
    ),
    # One must change options for their respective setting!
    "opt_options" => (
        SearchRange=[
            (0.98, 0.995), # β
            (0.05, 0.1), # λ
            (1.0, 1.35), # Tlev
            (0.0001, 0.0004), # ζ
            (0.0, 0.05), # Rbar
        ],
        Method=:adaptive_de_rand_1_bin_radiuslimited,
        MaxTime=28800, # 8 hours
        TraceInterval=30,
        TraceMode=:compact,
        TargetFitness=1e-3,   # stops if fitness ≤ tolerance
    ),
);
#

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
KG = exp.(sr_full.XSS[sr_full.indexes.KGSS]);

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
        "Public Capital to Output Ratio" KG/Y/4.0
    ];
    header = ["Variable", "Value"],
    title = "Steady State Moments",
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

IRFs_order[IRFs_order .== :Auth] .= :GI

mkpath(paths["bld_example"] * "/IRFs");
plot_irfs(
    [
        #(:ZI, "Inv.-spec. tech."),
        #(:μ, "Price markup"),
        #(:μw, "Wage markup"),
        #(:A, "Risk premium"),
        #(:Rshock, "Mon. policy"),
        (:Gshock, "Structural deficit"),
        (:GI, "Gov. Investment"),
        #(:Tprogshock, "Tax progr."),
        #(:Sshock, "Income risk"),
    ],
    [
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