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

@set! m_par.ξ = 4.0;
@set! m_par.γ = 2.0;
@set! m_par.β = 0.98255;
@set! m_par.λ = 0.06692937984; # 0.065;
@set! m_par.ρ_h = 0.98;
@set! m_par.σ_h = 0.12;
@set! m_par.ι = 0.0625;
@set! m_par.ζ = 0.00022222222222222223;
@set! m_par.α = 0.318;
@set! m_par.δ_0 = 0.021500000000000002;
@set! m_par.δ_s = 0.7055720197078786;
@set! m_par.ϕ = 1.9409223183717077;
@set! m_par.μ = 1.1;
@set! m_par.κ = 0.1456082664986374;
@set! m_par.μw = 1.1;
@set! m_par.κw = 0.23931075416274708;
@set! m_par.Tlev = 1.1775; #1.0 + (1 - 0.8225);
@set! m_par.Tprog = 1.0 + 0.1022;
@set! m_par.Tc = 1.0;
@set! m_par.Tk = 1.0;
@set! m_par.Ttr_1 = 1.0;
@set! m_par.Ttr_2 = 1.0;
@set! m_par.RRB = 1.0;
@set! m_par.Rbar = 0.021778180864641117;
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
@set! m_par.ρ_s = 0.544722245741144;
@set! m_par.σ_Sshock = 0.6918558038597916;
@set! m_par.Σ_n = 28.879770107327673;
@set! m_par.ρ_R = 0.8030565250630299;
@set! m_par.σ_Rshock = 0.002306627917745612;
@set! m_par.θ_π = 2.0780841671981856;
@set! m_par.θ_Y = 0.0; # 0.21872568927661648;
@set! m_par.γ_B = 0.020131162775595176;
@set! m_par.γ_π = -2.1737350397931947;
@set! m_par.γ_Y = -0.4363130165391906;
@set! m_par.ρ_Gshock = 0.9682224473297878;
@set! m_par.σ_Gshock = 0.003761816459554433;
@set! m_par.ρ_τ = 0.4926482696848203;
@set! m_par.γ_Bτ = 3.293063617271948;
@set! m_par.γ_Yτ = -0.9207283604196101;
@set! m_par.ρ_P = 0.9194235885358465;
@set! m_par.σ_Tprogshock = 0.06865440038519788;
@set! m_par.γ_BP = 0.0;
@set! m_par.γ_YP = 0.0;
@set! m_par.γ_WP = 0.0;
@set! m_par.ρ_Rshock = 1.0e-8;
@set! m_par.ρ_Tprogshock = 1.0e-8;
@set! m_par.ρ_Sshock = 1.0e-8;

# new govt investment parameters
@set! m_par.γ_GI = 1.0;                     # Deficit reaction to GI (0 = tax financed, 1 = debt)
@set! m_par.GI_share = 0.03;                # Steady state share of govt investment
@set! m_par.ϕ_GI = 1/4;                     # Pipeline efficiency (1/4 builds per quarter)
@set! m_par.δ_KG = 0.015;                   # Depreciation of public capital
@set! m_par.η_KG = 0.05;                    # Elasticity of output w.r.t public capital
@set! m_par.ρ_GI = 0.90;                    # Persistence of GI shock
@set! m_par.σ_GI = 0.01;                    # Std dev of GI shock
@set! m_par.ρ_TFP = 0.9978155269262137;     # Persistence of TFP shock
@set! m_par.σ_TFP = 0.00600947811158941;    # Std dev of TFP shock

## ------------------------------------------------------------------------------------------
## Preparing the calibration
## ------------------------------------------------------------------------------------------

# `moments_function`
function moments_function_example(m_par)
    # calculate the steady state associated with the current parameter vector
    ss_full = quiet_call(call_find_steadystate, m_par)

    # extract numerical parameters for the calculation of the aggregates
    n_par = ss_full.n_par

    # Compute aggregates
    args_hh_prob = BASEforHANK.IncomesETC.compute_args_hh_prob_ss(ss_full.KSS, m_par, n_par)
    BASEforHANK.Parsing.@read_args_hh_prob()

    # Compute aggregates
    # capital
    K = ss_full.KSS

    # bonds
    B = sum(ss_full.distrSS .* ss_full.n_par.mesh_b)

    # marginal cost, wages, output
    mc = 1.0 ./ m_par.μ
    wF = BASEforHANK.IncomesETC.wage(mc, m_par.Z, K, N, m_par)
    Y = BASEforHANK.IncomesETC.output(m_par.Z, K, N, m_par)

    # taxes
    T =
        (Tbar - 1.0) .* (1.0 ./ m_par.μw .* wF .* N) + # labor income
        (Tbar - 1.0) .* Π_E + # profit income
        (Tbar - 1.0) * ((1.0 .- 1.0 ./ m_par.μw) .* wF .* N) # union profit income

    # government bonds
    RRL = m_par.RRB # in ss
    qΠ = m_par.ωΠ .* (1.0 .- 1.0 ./ m_par.μ) .* Y ./ (RRL .- 1 .+ m_par.ιΠ) + 1.0
    Bgov = B .- qΠ .+ 1.0

    # Bgov = B - (m_par.ωΠ .* (1.0 .- 1.0 ./ m_par.μ) .* Y ./ (RRB .- 1 .+ m_par.ιΠ)) # Needs to be checked.

    # government investment
    GI = m_par.GI_share * Y

    # government spending
    G = T - (RRL - 1.0) * Bgov - GI

    # calculate the fraction of borrowers
    distrSS = ss_full.distrSS
    # fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0)
    fr_borr = sum(distrSS[n_par.mesh_b .<= 0.0])

    # Price of capital is 1 in the steady-state
    q = 1

    # calculate the Top 10% wealth share
    total_wealth = Array{eltype(distrSS)}(undef, n_par.nk .* n_par.nb)
    for k = 1:(n_par.nk)
        for b = 1:(n_par.nb)
            total_wealth[b + (k - 1) * n_par.nb] = n_par.grid_b[b] .+ q .* n_par.grid_k[k]
        end
    end
    # Wealth shares
    IX = sortperm(total_wealth)
    total_wealth = total_wealth[IX]
    total_wealth_pdf = sum(distrSS; dims = 3)
    total_wealth_pdf = total_wealth_pdf[IX]
    total_wealth_cdf = cumsum(total_wealth_pdf)
    total_wealth_w = total_wealth .* total_wealth_pdf # weighted
    wealthshares = cumsum(total_wealth_w) ./ sum(total_wealth_w)
    TOP10Wshare =
        1.0 -
        BASEforHANK.Tools.mylinearinterpolate(total_wealth_cdf, wealthshares, [0.9])[1]

    # Compute model moments -- note that the keys match the keys of `target_moments`
    model_moments = Dict(
        "K/Y" => K / Y / 4,
        "B/K" => B / K,
        "G/Y" => G / Y,
        "T10W" => TOP10Wshare,
        "Frac Borrowers" => fr_borr,
        # "GI/Y" => GI / Y,
    )

    return model_moments
end;

# Generate dictionary for calibration
using Optim;

# For Nelder-Mead
cal_dict = Dict(
    "params_to_calibrate" => [:β, :λ, :Tlev, :ζ, :Rbar],
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 11.72 / 4,  # Capital-output ratio (11.22 US)
        "B/K" => 0.25,  # Liquid to illiquid ratio
        "G/Y" => 0.22,  # Gov. spending-output ratio (0.22 GER)
        "T10W" => 0.67,  # Top 10% wealth share
        "Frac Borrowers" => 0.16,  # Fraction of borrowers
        # "GI/Y" => 0.03,  # Gov. investment to output ratio
    ),
    # One must change options for their respective setting!
    "opt_options" => Optim.Options(;
        time_limit = 10800,
        show_trace = true,
        show_every = 10, # iteration count
        f_reltol = 1e-3,   # stops if fitness ≤ tolerance
    ),
);


#= For BBO
cal_dict_BBO = Dict(
    "params_to_calibrate" => [:β, :λ, :Tlev, :ζ, :Rbar],
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 11.22 / 4,  # Capital-output (quarterly) ratio
        "B/K" => 0.25,  # Liquid to illiquid ratio
        "G/Y" => 0.20,  # Gov. spending-output (annualy) ratio
        "T10W" => 0.67,  # Top 10% wealth share
        "Frac Borrowers" => 0.16,  # Fraction of borrowers
    ),
    # One must change options for their respective setting!
    "opt_options" => (
        SearchRange=[
            (0.90, 0.999), # β
            (0.01, 0.2), # λ
            (1.0, 1.5), # Tlev
            (0.0, 0.0005), # ζ
            (0.0, 0.05), # Rbar
        ],
        Method=:adaptive_de_rand_1_bin_radiuslimited,
        MaxTime=10800, # 3 hours
        TraceInterval=30,
        TraceMode=:compact,
        TargetFitness=1e-3,   # stops if fitness ≤ tolerance
    ),
);
=#

# Run calibration. Exports parameters
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
G = exp.(sr_full.XSS[sr_full.indexes.GSS]);
fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0);
# new 
GI = exp.(sr_full.XSS[sr_full.indexes.GISS]);
KG = exp.(sr_full.XSS[sr_full.indexes.KGSS]);
Sp = exp.(sr_full.XSS[sr_full.indexes.SpSS]);

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
IRFs, _, IRFs_order = compute_irfs( # , IRFs_dist
    exovars,
    lr_full.State2Control,
    lr_full.LOMstate,
    sr_full.XSS,
    sr_full.indexes;
    init_val = stds,
    distribution = false,
    comp_ids = sr_full.compressionIndexes,
    n_par = sr_full.n_par,
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
        (:ZI, "Inv.-spec. tech."),
        (:μ, "Price markup"),
        (:μw, "Wage markup"),
        (:A, "Risk premium"),
        (:Rshock, "Mon. policy"),
        (:Gshock, "Structural deficit"),
        (:GI, "Gov. Investment"),
        (:Tprogshock, "Tax progr."),
        (:Sshock, "Income risk"),
    ],
    [
        (:Ygrowth, "Output growth"),
        (:Cgrowth, "Consumption growth"),
        #(:GI, "Gov. Investment"),
        (:Igrowth, "Investment growth"),
        (:N, "Employment"),
        (:wgrowth, "Wage growth"),
        (:RB, "Nominal rate"),
        (:π, "Inflation"),
        (:σ, "Income risk"),
        (:Tprog, "Tax progressivity"),
        (:TOP10Wshare, "Top 10 wealth share"),
        (:TOP10Ishare, "Top 10 inc. share"),
    ],
    [(IRFs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
);

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")