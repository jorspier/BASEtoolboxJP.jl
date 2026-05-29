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
@set! m_par.β = 0.995;

## ------------------------------------------------------------------------------------------
## Preparing the calibration
## ------------------------------------------------------------------------------------------

function moments_function_example(m_par)
    try
        ss_full = quiet_call(call_find_steadystate, m_par)
        sr_full = quiet_call(call_prepare_linearization, ss_full, m_par)
      
        K    = exp(sr_full.XSS[sr_full.indexes.KSS])
        Y    = exp(sr_full.XSS[sr_full.indexes.YSS])
        G    = exp(sr_full.XSS[sr_full.indexes.GSS])
        Bgov = exp(sr_full.XSS[sr_full.indexes.BgovSS])
        T    = exp(sr_full.XSS[sr_full.indexes.TSS])
        
        return Dict(
            "K/Y"            => (K / Y) / 4.0,
            "Bgov/Y"         => (Bgov / Y) / 4.0,
            "G/Y"            =>  G / Y,
        )

    catch
        return Dict(
            "K/Y"            => 1e6,
            "Bgov/Y"         => 1e6,
            "G/Y"            => 1e6,
        )
    end
end

# Generate dictionary for calibration
using Optim;

# For Nelder-Mead
cal_dict = Dict(
    "params_to_calibrate" => [  :β,   # discount factor
    ],
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 2.7933,  # Capital-output ratio (Bundesbank assumes 3-3.2)
    ),
    # One must change options for their respective setting!
    "opt_options" => Optim.Options(;
        time_limit = 3*60, # 10800 for 3h
        show_trace = true,
        show_every = 10, # iteration count
        f_reltol = 1e-8,   # stops if fitness ≤ tolerance
    ),
);
#

# For BBO
cal_dict_BBO = Dict(
    "params_to_calibrate" => [:β, :ωΠ, :Tlev], 
    "target_moments" => Dict( # User-defined targets # these are from paper
        "K/Y" => 2.7933,  # Capital-output (quarterly) ratio
        "Bgov/Y" => 0.652,
        "G/Y" => 0.232,
    ),
    # One must change options for their respective setting!
    "opt_options" => (
        SearchRange=[
            (0.993, 0.997), # β
            (0.1,0.3),      # ωΠ
            (1.0, 1.30), # Tlev
        ],
        Method=:adaptive_de_rand_1_bin_radiuslimited,
        MaxTime=3*60, 
        TraceInterval=30,
        TraceMode=:compact,
        TargetFitness=1e-5,   # stops if fitness ≤ tolerance
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
G = exp.(sr_full.XSS[sr_full.indexes.GSS]);
KG = exp.(sr_full.XSS[sr_full.indexes.KGSS]);

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "Private Capital to Output Ratio" K / Y/4.0
        "Government Debt to Output Ratio" Bgov / Y/4.0
        "Government Spending to Output Ratio" G/Y
    ];
    header = ["Variable", "Value"],
    title = "Steady State Moments - RANK",
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
IRFs, _, IRFs_order = compute_irfs(
    exovars,
    lr_full.State2Control,
    lr_full.LOMstate,
    sr_full.XSS,
    sr_full.indexes;
    init_val = stds,
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

# Define here once all variables and shocks to plot for all figures 
horizon = 80; 

IRFs_order[IRFs_order .== :Auth] .= :GI

shocks_to_plot = [
    #(:Z, "Effective TFP"),
    (:GI, "Gov. Investment"), 
    #(:TFP, "TFP Shock"),
    #(:ZI, "Inv.-spec. tech."),
    #(:μ, "Price markup"),
    #(:μw, "Wage markup"),
    #(:A, "Risk premium"),
    #(:Rshock, "Mon. policy"),
    (:Gshock, "Structural deficit"),
    #(:Tprogshock, "Tax progr."),
    #(:Sshock, "Income risk"),
]

vars_to_plot = [
    (:GI, "Gov. Investment"),
    (:KG, "Gov. Capital"),
    (:Bgov, "Gov. Debt"),
    (:G, "Gov. Spending"),
    (:Y, "Output"), # removed growth
    (:C, "Private Consumption"),
    (:I, "Private Investment"),
    (:K, "Private Capital"),
    (:N, "Employment"),
    (:wF, "Wage"),
    (:π, "Inflation"),
    (:RB, "Nominal rate"),
    (:Tlev, "Tax Level"),
    (:T, "Net Tax Revenue"),
];


mkpath(paths["bld_example"] * "/IRFs");
plot_irfs(
    shocks_to_plot,
    vars_to_plot,
    [(IRFs, "Complete Markets")],
    IRFs_order,
    sr_full.indexes;
    horizon,
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
        ("Fiscal", "fis") => [:Gshock, :GI], # , :Tprogshock
        #("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
    ),
    vars_to_plot,
    IRFs,
    IRFs_order,
    sr_full.indexes;
    horizon,
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs_cat",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red, :green, :orange], linestyle = [:solid, :dash, :dot]),
);

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")