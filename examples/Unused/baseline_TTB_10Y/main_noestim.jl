"""
Mainboard for the baseline example of the BASEforHANK package, no estimation.
"""
global_start_time = time()

using PrettyTables, Printf, JLD2;

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
    "bld_example" => replace(@__DIR__, "examples" => "bld") * "_noestim",
);

# create bld directory for the current example
mkpath(paths["bld_example"]);

# pre-process user inputs for model setup
include(paths["src"] * "/Preprocessor/PreprocessInputs.jl");
include(paths["src"] * "/BASEforHANK.jl");
using .BASEforHANK;

# set BLAS threads to the number of Julia threads, prevents grabbing all
BASEforHANK.LinearAlgebra.BLAS.set_num_threads(Threads.nthreads());

## ------------------------------------------------------------------------------------------
## Initialize: set up model parameters
## ------------------------------------------------------------------------------------------

m_par = ModelParameters();


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
    title = "Steady State Moments - HANK 10Y",
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


correct_init_vals = [sr_full.m_par.σ_GI_NS]
    
IRFs, _, IRFs_order, IRFs_dist = compute_irfs( # IRFs_dist
    exovars,
    lr_full.State2Control,
    lr_full.LOMstate,
    sr_full.XSS,
    sr_full.indexes;
    T = 100, # reduce horizon for faster computation
    init_val = stds,
    distribution = true,
    comp_ids = sr_full.compressionIndexes,
    transform_elements = transform_elements,
    n_par = sr_full.n_par,
    m_par = sr_full.m_par,
);

# Export IRFs
idx_dict_hank_10Y_debt = Dict{Symbol, Int}(
    name => getfield(sr_full.indexes, name) 
    for name in fieldnames(typeof(sr_full.indexes)) 
    if getfield(sr_full.indexes, name) isa Int
)
jldsave(paths["bld_example"] * "/IRFs_HANK_10Y_debt.jld2", true; 
    IRFs, 
    IRFs_order, 
    IRFs_dist,
    idx_dict_hank_10Y_debt
)

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
horizon = 100; 

IRFs_order[IRFs_order .== :GI_NS] .= :GI

shocks_to_plot = [
    (:GI, "Gov. Investment"), 
]

vars_agg = [
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
    (:T, "Tax Revenue"),
    (:TR, "Transfers"),
];

vars_dist = [
    (:GiniC, "Consumption Gini"),
    (:GiniI, "Pre-tax Income Gini"),
    (:GiniInet, "Net Income Gini"),
    (:GiniW, "Wealth Gini"),
    (:TOP10Ishare, "Top 10 pre-tax inc. share"),
    (:TOP10Inetshare, "Top 10 net inc. share"),
    (:TOP10Wshare, "Top 10 wealth share"),
    (:sdlogy, "Std. log income")
]

# IRFs
mkpath(paths["bld_example"] * "/IRFs_agg");
plot_irfs(
    shocks_to_plot,
    vars_agg,
    [(IRFs, "10Y HANK")],
    IRFs_order,
    sr_full.indexes;
    horizon,
    save_fig_indiv = false,
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs_agg",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
);

mkpath(paths["bld_example"] * "/IRFs_dist");
plot_irfs(
    shocks_to_plot,
    vars_dist,
    [(IRFs, "10Y HANK")],
    IRFs_order,
    sr_full.indexes;
    horizon,
    save_fig_indiv = false,
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs_dist",
    yscale = "standard",
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
);

# mkpath(paths["bld_example"] * "/IRFs_cat");
# plot_irfs_cat(
#     Dict(
#         #("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :GI], # :Tprogshock
#         #("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
#     ),
#     vars_to_plot,
#     IRFs,
#     IRFs_order,
#     sr_full.indexes;
#     horizon,
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/IRFs_cat",
#     yscale = "standard",
#     style_options = (lw = 2, color = [:blue, :red, :green, :orange], linestyle = [:solid, :dash, :dot]),
# );


# # Variance decomposition
# mkpath(paths["bld_example"] * "/VDs");
# plot_vardecomp(
#     vars_to_plot,
#     [(VDs, "10Y HANK")],
#     IRFs_order,
#     sr_full.indexes;
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDs",
# );

# mkpath(paths["bld_example"] * "/VDs_cat");
# plot_vardecomp(
#     vars_to_plot,
#     [(VDs, "10Y HANK")],
#     IRFs_order,
#     sr_full.indexes;
#     shock_categories = Dict(
#         ("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :GI], #:Tprogshock,
#         ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
#     ),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDs_cat",
# );

# # Business cycle frequency variance decomposition
# mkpath(paths["bld_example"] * "/VDbcs");
# plot_vardecomp_bcfreq(
#     vars_to_plot,
#     [(VDbcs, "10Y HANK")],
#     IRFs_order,
#     sr_full.indexes;
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDbcs",
# );

# mkpath(paths["bld_example"] * "/VDbcs_cat");
# plot_vardecomp_bcfreq(
#     vars_to_plot,
#     [(VDbcs, "10Y HANK")],
#     IRFs_order,
#     sr_full.indexes;
#     shock_categories = Dict(
#         ("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :GI], #:Tprogshock,
#         ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
#     ),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDbcs_cat",
# );

# Distributional IRFs
irfs_to_plot = [
    ("Wb_b", "Marginal Value of Bonds, over Bonds"),
    ("Wk_k", "Marginal Value of Capital, over Capital"),
    ("PDF_b", "Marginal PDF of Bonds"),
    ("PDF_k", "Marginal PDF of Capital"),
    ("PDF_bk", "Marginal PDF of Bonds and Capital"),
    ("PDF_bh", "Marginal PDF of Bonds and Human Capital"),
    ("PDF_kh", "Marginal PDF of Capital and Human Capital"),
]

mkpath(paths["bld_example"] * "/IRFs_dist");
plot_distributional_irfs(
    shocks_to_plot,
    irfs_to_plot,
    IRFs_dist,
    IRFs_order,
    sr_full.n_par;
    horizon,
    bounds = Dict(
        "b" => (sr_full.n_par.grid_b[1], 100.0),
        "k" => (sr_full.n_par.grid_k[1], 100.0),
    ),
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/IRFs_dist",
);

mkpath(paths["bld_example"] * "/IRFs_dist_dev");
plot_distributional_irfs_deviation(
    shocks_to_plot,
    irfs_to_plot,
    IRFs_dist,
    IRFs_order,
    sr_full.n_par;
    horizon,
    bounds = Dict(
        "b" => (sr_full.n_par.grid_b[1], 100.0),
        "k" => (sr_full.n_par.grid_k[1], 100.0),
    ),
    show_fig = false,
    save_fig = true, 
    path = paths["bld_example"] * "/IRFs_dist_dev"
)
#

# Print cumulative mutipliers
println("\n--- Cumulative PV Multipliers: Public Investment (GI) ---")
table_GI = compute_pv_multipliers(IRFs, IRFs_order, sr_full.indexes_r, sr_full.XSS, :GI; max_horizon = 100)
display(table_GI)

jldsave(paths["bld_example"] * "/Multipliers_debt.jld2", true; 
    table_GI
)

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")