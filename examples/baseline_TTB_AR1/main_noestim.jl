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
G = exp.(sr_full.XSS[sr_full.indexes.GSS]);
T10W = exp(sr_full.XSS[sr_full.indexes.TOP10WshareSS]);
T10I = exp(sr_full.XSS[sr_full.indexes.TOP10IshareSS]);
T10Inet = exp(sr_full.XSS[sr_full.indexes.TOP10InetshareSS]);
B50W = exp(sr_full.XSS[sr_full.indexes.BOT50WshareSS]);
B50I = exp(sr_full.XSS[sr_full.indexes.BOT50IshareSS]);
B50Inet = exp(sr_full.XSS[sr_full.indexes.BOT50InetshareSS]);
GiniC = exp(sr_full.XSS[sr_full.indexes.GiniCSS]);
GiniI = exp(sr_full.XSS[sr_full.indexes.GiniISS]);
GiniInet = exp(sr_full.XSS[sr_full.indexes.GiniInetSS]);
GiniW = exp(sr_full.XSS[sr_full.indexes.GiniWSS]);
fr_borr = BASEforHANK.eval_cdf(sr_full.distrSS, :b, sr_full.n_par, 0.0);

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "TOP 10 Wealth Share" T10W
        "BOT 50 Wealth Share" B50W
        "TOP 10 Income Share" T10I
        "BOT 50 Income Share" B50I
        "TOP 10 Income Share (Net)" T10Inet
        "BOT 50 Income Share (Net)" B50Inet
        "Gini of Consumption" GiniC
        "Gini of Income" GiniI
        "Gini of Net Income" GiniInet
        "Gini of Wealth" GiniW
        "Fraction of Borrower" fr_borr
        "Liquid to Illiquid Assets Ratio" B/K
        "Private Capital to Output Ratio" K / Y/4.0
        "Government Debt to Output Ratio" Bgov / Y/4.0
        "Government Spending to Output Ratio" G/Y
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

IRFs, _, IRFs_order = compute_irfs( # , IRFs_dist
    exovars,
    lr_full.State2Control,
    lr_full.LOMstate,
    sr_full.XSS,
    sr_full.indexes;
    init_val = stds,
    distribution = false,
    comp_ids = sr_full.compressionIndexes,
    transform_elements = transform_elements,
    n_par = sr_full.n_par,
    m_par = sr_full.m_par,
);

# Export IRFs
idx_dict_HANK_noTTB_tax_Yneg = Dict{Symbol, Int}(
    name => getfield(sr_full.indexes, name)
    for name in fieldnames(typeof(sr_full.indexes))
    if getfield(sr_full.indexes, name) isa Int
        && !endswith(string(name), "SS")
)

# Steady-state scalars needed for multiplier computations in IRF_comparison.jl.
# Shared across all HANK variants (same steady state), so saved once to its own file.
ss_dict = Dict{Symbol, Float64}(
    :Y     => exp(sr_full.XSS[sr_full.indexes.YSS]),
    :C     => exp(sr_full.XSS[sr_full.indexes.CSS]),
    :GI    => exp(sr_full.XSS[sr_full.indexes.GISS]),
    :GiniW => exp(sr_full.XSS[sr_full.indexes.GiniWSS]),
    :GiniI => exp(sr_full.XSS[sr_full.indexes.GiniISS]),
)
jldsave(paths["bld_example"] * "/ss_dict.jld2"; ss_dict)

jldsave(paths["bld_example"] * "/IRFs_HANK_noTTB_tax_Yneg.jld2", true;
    IRFs,
    IRFs_order,
   # IRFs_dist,
    idx_dict_HANK_noTTB_tax_Yneg,
)
# Compute variance decomposition of IRFs
# VDs = compute_vardecomp(IRFs);

# # Compute business cycle frequency variance decomposition
# VDbcs, UnconditionalVar =
#     compute_vardecomp_bcfreq(exovars, stds, lr_full.State2Control, lr_full.LOMstate);

## ------------------------------------------------------------------------------------------
## Graphical outputs
## ------------------------------------------------------------------------------------------

# @printf "\n"
# @printf "Plotting...\n"

# # σ_GI = 0.0882 and σ_Gshock = 0.01555 were set to 10% of their true values
# # (0.882 and 0.1555) to keep the linearization valid. Since IRFs scale linearly,
# # multiply those shock columns by 10 so plots reflect true shock magnitudes.
# # The saved JLD2 and VDs use the original (unscaled) IRFs.
# irf_plot_scales = Dict(:GI => 10.0, :Gshock => 10.0)
# IRFs_plot = copy(IRFs)
# for (shock, scale) in irf_plot_scales
#     idx = findfirst(==(shock), IRFs_order)
#     !isnothing(idx) && (IRFs_plot[:, :, idx] .*= scale)
# end

# # Define here once all variables and shocks to plot for all figures 
# horizon = 80; 

# shocks_to_plot = [
#     (:GI, "Gov. Investment"), 
#     (:Gshock, "Gov. Consumption"),
# ]

# vars_agg = [
#     (:Y, "Output"), # removed growth
#     (:C, "Consumption"),
#     (:Bgov, "Gov. Debt"),
#     (:KG, "Public Capital"), 
#     (:G, "Gov. Spending"),   
#     (:K, "Private Capital"),
#     (:I, "Investment"),
#     (:N, "Employment"),
#     (:wF, "Wage"),
#     (:π, "Inflation"),
#     (:RB, "Nominal rate"),
# ];

# vars_dist = [
#         (:GiniC, "Gini of Consumption"),
#         (:GiniW, "Gini of Wealth"),
#         (:GiniI, "Gini of Income"),
#         (:GiniInet, "Gini of Net Income"),
#         (:TOP10Wshare, "Top 10% Wealth Share"),
#         (:TOP10Ishare, "Top 10% Income Share"),
#         (:BOT50Wshare, "Bot 50% Wealth Share"),
#         (:BOT50Ishare, "Bot 50% Income Share"),
# ]

# # IRFs
# mkpath(paths["bld_example"] * "/IRFs");
# plot_irfs(
#     shocks_to_plot,
#     vars_agg,
#     [(IRFs_plot, "HANK")],
#     IRFs_order,
#     sr_full.indexes;
#     horizon,
#     save_fig_indiv = false,
#     show_fig = true,
#     save_fig = false,
#     path = paths["bld_example"] * "/IRFs",
#     yscale = "standard",
#     style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
# );

# plot_irfs(
#     shocks_to_plot,
#     vars_dist,
#     [(IRFs_plot, "HANK")],
#     IRFs_order,
#     sr_full.indexes;
#     horizon,
#     save_fig_indiv = false,
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/IRFs_dist",
#     yscale = "standard",
#     style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash]),
# );



# mkpath(paths["bld_example"] * "/IRFs_cat");
# plot_irfs_cat(
#     Dict(
#         ("Fiscal", "fis") => [:Gshock, :GI], # :Tprogshock
#     ),
#     vars_agg,
#     IRFs_plot,
#     IRFs_order,
#     sr_full.indexes;
#     horizon,
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/IRFs_cat",
#     yscale = "standard",
#     style_options = (lw = 2, color = [:blue, :red, :green, :orange], linestyle = [:solid, :dash, :dot]),
# );

# plot_irfs_cat(
#     Dict(
#         ("Fiscal", "fis") => [:Gshock, :GI], 
#     ),
#     vars_dist,
#     IRFs_plot,
#     IRFs_order,
#     sr_full.indexes;
#     horizon,
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/IRFs_dist",
#     yscale = "standard",
#     style_options = (lw = 2, color = [:blue, :red, :green, :orange], linestyle = [:solid, :dash, :dot]),
# );

# #=
# # Variance decomposition
# mkpath(paths["bld_example"] * "/VDs");
# plot_vardecomp(
#     vars_to_plot,
#     [(VDs, "Baseline")],
#     IRFs_order,
#     sr_full.indexes;
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDs",
# );

# mkpath(paths["bld_example"] * "/VDs_cat");
# plot_vardecomp(
#     vars_to_plot,
#     [(VDs, "Baseline")],
#     IRFs_order,
#     sr_full.indexes;
#     shock_categories = Dict(
#         ("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :Tprogshock, :GI],
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
#     [(VDbcs, "Baseline")],
#     IRFs_order,
#     sr_full.indexes;
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDbcs",
# );

# mkpath(paths["bld_example"] * "/VDbcs_cat");
# plot_vardecomp_bcfreq(
#     vars_to_plot,
#     [(VDbcs, "Baseline")],
#     IRFs_order,
#     sr_full.indexes;
#     shock_categories = Dict(
#         ("Monetary", "mon") => [:Rshock, :A],
#         ("Fiscal", "fis") => [:Gshock, :Tprogshock, :GI],
#         ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
#     ),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/VDbcs_cat",
# );
# =#

# # Distributional IRFs
# irfs_to_plot = [
#     ("Wb_b", "Marginal Value of Bonds, over Bonds"),
#     ("Wk_k", "Marginal Value of Capital, over Capital"),
#     ("PDF_b", "Marginal PDF of Bonds"),
#     ("PDF_k", "Marginal PDF of Capital"),
#     ("PDF_bk", "Marginal PDF of Bonds and Capital"),
#     ("PDF_bh", "Marginal PDF of Bonds and Human Capital"),
#     ("PDF_kh", "Marginal PDF of Capital and Human Capital"),
# ]

# mkpath(paths["bld_example"] * "/IRFs_dist");
# plot_distributional_irfs(
#     shocks_to_plot,
#     irfs_to_plot,
#     IRFs_dist,
#     IRFs_order,
#     sr_full.n_par;
#     horizon,
#     bounds = Dict(
#         "b" => (sr_full.n_par.grid_b[1], 100.0),
#         "k" => (sr_full.n_par.grid_k[1], 100.0),
#     ),
#     show_fig = false,
#     save_fig = true,
#     path = paths["bld_example"] * "/IRFs_dist",
# );

# mkpath(paths["bld_example"] * "/IRFs_dist_dev");
# plot_distributional_irfs_deviation(
#     shocks_to_plot,
#     irfs_to_plot,
#     IRFs_dist,
#     IRFs_order,
#     sr_full.n_par;
#     horizon,
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