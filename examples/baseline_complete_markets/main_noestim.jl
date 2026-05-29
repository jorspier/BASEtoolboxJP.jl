"""
Mainboard for a complete markets version of the baseline example, no estimation.
"""

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
T = exp.(sr_full.XSS[sr_full.indexes.TSS]);

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "Liquid to Illiquid Assets Ratio" B/K
        "Capital to Output Ratio" K / Y/4.0
        "Government Debt to Output Ratio" Bgov / Y/4.0
        "Government Spending to Output Ratio" G/Y
        "Net Tax Revenue to Output Ratio" T/Y
    ];
    header = ["Variable", "Value"],
    title = "Steady State Moments - Complete Markets",
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

# Export IRFs
idx_dict_CM_noTTB_HANKpars = Dict{Symbol, Int}(
    name => getfield(sr_full.indexes, name) 
    for name in fieldnames(typeof(sr_full.indexes)) 
    if getfield(sr_full.indexes, name) isa Int
        && !endswith(string(name), "SS")
)
jldsave(paths["bld_example"] * "/IRFs_CM_noTTB_HANKpars.jld2", true; 
    IRFs, 
    IRFs_order, 
    # IRFs_dist,
    idx_dict_CM_noTTB_HANKpars
)

# Compute variance decomposition of IRFs
# VDs = compute_vardecomp(IRFs);

# # Compute business cycle frequency variance decomposition
# VDbcs, UnconditionalVar =
#     compute_vardecomp_bcfreq(exovars, stds, lr_full.State2Control, lr_full.LOMstate);

## ------------------------------------------------------------------------------------------
## Graphical outputs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Plotting...\n"

irf_plot_scales = Dict(:GI => 10.0, :Gshock => 10.0)
IRFs_plot = copy(IRFs)
for (shock, scale) in irf_plot_scales
    idx = findfirst(==(shock), IRFs_order)
    !isnothing(idx) && (IRFs_plot[:, :, idx] .*= scale)
end

# Define here once all variables and shocks to plot for all figures 
horizon = 80; 

shocks_to_plot = [
    (:GI, "Gov. Investment"), 
    #(:Gshock, "Gov. Consumption"),
]

vars_agg = [
    (:Y, "Output"), # removed growth
    (:C, "Consumption"),
    (:G, "Gov. Consumption"),
    (:Bgov, "Gov. Debt"),
    (:KG, "Public Capital"),    
    (:K, "Private Capital"),
    (:I, "Investment"),
    (:N, "Employment"),
    (:wF, "Wage"),
    (:π, "Inflation"),
    (:RB, "Nominal rate"),
    (:T, "Tax Revenue"),
];


# IRFs
mkpath(paths["bld_example"] * "/IRFs_agg");
plot_irfs(
    shocks_to_plot,
    vars_agg,
    [(IRFs, "10Y CM")],
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


mkpath(paths["bld_example"] * "/IRFs_cat");
plot_irfs_cat(
    Dict(
        ("Fiscal", "fis") => [:Gshock, :GI],
    ),
    vars_agg,
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



# Print cumulative mutipliers
println("\n--- Cumulative PV Multipliers: Public Investment (GI) ---")
table_GI = compute_pv_multipliers(IRFs, IRFs_order, sr_full.indexes_r, sr_full.XSS, :GI; max_horizon = 100)
display(table_GI)

# jldsave(paths["bld_example"] * "/Multipliers_CM_debt_TTB12.jld2", true; 
#     table_GI
# )