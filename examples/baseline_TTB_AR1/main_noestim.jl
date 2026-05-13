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

# @set! m_par.ξ = 4.0;
# @set! m_par.γ = 2.0;
# @set! m_par.β = 0.991673574071822;
# @set! m_par.λ = 0.02234890580840817; 
# @set! m_par.ρ_h = 0.9815;
# @set! m_par.σ_h = 0.135;
# @set! m_par.ι = 0.0625;
# @set! m_par.ζ = 0.0005; 
# @set! m_par.α = 0.32;
# @set! m_par.δ_0 = 0.015;
# @set! m_par.δ_s = 0.7180368636852436;
# @set! m_par.ϕ = 1.966794427917757;
# @set! m_par.μ = 1.05;
# @set! m_par.κ = 0.1479251925067033;
# @set! m_par.μw = 1.05;
# @set! m_par.κw = 0.24550729545132235;
# @set! m_par.Tlev = 1.0 + 0.1225298495374623756;
# @set! m_par.Tprog = 1.0 + 0.2;
# @set! m_par.Tc = 1.0;
# @set! m_par.Tk = 1.0;
# @set! m_par.Ttr_1 = 1.55;
# @set! m_par.Ttr_2 = 1.8;
# @set! m_par.RRB = 1.0;
# @set! m_par.Rbar = 0.036798968157815665;
# @set! m_par.ωΠ = 0.2;
# @set! m_par.ιΠ = 0.016;
# @set! m_par.shiftΠ = 0.7138438278245689;
# @set! m_par.ρ_A = 0.9998728292904844;
# @set! m_par.σ_A = 0.0023446421723594727;
# @set! m_par.ρ_ZI = 0.7618447236808487;
# @set! m_par.σ_ZI = 0.07436546031946978;
# @set! m_par.ρ_μ = 0.9733829673391314;
# @set! m_par.σ_μ = 0.014809661702069623;
# @set! m_par.ρ_μw = 0.9110208706720851;
# @set! m_par.σ_μw = 0.036870468298782905;
# @set! m_par.ρ_s = 0.5559631986300858;
# @set! m_par.σ_Sshock = 0.6918558038597916; # 0.0
# @set! m_par.Σ_n = 28.879770107327673; # 0.0
# @set! m_par.ρ_R = 0.8238297433452639;
# @set! m_par.σ_Rshock = 0.0031182372723710205;
# @set! m_par.θ_π = 1.25;
# @set! m_par.θ_Y = 0.0;
# @set! m_par.γ_B = 0.1;
# @set! m_par.γ_π = -2.1989682009232125;
# @set! m_par.γ_Y = -0.4417816197316064;
# @set! m_par.ρ_Gshock = 0.9986388882841384;
# @set! m_par.σ_Gshock = 0.01; # 0.004712013201311698;
# @set! m_par.ρ_τ = 0.5052588859076916;
# @set! m_par.γ_Bτ = 0.0;
# @set! m_par.γ_Yτ = -0.9430455116131855;
# @set! m_par.ρ_P = 0.9410326697915214;
# @set! m_par.σ_Tprogshock = 0.0;
# @set! m_par.γ_BP = 0.0;
# @set! m_par.γ_YP = 0.0;
# @set! m_par.γ_WP = 0.0;
# @set! m_par.ρ_Rshock = 1.0e-8;
# @set! m_par.ρ_Tprogshock = 1.0e-8;
# @set! m_par.ρ_Sshock = 1.0e-8;
# @set! m_par.ρ_TFP = 0.5045353093162687;     
# @set! m_par.σ_TFP = 0.025251212935269763; 

# # new govt investment parameters
# @set! m_par.γ_GI = 1.0;                    # Deficit reaction to GI (0 = tax financed, 1 = debt)
# @set! m_par.GI_share = 0.028;              # Steady state share of govt investment
# @set! m_par.δ_KG = 0.01;                   # Depreciation of public capital
# @set! m_par.η_KG = 0.1;                    # Elasticity of output w.r.t public capital
# @set! m_par.ρ_GI = 0.5209852270953002;     # Spending is announced once; persistence originates from TTB
# @set! m_par.σ_GI = 0.01; # 0.005158611558252573 


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
idx_dict_hank_AR1 = Dict{Symbol, Int}(
    name => getfield(sr_full.indexes, name) 
    for name in fieldnames(typeof(sr_full.indexes)) 
    if getfield(sr_full.indexes, name) isa Int
)
jldsave(paths["bld_example"] * "/IRFs_HANK_AR1.jld2", true; 
    IRFs, 
    IRFs_order, 
    IRFs_dist,
    idx_dict_hank_AR1
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
horizon = 80; 

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
    #(:σ, "Income risk"),
    #(:Tprog, "Tax progressivity"),
    #(:TOP10Wshare, "Top 10 wealth share"),
    #(:TOP10Ishare, "Top 10 gross inc. share"),
    #(:TOP10Inetshare, "Top 10 net inc. share"),
    (:GiniW, "Wealth Gini"),
    (:GiniC, "Consumption Gini") 
];

# IRFs
mkpath(paths["bld_example"] * "/IRFs");
plot_irfs(
    shocks_to_plot,
    vars_to_plot,
    [(IRFs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    horizon,
    save_fig_indiv = false,
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
        ("Fiscal", "fis") => [:Gshock, :GI], # :Tprogshock
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

## -----------
## Comparison
## -----------
# Load the smoothed data
path_smoothed = joinpath(paths["bld"], "baseline_TTB_smooth_noestim", "IRFs_smoothed.jld2")
IRFs_smooth       = load(path_smoothed, "IRFs")
idx_dict_smooth   = load(path_smoothed, "idx_dict_smooth")
IRFs_order_smooth = load(path_smoothed, "IRFs_order")

# Active AR(1) data
IRFs_front = IRFs
ids_front  = sr_full.indexes

# Scale Gini coefficients
IRFs_front[ids_front.GiniW, :, :] ./= 100.0
IRFs_front[ids_front.GiniC, :, :] ./= 100.0
IRFs_smooth[idx_dict_smooth[:GiniW], :, :] ./= 100.0
IRFs_smooth[idx_dict_smooth[:GiniC], :, :] ./= 100.0

IRFs_front[ids_front.π, :, :] ./= 100.0
IRFs_front[ids_front.RB, :, :] ./= 100.0
IRFs_smooth[idx_dict_smooth[:π], :, :] ./= 100.0
IRFs_smooth[idx_dict_smooth[:RB], :, :] ./= 100.0

# ---------------------------------------------------------
# Map MA(39) to AR(1) dimensions (Variables AND Shocks)
# ---------------------------------------------------------
IRFs_smooth_aligned = zeros(size(IRFs_front))

# Rename Auth to GI in the loaded shock order array
IRFs_order_smooth[IRFs_order_smooth .== :Auth] .= :GI

# Map variables (1st dimension) and shocks (3rd dimension)
for (var, idx_smooth_var) in idx_dict_smooth
    if hasproperty(ids_front, var)
        idx_front_var = getfield(ids_front, var)
        
        # Check if they are integers AND within the matrix row bounds
        if idx_front_var isa Int && idx_smooth_var isa Int
            if 0 < idx_front_var <= size(IRFs_front, 1) && 0 < idx_smooth_var <= size(IRFs_smooth, 1)
                for (front_shock_idx, shock_name) in enumerate(IRFs_order)
                    # Find where this shock is located in the MA(39) matrix
                    smooth_shock_idx = findfirst(==(shock_name), IRFs_order_smooth)
                    
                    if !isnothing(smooth_shock_idx)
                        IRFs_smooth_aligned[idx_front_var, :, front_shock_idx] = IRFs_smooth[idx_smooth_var, :, smooth_shock_idx]
                    end
                end
            end
        end
    end
end

# Define the plot comparison
IRFs_to_plot = [
    (IRFs_front, "Front-loaded AR(1)"),
    (IRFs_smooth_aligned, "10-Year Plan")
]

# Plot combined IRFs using the active AR(1) IDs and shock order
mkpath(paths["bld_example"] * "/IRFs_comparison");
plot_irfs(
    shocks_to_plot,
    vars_to_plot,
    IRFs_to_plot,
    IRFs_order,
    ids_front;
    horizon = 80,
    save_fig = true,
    path = joinpath(paths["bld_example"], "/IRFs_comparison"),
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash])
)


#=
# Variance decomposition
mkpath(paths["bld_example"] * "/VDs");
plot_vardecomp(
    vars_to_plot,
    [(VDs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/VDs",
);

mkpath(paths["bld_example"] * "/VDs_cat");
plot_vardecomp(
    vars_to_plot,
    [(VDs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    shock_categories = Dict(
        ("Monetary", "mon") => [:Rshock, :A],
        ("Fiscal", "fis") => [:Gshock, :Tprogshock, :GI],
        ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
    ),
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/VDs_cat",
);

# Business cycle frequency variance decomposition
mkpath(paths["bld_example"] * "/VDbcs");
plot_vardecomp_bcfreq(
    vars_to_plot,
    [(VDbcs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/VDbcs",
);

mkpath(paths["bld_example"] * "/VDbcs_cat");
plot_vardecomp_bcfreq(
    vars_to_plot,
    [(VDbcs, "Baseline")],
    IRFs_order,
    sr_full.indexes;
    shock_categories = Dict(
        ("Monetary", "mon") => [:Rshock, :A],
        ("Fiscal", "fis") => [:Gshock, :Tprogshock, :GI],
        ("Productivity", "pro") => [:TFP, :ZI, :μ, :μw],
    ),
    show_fig = false,
    save_fig = true,
    path = paths["bld_example"] * "/VDbcs_cat",
);

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
=#

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")