using JLD2
using Plots

# ------------------------------------------------------------------------------
## 1. Setup
# ------------------------------------------------------------------------------

base_dir = "c:/GitRepos/BASEtoolboxJP.jl"
include(joinpath(base_dir, "examples/RANK/8_PostEstimation/plot_irfs.jl"))

rank_data = load(joinpath(base_dir, "examples/RANK/7_Saves/IRFs_RANK_export.jld2"))
hank_data = load(joinpath(base_dir, "bld/baseline_TTB_10Y_noestim/IRFs_HANK_export.jld2"))

# ------------------------------------------------------------------------------
## 2. What to plot?
# ------------------------------------------------------------------------------

# Rename the investment shock to observe the actual one of interest
IRFs_order[IRFs_order .== :Auth] .= :GI

target_shocks = [:Gshock, :GI]
nice_s_names = ["Government Spending", "50 Mrd. 10Y Investment"]

comp_vars = [:Y, :C, :I, :N, :w, :π, :RB, :B]
nice_var_names = ["Output", "Consumption", "Private Investment", "Employment", "Real Wage", "Inflation", "Nominal Rate", "Bonds"]

# ------------------------------------------------------------------------------
## 3. Matrix slicing to get the relevant IRFs
# ------------------------------------------------------------------------------

# Find shock index
s_idx_rank = [findfirst(==(s), rank_data["IRFs_order"]) for s in target_shocks]
s_idx_hank = [findfirst(==(s), hank_data["IRFs_order"]) for s in target_shocks]

# Find variable index (different calls bc different functions compute the IRFs)
v_idx_hank = [hank_data["idx_dict_hank_10Y"][v] for v in comp_vars]
v_idx_rank = [findfirst(==(v), rank_data["all_rank_vars"]) for v in comp_vars]

# get arrays
IRF_rank_comp = rank_data["IRFs"][v_idx_rank, :, s_idx_rank]
IRF_hank_comp = hank_data["IRFs"][v_idx_hank, :, s_idx_hank]

# ------------------------------------------------------------------------------
## 4. Plotting of combined IRFs
# ------------------------------------------------------------------------------

plot_irfs(
    [IRF_rank_comp, IRF_hank_comp], 
    target_shocks, 
    comp_vars, 
    nice_var_names, 
    nice_s_names, 
    80, 
    ["RANK", "HANK"], 
    3; 
    savepdf = true, 
    suffix = "_Comparison_RANK_HANK"  
)

# ------------------------------------------------------------------------------
## 5. Plotting of HANK IRFs only
# ------------------------------------------------------------------------------

vars_to_plot_hank = [:KG, :RRL, :RK, :LP, :LPXA, :GiniW, :GiniC]

# get variable indices
v_idx_hank_solo = [hank_data["idx_dict_hank"][v] for v in vars_to_plot_hank]
IRF_hank_solo = hank_data["IRFs"][v_idx_hank_solo, :, s_idx_hank]

plot_irfs(
    [IRF_hank_solo], 
    target_shocks, 
    vars_to_plot_hank, 
    nice_var_names, 
    nice_s_names, 
    80, 
    ["HANK"], 
    3; 
    savepdf = true, 
    suffix = "_HANK"  
)