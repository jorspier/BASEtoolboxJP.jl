root_dir = replace(Base.current_project(), "Project.toml" => "")
cd(root_dir)
include(joinpath(root_dir, "src", "Preprocessor", "PreprocessInputs.jl"))
include(joinpath(root_dir, "src", "BASEforHANK.jl"))
using .BASEforHANK
using JLD2


# --------------------------
## 10Y vs. AR(1) comparison
# --------------------------
# Define paths
path_AR1 = joinpath(paths["bld"], "baseline_TTB_AR1_noestim", "IRFs_HANK_AR1.jld2")
path_10Y = joinpath(paths["bld"], "baseline_TTB_10Y_noestim", "IRFs_HANK_10Y.jld2")

# Load matrices, shock orders, and index structs
IRFs_AR1 = load(path_AR1, "IRFs")
ids_AR1  = load(path_AR1, "indexes")

IRFs_10Y = load(path_10Y, "IRFs")
ids_10Y  = load(path_10Y, "indexes")
IRFs_order  = load(path_10Y, "IRFs_order")

# Scale Gini coefficients using their respective indices
IRFs_AR1[ids_AR1.GiniW, :, :] ./= 100.0
IRFs_AR1[ids_AR1.GiniC, :, :] ./= 100.0

IRFs_10Y[ids_10Y.GiniW, :, :] ./= 100.0
IRFs_10Y[ids_10Y.GiniC, :, :] ./= 100.0

# Create an aligned matrix for the front-loaded IRFs matching the 10Y dimensions
IRFs_front_aligned = zeros(size(IRFs_10Y))

# Map the rows of shared variables from the front-loaded matrix to the aligned matrix
for var in fieldnames(typeof(ids_10Y))
    if hasproperty(ids_AR1, var)
        idx_10Y = getfield(ids_10Y, var)
        idx_AR1  = getfield(ids_AR1, var)
        
        # Ensure the variable is an assigned index (usually > 0)
        if idx_10Y isa Int && idx_AR1 isa Int && idx_10Y > 0 && idx_AR1 > 0
            IRFs_front_aligned[idx_10Y, :, :] = IRFs_AR1[idx_AR1, :, :]
        end
    end
end

# Define the plot comparison using the aligned matrix
IRFs_to_plot = [
    (IRFs_AR1_aligned, "Front-loaded AR(1)"),
    (IRFs_10Y, "10-Year Plan")
]

# Plot combined IRFs (pass ids_10Y since both matrices now follow its structure)
plot_irfs(
    shocks_to_plot,
    vars_to_plot,
    IRFs_to_plot,
    IRFs_order,
    ids_10Y;
    horizon = 80,
    save_fig = true,
    path = joinpath(paths["bld"], "IRFs_comparison_AR1"),
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash])
)

# --------------------------
## HANK vs. Copmlete Markets comparison
# --------------------------

path_CM = joinpath(paths["bld"], "baseline_complete_markets_noestim", "IRFs_CM_10Y.jld2")

IRFs_CM = load(path_CM, "IRFs")
ids_CM  = load(path_CM, "indexes")
IRFs_order  = load(path_CM, "IRFs_order")

IRFs_CM[ids_CM.GiniW, :, :] ./= 100.0
IRFs_CM[ids_CM.GiniC, :, :] ./= 100.0

IRFs_to_plot_CM = [
    (IRFs_CM, "Complete Markets"),
    (IRFs_10Y, "10-Year Plan HANK")
]

plot_irfs(
    shocks_to_plot,
    vars_to_plot,
    IRFs_to_plot_CM,
    IRFs_order,
    ids_10Y;
    horizon = 80,
    save_fig = true,
    path = joinpath(paths["bld"], "IRFs_comparison_CM"),
    style_options = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash])
)