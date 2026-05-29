"""
IRF_comparison.jl

Standalone script that loads pre-computed IRFs from multiple model variants and
produces comparison plots. Run the four scripts listed below first to generate
the required JLD2 files.

Pre-requisites (run in order, renaming the export file as indicated):
  1. examples/baseline_TTB_AR1/main_noestim.jl  (TTB, debt-financed)
     → bld/baseline_TTB_AR1_noestim/IRFs_HANK_AR1_debt.jld2
  2. Same script with tax-financing rule
     → bld/baseline_TTB_AR1_noestim/IRFs_HANK_AR1_tax.jld2
  3. Same script without TTB constraint
     → bld/baseline_TTB_AR1_noestim/IRFs_HANK_AR1_noTTB.jld2
  4. examples/baseline_complete_markets/main_noestim.jl
     → bld/baseline_complete_markets_noestim/IRFs_CM_noTTB.jld2

Comparisons produced:
  1. HANK AR1 (TTB, debt) vs. Complete Markets
  2. Debt-financed vs. Tax-financed (AR1, TTB)
  3. TTB vs. No-TTB (AR1, debt-financed)
"""

# ==============================================================================
# Setup
# ==============================================================================

root_dir = replace(Base.current_project(), "Project.toml" => "")
cd(root_dir)

using JLD2, Printf

bld_ar1  = joinpath(root_dir, "bld", "baseline_TTB_AR1_noestim")
bld_cm   = joinpath(root_dir, "bld", "baseline_complete_markets_noestim")
bld_comp = joinpath(root_dir, "bld", "IRF_comparisons")
mkpath(bld_comp)

# Load BASEforHANK using the AR1 model as the preprocessing reference
paths = Dict(
    "root"        => root_dir,
    "src"         => joinpath(root_dir, "src"),
    "bld"         => joinpath(root_dir, "bld"),
    "src_example" => joinpath(root_dir, "examples", "baseline_TTB_AR1"),
    "bld_example" => bld_ar1,
)
include(paths["src"] * "/Preprocessor/PreprocessInputs.jl")
include(paths["src"] * "/BASEforHANK.jl")
using .BASEforHANK

# ==============================================================================
# Helper functions
# ==============================================================================

"""
    load_irfs(filepath)

Load an IRF JLD2 file. Returns `(IRFs, IRFs_order, idx_dict)` regardless of
the exact variable name used for the index dictionary when the file was saved.
"""
function load_irfs(filepath)
    f = load(filepath)
    IRFs       = f["IRFs"]
    IRFs_order = f["IRFs_order"]
    idx_key    = first(k for k in keys(f) if startswith(k, "idx_dict"))
    return IRFs, IRFs_order, f[idx_key]::Dict{Symbol,Int}
end

"""
    dict_to_nt(d)

Convert a `Dict{Symbol,Int}` to a `NamedTuple`. The result supports
`hasfield` / `getfield` and is therefore compatible with `plot_irfs`.
"""
dict_to_nt(d::Dict{Symbol,Int}) = NamedTuple(d)

"""
    align_irfs(IRFs_src, idx_src, order_src, IRFs_ref, idx_ref, order_ref)

Re-index `IRFs_src` (from a source model) into the row and shock-column order
of a reference model. Rows / shocks not shared between models are left as zero.
"""
function align_irfs(IRFs_src, idx_src, order_src, IRFs_ref, idx_ref, order_ref)
    n_ref, T, n_shocks = size(IRFs_ref)
    aligned = zeros(n_ref, T, n_shocks)
    for (var, i_ref) in idx_ref
        endswith(string(var), "SS") && continue   # ← add this
        haskey(idx_src, var) || continue
        i_src = idx_src[var]
        for (j_ref, shock) in enumerate(order_ref)
            j_src = findfirst(==(shock), order_src)
            isnothing(j_src) && continue
            aligned[i_ref, :, j_ref] = IRFs_src[i_src, :, j_src]
        end
    end
    return aligned
end


"""
    prescale_rows(IRFs, ids, vars; scale = 4.0)

Return a copy of `IRFs` with selected variable rows multiplied by `scale`.
Use this to bake a per-variable factor into the array before a uniform
`factor=100` call to `plot_irfs` — e.g. multiply quarterly-rate rows by 4 so
that `factor=100` gives annualised pp while quantity rows stay as % change.
Variables absent from `ids` are silently skipped.
"""
function prescale_rows(
    IRFs::Array{Float64,3},
    ids,
    vars::Vector{Symbol};
    scale::Float64 = 4.0,
)
    out = copy(IRFs)
    for var in vars
        hasproperty(ids, var) || continue
        i = getfield(ids, var)
        out[i, :, :] .*= scale
    end
    return out
end

"""
    apply_scales(IRFs, IRFs_order, scales)

Return a copy of `IRFs` with selected shock columns multiplied by the factors
in `scales::Dict{Symbol,Float64}`.
"""
function apply_scales(
    IRFs::Array{Float64,3},
    IRFs_order::Vector{Symbol},
    scales::Dict{Symbol,Float64},
)
    out = copy(IRFs)
    for (shock, s) in scales
        idx = findfirst(==(shock), IRFs_order)
        isnothing(idx) && continue
        out[:, :, idx] .*= s
    end
    return out
end

# ==============================================================================
# Load pre-computed IRFs
# ==============================================================================

@printf "Loading IRF files...\n"

IRFs_HANK_noTTB, order_hank, idx_HANK_noTTB = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB.jld2"));
IRFs_HANK_TTB4, _, idx_HANK_TTB4  = load_irfs(joinpath(bld_ar1, "IRFs_HANK_TTB4.jld2"));
IRFs_HANK_TTB12, _, idx_HANK_TTB12 = load_irfs(joinpath(bld_ar1, "IRFs_HANK_TTB12.jld2"));
IRFs_HANK_noTTB_eta5, _, idx_HANK_noTTB_eta5 = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_eta5.jld2"));
IRFs_HANK_noTTB_eta20, _, idx_HANK_noTTB_eta20 = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_eta20.jld2"));

#IRFs_tax, _, idx_tax   = load_irfs(joinpath(bld_ar1, "IRFs_HANK_tax.jld2"));

IRFs_CM_noTTB,    order_cm,  idx_CM_noTTB    = load_irfs(joinpath(bld_cm,  "IRFs_CM_noTTB.jld2"));
IRFs_CM_TTB4,     _,         idx_CM_TTB4   = load_irfs(joinpath(bld_cm,  "IRFs_CM_TTB4.jld2"));
IRFs_CM_TTB12,    _,         idx_CM_TTB12  = load_irfs(joinpath(bld_cm,  "IRFs_CM_TTB12.jld2"));
IRFs_CM_noTTB_eta5, _, idx_CM_noTTB_eta5 = load_irfs(joinpath(bld_cm,  "IRFs_CM_noTTB_eta5.jld2"));
IRFs_CM_noTTB_eta20, _, idx_CM_noTTB_eta20 = load_irfs(joinpath(bld_cm,  "IRFs_CM_noTTB_eta20.jld2"));
IRFs_CM_noTTB_HANKpars, _, idx_CM_noTTB_HANKpars = load_irfs(joinpath(bld_cm,  "IRFs_CM_noTTB_HANKpars.jld2"));

# Build a NamedTuple index handle for the AR1 model (identical across all AR1 variants)
ids_hank = dict_to_nt(filter(p -> !endswith(string(p.first), "SS"), idx_HANK_noTTB));


# ==============================================================================
# Shock scaling
# σ_GI and σ_Gshock were set to 10% of their true values to keep the
# linearisation valid. Since IRFs scale linearly, multiply those columns by 10
# so plots reflect true shock magnitudes. The raw JLD2 files are unscaled.
# ==============================================================================

scales = Dict{Symbol,Float64}(:GI => 10.0, :Gshock => 10.0);

IRFs_HANK_noTTB_plot  = apply_scales(IRFs_HANK_noTTB,  order_hank, scales);
IRFs_HANK_TTB4_plot = apply_scales(IRFs_HANK_TTB4, order_hank, scales);
IRFs_HANK_TTB12_plot = apply_scales(IRFs_HANK_TTB12, order_hank, scales);

IRFs_HANK_noTTB_eta5_plot  = apply_scales(IRFs_HANK_noTTB_eta5,  order_hank, scales);
IRFs_HANK_noTTB_eta20_plot  = apply_scales(IRFs_HANK_noTTB_eta20,  order_hank, scales);

#IRFs_tax_plot   = apply_scales(IRFs_tax,   order_hank, scales);


# Align complete-markets IRFs to the AR1 row / shock-column structure, then scale
IRFs_CM_noTTB_aligned      = align_irfs(IRFs_CM_noTTB, idx_CM_noTTB, order_cm, IRFs_HANK_noTTB, idx_HANK_noTTB, order_hank);
IRFs_CM_noTTB_aligned_plot = apply_scales(IRFs_CM_noTTB_aligned, order_hank, scales);

IRFs_CM_noTTB_HANKpars_aligned = align_irfs(IRFs_CM_noTTB_HANKpars, idx_CM_noTTB_HANKpars, order_cm, IRFs_HANK_noTTB, idx_HANK_noTTB, order_hank);
IRFs_CM_noTTB_HANKpars_aligned_plot = apply_scales(IRFs_CM_noTTB_HANKpars_aligned, order_hank, scales);

# Pre-scale quarterly-rate rows ×4 so that factor=100 in plot_irfs gives
# annualised pp for π, RB, LPXA, RR while quantity rows remain as % change.
# This lets all variables appear in one combined figure with a single call.
rate_vars = [:π, :RB, :LPXA]

IRFs_HANK_noTTB_paper                = prescale_rows(IRFs_HANK_noTTB_plot,                   ids_hank, rate_vars)
IRFs_HANK_TTB4_paper                 = prescale_rows(IRFs_HANK_TTB4_plot,                    ids_hank, rate_vars)
IRFs_HANK_TTB12_paper                = prescale_rows(IRFs_HANK_TTB12_plot,                   ids_hank, rate_vars)
IRFs_HANK_noTTB_eta5_paper           = prescale_rows(IRFs_HANK_noTTB_eta5_plot,              ids_hank, rate_vars)
IRFs_HANK_noTTB_eta20_paper          = prescale_rows(IRFs_HANK_noTTB_eta20_plot,             ids_hank, rate_vars)
IRFs_CM_noTTB_aligned_paper          = prescale_rows(IRFs_CM_noTTB_aligned_plot,             ids_hank, rate_vars)
IRFs_CM_noTTB_HANKpars_aligned_paper = prescale_rows(IRFs_CM_noTTB_HANKpars_aligned_plot,   ids_hank, rate_vars)

# ==============================================================================
# Common plot specification
# ==============================================================================

horizon = 80;

shocks_to_plot = [
    (:GI, "Gov. Investment"),
    #(:Gshock,  "Gov. Consumption"),
];

# Quantity rows: factor=100 → % change.
# Rate rows (π, RB, LPXA): pre-scaled ×4 in _paper arrays → factor=100 gives annualised pp.
vars_agg = [
    (:GI,   "Gov. Investment"),
    (:KG,   "Public Capital"),
    (:Bgov, "Gov. Debt"),
    (:G,    "Gov. Consumption"),
    (:Y,    "Output"),
    (:C,    "Consumption"),
    (:I,    "Investment"),
    (:N,    "Employment"),
    #(:wH,  "Real Wage"),
    (:π,    "Inflation"),
    (:RB,   "Nominal Rate"),
    (:T,    "Tax Revenue"),
    (:TR,   "Transfers"),
    #(:LPXA, "Ex ante liq. premium"),
];

vars_dist = [
    (:GiniC,       "Gini of Consumption"),
    (:GiniW,       "Gini of Wealth"),
    (:GiniI,       "Gini of Income"),
    #(:GiniInet,    "Gini of Net Income"),
    (:TOP10Wshare, "Top 10% Wealth Share"),
    (:FrBorr,       "Fraction of Borrowers"),
    (:TOP10Ishare, "Top 10% Income Share"),
    (:BOT50Wshare, "Bot 50% Wealth Share"),
    (:BOT50Ishare, "Bot 50% Income Share"),
];

style_12 = (lw = 2, color = [:blue, :red, :orange], linestyle = [:solid, :dash, :dot]);
style_3 = (lw = 2, color = [:red, :blue, :orange], linestyle = [:dash, :solid, :dot]);

## =============================================================================
# Comparison 1: HANK AR1 (TTB, debt-financed) vs. Complete Markets
# ==============================================================================

@printf "\n[1/3] HANK (TTB, debt) vs. Complete Markets\n"
path1 = joinpath(bld_comp, "HANK_vs_CM")

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_HANK_noTTB_paper, "HANK"),
        (IRFs_CM_noTTB_aligned_paper, "RANK"),
        (IRFs_CM_noTTB_HANKpars_aligned_paper, "RANK (HANK pars)")],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path1, "agg"), yscale = "standard", style_options = style_12,
    suffix = "_HANKvsCM",
)

# plot_irfs(
#     shocks_to_plot, vars_dist,
#     [(IRFs_HANK_noTTB_plot, "HANK (TTB, debt)"), (IRFs_CM_noTTB_aligned_plot, "Complete Markets")],
#     order_hank, ids_hank;
#     horizon, save_fig_indiv = false, show_fig = false, save_fig = true,
#     path = joinpath(path1, "dist"), yscale = "standard", style_options = style_12,
# )

## =============================================================================
# Comparison 2: Time-to-Build vs. No Time-to-Build (AR1, debt-financed)
# ==============================================================================

@printf "[2/3] TTB vs. No-TTB (AR1, debt-financed)\n"
path2 = joinpath(bld_comp, "TTB_vs_noTTB")

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_HANK_noTTB_paper, "No TTB"),
        (IRFs_HANK_TTB4_paper, "1-year lag"),
        (IRFs_HANK_TTB12_paper, "3-year lag")],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = false,
    path = joinpath(path2, "agg"), yscale = "standard", style_options = style_12,
    suffix = "_TTBvsNoTTB_agg", 
)

plot_irfs(
    shocks_to_plot, vars_dist,
    [(IRFs_HANK_noTTB_plot, "No TTB"),
        (IRFs_HANK_TTB4_plot, "1-year lag"),
        (IRFs_HANK_TTB12_plot, "3-year lag")],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path2, "dist"), yscale = "standard", style_options = style_12,
    suffix = "_TTBvsNoTTB_dist",
)

## ================== 2B: RANK with different TTB =========================
IRFs_CM_TTB12_aligned_to_noTTB = align_irfs(
    IRFs_CM_TTB12, idx_CM_TTB12, order_cm,
    IRFs_CM_noTTB, idx_CM_noTTB, order_cm
);

IRFs_CM_noTTB_plot    = apply_scales(IRFs_CM_noTTB, order_cm, scales);
IRFs_CM_TTB4_plot     = apply_scales(IRFs_CM_TTB4, order_cm, scales);
IRFs_CM_TTB12_plot    = apply_scales(IRFs_CM_TTB12, order_cm, scales);

ids_cm = dict_to_nt(filter(p -> !endswith(string(p.first), "SS"), idx_CM_noTTB));

IRFs_CM_noTTB_paper    = prescale_rows(IRFs_CM_noTTB_plot, ids_cm, rate_vars)
IRFs_CM_TTB4_paper     = prescale_rows(IRFs_CM_TTB4_plot, ids_cm, rate_vars)
IRFs_CM_TTB12_paper    = prescale_rows(IRFs_CM_TTB12_plot, ids_cm, rate_vars)


plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_CM_noTTB_paper, "No TTB"),
        (IRFs_CM_TTB4_paper, "1-year lag"),
        (IRFs_CM_TTB12_paper, "3-year lag")],
    order_cm, ids_cm;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = false,
    path = joinpath(path2, "agg"), yscale = "standard", style_options = style_12,
    suffix = "_TTBvsNoTTB_CM", 
)

## =============================================================================
# Comparison 3: Different elasticities
# ==============================================================================

@printf "[3/3] Different output elasticities\n"
path3 = joinpath(bld_comp, "elasticities")

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_HANK_noTTB_eta5_paper, "η_KG = 0.05"),
        (IRFs_HANK_noTTB_paper, "η_KG = 0.10"),
        (IRFs_HANK_noTTB_eta20_paper, "η_KG = 0.20")],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path3, "agg"), yscale = "standard", style_options = style_3,
    suffix = "_elast_agg",
)

plot_irfs(
    shocks_to_plot, vars_dist,
    [(IRFs_HANK_noTTB_eta5_plot, "η_KG = 0.05"),
        (IRFs_HANK_noTTB_plot, "η_KG = 0.10"),
        (IRFs_HANK_noTTB_eta20_plot, "η_KG = 0.20")],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path3, "dist"), yscale = "standard", style_options = style_3,
    suffix = "_elast_dist",
)

## ================== 3B: RANK with different elasticties =========================
IRFs_CM_noTTB_eta5_plot  = apply_scales(IRFs_CM_noTTB_eta5,  order_cm, scales);
IRFs_CM_noTTB_eta20_plot = apply_scales(IRFs_CM_noTTB_eta20, order_cm, scales);

IRFs_CM_noTTB_eta5_paper   = prescale_rows(IRFs_CM_noTTB_eta5_plot,  ids_cm, rate_vars)
IRFs_CM_noTTB_eta20_paper  = prescale_rows(IRFs_CM_noTTB_eta20_plot, ids_cm, rate_vars)

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_CM_noTTB_eta5_paper, "η_KG = 0.05"),
        (IRFs_CM_noTTB_paper, "η_KG = 0.10"),
        (IRFs_CM_noTTB_eta20_paper, "η_KG = 0.20")],
    order_cm, ids_cm;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = false,
    path = joinpath(path3, "RANK_agg"), yscale = "standard", style_options = style_3,
    suffix = "_elast_CM"
)

# ==============================================================================
# Comparison 4: Debt-financed vs. Tax-financed (AR1, TTB)
# ==============================================================================

# @printf "[3/4] Debt-financed vs. Tax-financed (AR1, TTB)\n"
# path2 = joinpath(bld_comp, "debt_vs_tax")

# plot_irfs(
#     shocks_to_plot, vars_agg,
#     [(IRFs_debt_plot, "Debt-financed"), (IRFs_tax_plot, "Tax-financed")],
#     order_hank, ids_hank;
#     horizon, save_fig_indiv = false, show_fig = false, save_fig = true,
#     path = joinpath(path4, "agg"), yscale = "standard", style_options = style_2,
# )

# plot_irfs(
#     shocks_to_plot, vars_dist,
#     [(IRFs_debt_plot, "Debt-financed"), (IRFs_tax_plot, "Tax-financed")],
#     order_hank, ids_hank;
#     horizon, save_fig_indiv = false, show_fig = false, save_fig = true,
#     path = joinpath(path4, "dist"), yscale = "standard", style_options = style_2,
# )


@printf "\nDone. All comparison plots saved to:\n  %s\n" bld_comp