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

using JLD2, Printf, CSV, DataFrames

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

# Steady-state scalars — each model family saves one shared file
ss_hank = load(joinpath(bld_ar1, "ss_dict.jld2"))["ss_dict"]::Dict{Symbol,Float64};
ss_cm   = load(joinpath(bld_cm,  "ss_dict.jld2"))["ss_dict"]::Dict{Symbol,Float64};

IRFs_HANK_noTTB,      order_hank, idx_HANK_noTTB      = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB.jld2"));

IRFs_HANK_TTB4,       _,          idx_HANK_TTB4        = load_irfs(joinpath(bld_ar1, "IRFs_HANK_TTB4.jld2"));
IRFs_HANK_TTB12,      _,          idx_HANK_TTB12       = load_irfs(joinpath(bld_ar1, "IRFs_HANK_TTB12.jld2"));

IRFs_HANK_noTTB_eta5, _,          idx_HANK_noTTB_eta5  = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_eta5.jld2"));
IRFs_HANK_noTTB_eta20,_,          idx_HANK_noTTB_eta20 = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_eta20.jld2"));

IRFs_HANK_noTTB_noStab, _, idx_HANK_noTTB_noStab = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_noStab.jld2"));
IRFs_HANK_noTTB_noPiStab, _, idx_HANK_noTTB_noPiStab = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_noPiStab.jld2"));

IRFs_HANK_noTTB_tr03, _, idx_HANK_noTTB_tr03 = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_tr03.jld2"));
IRFs_HANK_noTTB_tr2,  _, idx_HANK_noTTB_tr2  = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_tr2.jld2"));

IRFs_HANK_noTTB_tax, _, idx_HANK_noTTB_tax = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_tax.jld2"));
IRFs_HANK_noTTB_tax_Yneg, _, idx_HANK_noTTB_tax_Yneg = load_irfs(joinpath(bld_ar1, "IRFs_HANK_noTTB_tax_Yneg.jld2"));

IRFs_CM_noTTB,         order_cm, idx_CM_noTTB         = load_irfs(joinpath(bld_cm, "IRFs_CM_noTTB.jld2"));
IRFs_CM_TTB4,          _,        idx_CM_TTB4          = load_irfs(joinpath(bld_cm, "IRFs_CM_TTB4.jld2"));
IRFs_CM_TTB12,         _,        idx_CM_TTB12         = load_irfs(joinpath(bld_cm, "IRFs_CM_TTB12.jld2"));
IRFs_CM_noTTB_eta5,    _,        idx_CM_noTTB_eta5    = load_irfs(joinpath(bld_cm, "IRFs_CM_noTTB_eta5.jld2"));
IRFs_CM_noTTB_eta20,   _,        idx_CM_noTTB_eta20   = load_irfs(joinpath(bld_cm, "IRFs_CM_noTTB_eta20.jld2"));
IRFs_CM_noTTB_HANKpars,_,        idx_CM_noTTB_HANKpars= load_irfs(joinpath(bld_cm, "IRFs_CM_noTTB_HANKpars.jld2"));

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

IRFs_HANK_noTTB_noStab_plot = apply_scales(IRFs_HANK_noTTB_noStab, order_hank, scales);
IRFs_HANK_noTTB_noPiStab_plot = apply_scales(IRFs_HANK_noTTB_noPiStab, order_hank, scales);

IRFs_HANK_noTTB_tr03_plot = apply_scales(IRFs_HANK_noTTB_tr03, order_hank, scales);
IRFs_HANK_noTTB_tr2_plot  = apply_scales(IRFs_HANK_noTTB_tr2, order_hank, scales);

IRFs_HANK_noTTB_tax_plot = apply_scales(IRFs_HANK_noTTB_tax, order_hank, scales);
IRFs_HANK_noTTB_tax_Yneg_plot = apply_scales(IRFs_HANK_noTTB_tax_Yneg, order_hank, scales);

# Align complete-markets IRFs to the AR1 row / shock-column structure, then scale
IRFs_HANK_noTTB_tr03_aligned = align_irfs(IRFs_HANK_noTTB_tr03, idx_HANK_noTTB_tr03, order_hank, IRFs_HANK_noTTB, idx_HANK_noTTB, order_hank);
IRFs_HANK_noTTB_tr03_aligned_plot = apply_scales(IRFs_HANK_noTTB_tr03_aligned, order_hank, scales);

IRFs_HANK_noTTB_tr2_aligned = align_irfs(IRFs_HANK_noTTB_tr2, idx_HANK_noTTB_tr2, order_hank, IRFs_HANK_noTTB, idx_HANK_noTTB, order_hank);
IRFs_HANK_noTTB_tr2_aligned_plot = apply_scales(IRFs_HANK_noTTB_tr2_aligned, order_hank, scales);

IRFs_CM_noTTB_aligned      = align_irfs(IRFs_CM_noTTB, idx_CM_noTTB, order_cm, IRFs_HANK_noTTB, idx_HANK_noTTB, order_hank);
IRFs_CM_noTTB_aligned_plot = apply_scales(IRFs_CM_noTTB_aligned, order_hank, scales);

IRFs_CM_noTTB_HANKpars_aligned = align_irfs(IRFs_CM_noTTB_HANKpars, idx_CM_noTTB_HANKpars, order_cm, IRFs_HANK_noTTB, idx_HANK_noTTB, order_hank);
IRFs_CM_noTTB_HANKpars_aligned_plot = apply_scales(IRFs_CM_noTTB_HANKpars_aligned, order_hank, scales);

# Pre-scale quarterly-rate rows ×4 so that factor=100 in plot_irfs gives
# annualised pp for π, RB, LPXA, RR while quantity rows remain as % change.
# This lets all variables appear in one combined figure with a single call.
rate_vars = [:π, :RB, :LPXA];

IRFs_HANK_noTTB_paper                = prescale_rows(IRFs_HANK_noTTB_plot,                   ids_hank, rate_vars);
IRFs_HANK_TTB4_paper                 = prescale_rows(IRFs_HANK_TTB4_plot,                    ids_hank, rate_vars);
IRFs_HANK_TTB12_paper                = prescale_rows(IRFs_HANK_TTB12_plot,                   ids_hank, rate_vars);

IRFs_HANK_noTTB_eta5_paper           = prescale_rows(IRFs_HANK_noTTB_eta5_plot,              ids_hank, rate_vars);
IRFs_HANK_noTTB_eta20_paper          = prescale_rows(IRFs_HANK_noTTB_eta20_plot,             ids_hank, rate_vars);

IRFs_HANK_noTTB_noStab_paper         = prescale_rows(IRFs_HANK_noTTB_noStab_plot,           ids_hank, rate_vars);
IRFs_HANK_noTTB_noPiStab_paper       = prescale_rows(IRFs_HANK_noTTB_noPiStab_plot,         ids_hank, rate_vars);

IRFs_HANK_noTTB_tr03_paper           = prescale_rows(IRFs_HANK_noTTB_tr03_aligned_plot,     ids_hank, rate_vars);
IRFs_HANK_noTTB_tr2_paper            = prescale_rows(IRFs_HANK_noTTB_tr2_aligned_plot,      ids_hank, rate_vars);

IRFs_HANK_noTTB_tax_paper           = prescale_rows(IRFs_HANK_noTTB_tax_plot,              ids_hank, rate_vars);
IRFs_HANK_noTTB_tax_Yneg_paper     = prescale_rows(IRFs_HANK_noTTB_tax_Yneg_plot,        ids_hank, rate_vars);

IRFs_CM_noTTB_aligned_paper          = prescale_rows(IRFs_CM_noTTB_aligned_plot,             ids_hank, rate_vars);
IRFs_CM_noTTB_HANKpars_aligned_paper = prescale_rows(IRFs_CM_noTTB_HANKpars_aligned_plot,   ids_hank, rate_vars);

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
    (:wH,   "Real Wage (Households)"),
];

vars_agg_CM = [
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
]

vars_dist = [
    #(:sdlogy,      "SD log Income"),
    (:GiniI,       "Gini of Income"),
    (:TOP10Ishare, "Top 10% Income Share"),
    (:BOT50Ishare, "Bot 50% Income Share"),
    (:GiniC,       "Gini of Consumption"),
    (:GiniW,       "Gini of Wealth"),
    (:TOP10Wshare, "Top 10% Wealth Share"),
    (:BOT50Wshare, "Bot 50% Wealth Share"),
    (:FrBorr,       "Fraction of Borrowers"),
];

vars_inc = [
    (:wF, "Real Wage (Firms)"),
    (:wH, "Real Wage (Households)"),
    (:Π_F, "Profits (Firms)"),
    (:Π_U, "Profits (Unions)"),
    (:Π_E, "Profits (Entrepreneurs)"),
    (:q, "Capital Price"),
    (:qΠ, "Profit Price"),
    (:RL, "Liquid returns"),
    (:RK, "Capital returns"),
    (:RRL, "Real return on liquid assets"),
    (:RRD, "Real return on debt"),
    (:mc, "Price markup"),
    (:mcw, "Wage markup"),
    (:LPXA, "Ex ante liq. premium"),
]

style_12 = (lw = 2, color = [:blue, :red, :orange], linestyle = [:solid, :dash, :dot]);
style_3 = (lw = 2, color = [:red, :blue, :orange], linestyle = [:dash, :solid, :dot]);

## =============================================================================
# Comparison 1: HANK AR1 (TTB, debt-financed) vs. Complete Markets
# ==============================================================================

@printf "\n[1/6] HANK (TTB, debt) vs. Complete Markets\n"
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


## =============================================================================
# Comparison 2: Time-to-Build vs. No Time-to-Build (AR1, debt-financed)
# ==============================================================================

@printf "[2/6] TTB vs. No-TTB (AR1, debt-financed)\n"
path2 = joinpath(bld_comp, "TTB_vs_noTTB")

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_HANK_noTTB_paper, "No TTB"),
        (IRFs_HANK_TTB4_paper, "1-year lag"),
        (IRFs_HANK_TTB12_paper, "3-year lag")],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
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

plot_irfs(
    shocks_to_plot, vars_inc,
    [(IRFs_HANK_noTTB_plot, "No TTB"),
        (IRFs_HANK_TTB4_plot, "1-year lag"),
        (IRFs_HANK_TTB12_plot, "3-year lag")],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path2, "dist"), yscale = "standard", style_options = style_12,
    suffix = "_TTBvsNoTTB_inc",
)

## ================== 2B: RANK with different TTB =========================
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
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path2, "agg"), yscale = "standard", style_options = style_12,
    suffix = "_TTBvsNoTTB_CM", 
)

## =============================================================================
# Comparison 3: Different elasticities
# ==============================================================================

@printf "[3/6] Different output elasticities\n"
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
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path3, "RANK_agg"), yscale = "standard", style_options = style_3,
    suffix = "_elast_CM"
)

# ==============================================================================
# Comparison 4: No fiscal stabilization (γ_π = γ_Y = 0) vs. baseline fiscal rule (AR1, TTB)
# ==============================================================================

@printf "[4/6] No fiscal stabilization \n"
path4 = joinpath(bld_comp, "noStab")

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_HANK_noTTB_paper, "Baseline"),
    (IRFs_HANK_noTTB_noPiStab_paper, "γ_π = 0"), 
    #(IRFs_HANK_noTTB_noStab_paper, "γ_π = γ_Y = 0")
    ],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path4, "agg"), yscale = "standard", style_options = style_12,
    suffix = "_noStab_agg",
)

plot_irfs(
    shocks_to_plot, vars_dist,
    [(IRFs_HANK_noTTB_paper, "Baseline"), 
    (IRFs_HANK_noTTB_noPiStab_paper, "γ_π = 0"),
    #(IRFs_HANK_noTTB_noStab_paper, "γ_π = γ_Y = 0")
    ],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path4, "dist"), yscale = "standard", style_options = style_12,
    suffix = "_noStab_dist",
)

# ==============================================================================
# Comparison 5: Reduction in Transfer level
# ==============================================================================

@printf "[5/6] Reduction in transfer level \n"
path5 = joinpath(bld_comp, "transfers")

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_HANK_noTTB_paper, "Baseline"), 
    (IRFs_HANK_noTTB_tr03_paper, "Transfer level: 30%"),
    (IRFs_HANK_noTTB_tr2_paper, "Transfer withdrawal: 50%")
     ],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path5, "agg"), yscale = "standard", style_options = style_12,
    suffix = "_transfers_agg",
)

plot_irfs(
    shocks_to_plot, vars_dist,
    [(IRFs_HANK_noTTB_paper, "Baseline"), 
    (IRFs_HANK_noTTB_tr03_paper, "Transfer level: 30%"),
    (IRFs_HANK_noTTB_tr2_paper, "Transfer withdrawal: 50%")
     ],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path5, "dist"), yscale = "standard", style_options = style_12,
    suffix = "_transfers_dist",
)

# ==============================================================================
# Comparison 6: Dynamic Tax Function
# ==============================================================================

@printf "[6/6] Dynamic tax function \n"
path6 = joinpath(bld_comp, "taxes")

plot_irfs(
    shocks_to_plot, vars_agg,
    [(IRFs_HANK_noTTB_paper, "Baseline"), 
     (IRFs_HANK_noTTB_tax_paper, "γ_τB = 3.0, γ_τY = 1.5"),
     (IRFs_HANK_noTTB_tax_Yneg_paper, "γ_τB = 3.0, γ_τY = -0.9")
     ],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path6, "agg"), yscale = "standard", style_options = style_12,
    suffix = "_taxes_agg",
)

plot_irfs(
    shocks_to_plot, vars_dist,
    [(IRFs_HANK_noTTB_plot, "Baseline"), 
     (IRFs_HANK_noTTB_tax_plot, "γ_τB = 3.0, γ_τY = 1.5"),
     (IRFs_HANK_noTTB_tax_Yneg_plot, "γ_τB = 3.0, γ_τY = -0.9")
     ],
    order_hank, ids_hank;
    horizon, save_fig_indiv = false, show_fig = true, save_fig = true,
    path = joinpath(path6, "dist"), yscale = "standard", style_options = style_12,
    suffix = "_taxes_dist",
)


# ==============================================================================
# Multiplier tables
# GI shock: cumulative PV multipliers for Y, C, GiniW, GiniI across model variants.
#
# Use _plot arrays (raw log-deviations, shock-scaled ×10) — NOT _paper arrays,
# which have additional row prescaling that would distort the multiplier ratios.
# ids_hank has the extended RR row; ss_HANK_noTTB must be non-nothing (requires
# regenerating JLD2 files with the updated main_noestim.jl).
# ==============================================================================

@printf "\nComputing multipliers...\n"

mult_vars = [:Y, :C, :GiniI, :GiniW];
mult_horizons = [1, 40, 80];

function make_mult_table(variants; vars = mult_vars, horizons = mult_horizons)
    dfs = DataFrame[]
    for (irfs, label) in variants
        df = compute_pv_multipliers(
            irfs, order_hank, ids_hank, ss_hank, :GI;
            response_vars = vars,
            horizons      = horizons,
            model_name    = label,
            scale_by_ss   = true,
        )
        push!(dfs, df)
    end
    return vcat(dfs...)
end

# Comparison: TTB vs noTTB
mult_TTB = make_mult_table([
    (IRFs_HANK_noTTB_plot,  "No TTB"),
    (IRFs_HANK_TTB4_plot,   "TTB 1yr"),
    (IRFs_HANK_TTB12_plot,  "TTB 3yr"),
]);

# Comparison: elasticities
mult_eta = make_mult_table([
    (IRFs_HANK_noTTB_eta5_plot,  "η = 0.05"),
    (IRFs_HANK_noTTB_plot,       "η = 0.10"),
    (IRFs_HANK_noTTB_eta20_plot, "η = 0.20"),
]);

# Comparison: no fiscal stabilization
mult_noStab = make_mult_table([
    (IRFs_HANK_noTTB_plot, "Baseline"),
    (IRFs_HANK_noTTB_noPiStab_plot, "γ_π = 0"),
    (IRFs_HANK_noTTB_noStab_plot, "γ_π = γ_Y = 0"),
]);

# Comparison: Reduction in transfers
mult_trans = make_mult_table([
    (IRFs_HANK_noTTB_plot, "Baseline"),
    (IRFs_HANK_noTTB_tr03_paper, "Transfer level: 30%"),
    (IRFs_HANK_noTTB_tr2_paper, "Transfer withdrawal: 50%")
]);

# Comparison: Dynamic tax function
mult_tax = make_mult_table([
    (IRFs_HANK_noTTB_plot, "Baseline"),
    (IRFs_HANK_noTTB_tax_paper, "γ_τB = 3.0, γ_τY = 1.5"),
    (IRFs_HANK_noTTB_tax_Yneg_paper, "γ_τB = 3.0, γ_τY = -0.9")
]);

using PrettyTables
@printf "\n--- GI multipliers: TTB vs No-TTB ---\n"
pretty_table(mult_TTB; header = names(mult_TTB), title = "PV Multipliers (GI shock)",
    formatters = ft_printf("%.4f", 2:length(mult_vars)+1))

@printf "\n--- GI multipliers: output elasticity ---\n"
pretty_table(mult_eta; header = names(mult_eta), title = "PV Multipliers (GI shock)",
    formatters = ft_printf("%.4f", 2:length(mult_vars)+1))

@printf "\n--- GI multipliers: No fiscal stabilization ---\n"
pretty_table(mult_noStab; header = names(mult_noStab), title = "PV Multipliers (GI shock)",
    formatters = ft_printf("%.4f", 2:length(mult_vars)+1))
    
@printf "\n--- GI multipliers RANK: Transfer changes ---\n"
 pretty_table(mult_trans; header = names(mult_trans), title = "PV Multipliers (GI shock)",
     formatters = ft_printf("%.4f", 2:length(mult_vars)+1))

@printf "\n--- GI multipliers: Dynamic tax function ---\n"
pretty_table(mult_tax; header = names(mult_tax), title = "PV Multipliers (GI shock)",
    formatters = ft_printf("%.4f", 2:length(mult_vars)+1))

# Save to CSV
mkpath(bld_comp)
CSV.write(joinpath(bld_comp, "multipliers_TTB.csv"), mult_TTB);
CSV.write(joinpath(bld_comp, "multipliers_eta.csv"), mult_eta);
CSV.write(joinpath(bld_comp, "multipliers_noStab.csv"), mult_noStab);
CSV.write(joinpath(bld_comp, "multipliers_transfers.csv"), mult_trans);
# ==============================================================================
# Shock comparison: GI vs. Gshock for the same model
# plot_irfs_cat places both shocks as lines on the same subplot — one panel per
# variable. Use _paper arrays so the ×10 shock scaling and rate prescaling are
# already applied.
# ==============================================================================

@printf "\nShock comparison (GI vs. G)...\n"
fiscal_shocks  = Dict(("Fiscal", "fis") => [:GI, :Gshock])
fiscal_labels  = Dict(:GI => "Gov. Investment", :Gshock => "Gov. Consumption")
style_shocks   = (lw = 2, color = [:blue, :red], linestyle = [:solid, :dash])

vars_agg_shocks = [
    (:Y, "Output"),
    (:C, "Consumption"),
    (:I, "Investment"),
    (:N, "Employment"),
    (:π, "Inflation"),
    (:RB, "Nominal Rate"),
    (:T, "Tax Revenue"),
    #(:wH, "Real Wage (Households)"),
]

vars_dist_shocks = [
    (:GiniI, "Gini of Income"),
    (:TOP10Ishare, "Top 10% Income Share"),
    (:BOT50Ishare, "Bot 50% Income Share"),
    (:GiniC, "Gini of Consumption"),
    (:GiniW, "Gini of Wealth"),
    (:TOP10Wshare, "Top 10% Wealth Share"),
    (:BOT50Wshare, "Bot 50% Wealth Share"),
]

path_shocks = joinpath(bld_comp, "GI_vs_G")

plot_irfs_cat(
    fiscal_shocks, vars_agg_shocks,
    IRFs_HANK_noTTB_paper, order_hank, ids_hank;
    horizon, shock_labels = fiscal_labels,
    show_fig = true, save_fig = true,
    path = joinpath(path_shocks, "noTTB_agg"),
    yscale = "standard", style_options = style_shocks,
)

plot_irfs_cat(
    fiscal_shocks, vars_dist_shocks,
    IRFs_HANK_noTTB_paper, order_hank, ids_hank;
    horizon, shock_labels = fiscal_labels,
    show_fig = true, save_fig = true,
    path = joinpath(path_shocks, "noTTB_dist"),
    yscale = "standard", style_options = style_shocks,
)

@printf "\nDone. All comparison plots saved to:\n  %s\n" bld_comp