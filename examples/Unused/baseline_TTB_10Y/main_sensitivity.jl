"""
main_sensitivity.jl: Sensitivity analysis for government investment parameters.
Computes and overlays IRFs for different parameter values in the same graphs.
"""
global_start_time = time()

using PrettyTables, Printf, Plots, Setfield;

## ------------------------------------------------------------------------------------------
## Header: set up paths, pre-process user inputs, load module
## ------------------------------------------------------------------------------------------

root_dir = replace(Base.current_project(), "Project.toml" => "")
cd(root_dir)

paths = Dict(
    "root" => root_dir,
    "src" => joinpath(root_dir, "src"),
    "bld" => joinpath(root_dir, "bld"),
    "src_example" => @__DIR__,
    "bld_example" => replace(@__DIR__, "examples" => "bld") * "_sensitivity",
)

mkpath(paths["bld_example"])

include(paths["src"] * "/Preprocessor/PreprocessInputs.jl")
include(paths["src"] * "/BASEforHANK.jl")
using .BASEforHANK

BASEforHANK.LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

## ------------------------------------------------------------------------------------------
## Initialize: set up baseline model parameters
## ------------------------------------------------------------------------------------------

m_par = ModelParameters()

@set! m_par.ξ = 4.0
@set! m_par.γ = 2.0
@set! m_par.β = 0.9828470527212994
@set! m_par.λ = 0.06038247534019993
@set! m_par.ρ_h = 0.98
@set! m_par.σ_h = 0.12
@set! m_par.ι = 0.0625
@set! m_par.ζ = 0.000258606126756716
@set! m_par.α = 0.318
@set! m_par.δ_0 = 0.021500000000000002
@set! m_par.δ_s = 0.7055720197078786
@set! m_par.ϕ = 1.9409223183717077
@set! m_par.μ = 1.1
@set! m_par.κ = 0.1456082664986374
@set! m_par.μw = 1.1
@set! m_par.κw = 0.23931075416274708
@set! m_par.Tlev = 1.2596633554411527
@set! m_par.Tprog = 1.0 + 0.1022
@set! m_par.Tc = 1.0
@set! m_par.Tk = 1.0
@set! m_par.Ttr_1 = 1.0
@set! m_par.Ttr_2 = 1.0
@set! m_par.RRB = 1.0
@set! m_par.Rbar = 0.024276573587297953
@set! m_par.ωΠ = 0.2
@set! m_par.ιΠ = 0.016
@set! m_par.shiftΠ = 0.7002848330469671
@set! m_par.ρ_A = 0.9724112284399131
@set! m_par.σ_A = 0.0015812471705012755
@set! m_par.ρ_ZI = 0.7637111671257767
@set! m_par.σ_ZI = 0.0721141538701523
@set! m_par.ρ_μ = 0.903740078830077
@set! m_par.σ_μ = 0.01350860622318172
@set! m_par.ρ_μw = 0.9057892147641305
@set! m_par.σ_μw = 0.035058308969408175
@set! m_par.ρ_s = 0.544722245741144
@set! m_par.σ_Sshock = 0.6918558038597916
@set! m_par.Σ_n = 28.879770107327673
@set! m_par.ρ_R = 0.8030565250630299
@set! m_par.σ_Rshock = 0.002306627917745612
@set! m_par.θ_π = 2.0780841671981856
@set! m_par.θ_Y = 0.21872568927661648
@set! m_par.γ_B = 0.020131162775595176
@set! m_par.γ_π = -2.1737350397931947
@set! m_par.γ_Y = -0.4363130165391906
@set! m_par.ρ_Gshock = 0.9682224473297878
@set! m_par.σ_Gshock = 0.003761816459554433
@set! m_par.ρ_τ = 0.4926482696848203
@set! m_par.γ_Bτ = 3.293063617271948
@set! m_par.γ_Yτ = -0.9207283604196101
@set! m_par.ρ_P = 0.9194235885358465
@set! m_par.σ_Tprogshock = 0.06865440038519788
@set! m_par.γ_BP = 0.0
@set! m_par.γ_YP = 0.0
@set! m_par.γ_WP = 0.0
@set! m_par.ρ_Rshock = 1.0e-8
@set! m_par.ρ_Tprogshock = 1.0e-8
@set! m_par.ρ_Sshock = 1.0e-8
@set! m_par.ρ_TFP = 0.9978155269262137     
@set! m_par.σ_TFP = 0.00600947811158941 

# baseline govt investment parameters
@set! m_par.γ_GI = 1.0
@set! m_par.GI_share = 0.03
@set! m_par.ϕ_GI = 1/8
@set! m_par.δ_KG = 0.04
@set! m_par.η_KG = 0.10
@set! m_par.ρ_GI = 0.95
@set! m_par.σ_GI = m_par.σ_Gshock * (0.135 / m_par.GI_share)

## ------------------------------------------------------------------------------------------
## Define Sensitivity Scenarios
## ------------------------------------------------------------------------------------------

scenarios = Dict(
    "Baseline"                  => m_par,
    "Fast Pipeline (ϕ=1/4)"     => Setfield.@set(m_par.ϕ_GI = 1/4),
    "Slow Pipeline (ϕ=1/36)"    => Setfield.@set(m_par.ϕ_GI = 1/12),
    #"High Depr. (δ=0.1)"       => Setfield.@set(m_par.δ_KG = 0.1),
    #"High Elast. (η=0.10)"     => Setfield.@set(m_par.η_KG = 0.15),
    #"Low Elast. (η=0.05)"      => Setfield.@set(m_par.η_KG = 0.05),
    #"Low Persist. (ρ=0.80)"    => Setfield.@set(m_par.ρ_GI = 0.80),
    #"Low Shock (σ/2)"          => Setfield.@set(m_par.σ_GI = m_par.σ_GI * 0.5)
)

# Colors for plotting mapping exactly to scenarios
colors = [ :black, :blue, :red, :green, :purple, :orange]
linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :solid]

## ------------------------------------------------------------------------------------------
## Loop Over Scenarios and Compute IRFs
## ------------------------------------------------------------------------------------------

results_irf = Dict()
global irf_order_ref = nothing
global var_indexes_ref = nothing

for (name, par) in scenarios
    @printf "\nComputing scenario: %s\n" name
    
    # 1. Steady State & Linearization
    ss_tmp = call_find_steadystate(par)
    sr_tmp = call_prepare_linearization(ss_tmp, par)

    # compute steady state moments
    K = exp.(sr_tmp.XSS[sr_tmp.indexes.KSS]);
    B = exp.(sr_tmp.XSS[sr_tmp.indexes.BSS]);
    Bgov = exp.(sr_tmp.XSS[sr_tmp.indexes.BgovSS]);
    Y = exp.(sr_tmp.XSS[sr_tmp.indexes.YSS]);
    T10W = exp(sr_tmp.XSS[sr_tmp.indexes.TOP10WshareSS]);
    G = exp.(sr_tmp.XSS[sr_tmp.indexes.GSS]);
    fr_borr = BASEforHANK.eval_cdf(sr_tmp.distrSS, :b, sr_tmp.n_par, 0.0);
    # new 
    GI = exp.(sr_tmp.XSS[sr_tmp.indexes.GISS]);
    KG = exp.(sr_tmp.XSS[sr_tmp.indexes.KGSS]);
    Sp = exp.(sr_tmp.XSS[sr_tmp.indexes.SpSS]);

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


    lr_tmp = linearize_full_model(sr_tmp, par)
    
    # Extract structural variables needed for correct index mapping
    if name == "Baseline"
        global var_indexes_ref = sr_tmp.indexes
    end

    # 2. Exogenous variables and standard deviations
    exovars_tmp = [getfield(sr_tmp.indexes, shock_names[i]) for i = 1:length(shock_names)]
    stds_tmp = [getfield(sr_tmp.m_par, Symbol("σ_", i)) for i in shock_names]
    
    # 3. Compute IRFs (distribution = false to save time for aggregates)
    transform_elements = transformation_elements(sr_tmp, sr_tmp.n_par.model, sr_tmp.n_par.distribution_states)

    IRFs_tmp, _, IRFs_order_tmp = compute_irfs( # no distribution
        exovars_tmp,
        lr_tmp.State2Control,
        lr_tmp.LOMstate,
        sr_tmp.XSS,
        sr_tmp.indexes;
        init_val = stds_tmp,
        distribution = false,
        comp_ids = sr_tmp.compressionIndexes,
        transform_elements = transform_elements,
        n_par = sr_tmp.n_par,
        m_par = sr_tmp.m_par,
    )
    
    results_irf[name] = IRFs_tmp
    
    if irf_order_ref === nothing
        global irf_order_ref = IRFs_order_tmp
    end
end

## ------------------------------------------------------------------------------------------
## Overlay Plotting
## ------------------------------------------------------------------------------------------

@printf "\nGenerating overlaid plots...\n"

horizon = 80
shock_name = :GI
shock_idx = findfirst(x -> x == shock_name, irf_order_ref)

vars_to_plot = [
    (:Ygrowth, "Output growth", 100, "%"),
    (:Cgrowth, "Consumption growth", 100, "%"),
    (:RB, "Nominal rate", 10000, "bp"),
    (:Bgov, "Gov. Debt", 100, "%"),
    (:KG, "Public Capital", 100, "%"),
    (:GiniW, "Wealth Gini", 10000, "bp"),
    (:GiniC, "Consumption Gini", 10000, "bp")
]

mkpath(paths["bld_example"] * "/SensitivityPlots")

for (var_sym, title, scale, unit) in vars_to_plot
    # Retrieve the correct row index for this variable from the model structure
    var_idx = getfield(var_indexes_ref, var_sym)
    
    # Initialize the plot with the Baseline scenario
    p = plot(
        1:horizon, 
        results_irf["Baseline"][var_idx, 1:horizon, shock_idx] .* scale,
        label = "Baseline",
        linewidth = 2,
        color = colors[1],
        linestyle = linestyles[1],
        xlabel = "Periods",
        ylabel = "Deviation ($unit)",
        title = "$title ($shock_name)",
        legend = :topright
    )
    
    # Loop over the remaining scenarios and overlay them
    i = 2
    for (name, _) in scenarios
        if name != "Baseline"
            plot!(
                p,
                1:horizon,
                results_irf[name][var_idx, 1:horizon, shock_idx] .* scale,
                label = name,
                linewidth = 2,
                color = colors[i],
                linestyle = linestyles[i]
            )
            i += 1
        end
    end
    
    # Save the plot
    file_name = paths["bld_example"] * "/SensitivityPlots/Sens_$(var_sym)_$(shock_name).pdf"
    savefig(p, file_name)
    display(p)
end

@printf "\nDone. Sensitivity plots saved in %s\n" paths["bld_example"]*"/SensitivityPlots"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")