"""
Main script for estimation of RANK model that has time to build constraint 
and 10-year public investment process. Equivalent to HANK baseline.
"""
global_start_time = time()

using PrettyTables, Printf, BenchmarkTools, LinearAlgebra;

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
    "bld_example" => replace(@__DIR__, "examples" => "bld") * "_estim",
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
## Initialize: set up model parameters, priors, and estimation settings
## ------------------------------------------------------------------------------------------

# model parameters and priors
m_par = ModelParameters();
priors = collect(metaflatten(m_par, prior));
par_prior = mode.(priors);
m_par = BASEforHANK.Flatten.reconstruct(m_par, par_prior);
e_set = BASEforHANK.e_set;

# set some paths
@set! e_set.save_mode_file = paths["bld_example"] * "/RANK_mode.jld2";
@set! e_set.save_posterior_file = paths["bld_example"] * "/RANK_chain.jld2";
@set! e_set.mode_start_file = paths["src_example"] * "/Data/par_final_dict.txt"; 
@set! e_set.data_file = paths["src_example"] * "/Data/GER_growth.csv";

# fix seed for random number generation
BASEforHANK.Random.seed!(e_set.seed);

## ------------------------------------------------------------------------------------------
## Calculate Steady State and prepare linearization
## ------------------------------------------------------------------------------------------

# steady state at the prior mode (Using the unified wrapper)
ss_full = call_find_steadystate(m_par);

# sparse DCT representation (For RANK, this is trivial but structurally equivalent)
sr_full = call_prepare_linearization(ss_full, m_par);

# save the steady state
jldsave(paths["bld_example"] * "/steadystate.jld2", true; sr_full);

# compute steady state moments
K       = exp.(sr_full.XSS[sr_full.indexes_r.KSS])
Bgov    = exp.(sr_full.XSS[sr_full.indexes_r.BgovSS])
Y       = exp.(sr_full.XSS[sr_full.indexes_r.YSS])
G       = exp.(sr_full.XSS[sr_full.indexes_r.GSS])

# Display steady state moments
@printf "\n"
pretty_table(
    [
        "Capital to Output Ratio" K / Y/4.0
        "Government Debt to Output Ratio" Bgov / Y/4.0
        "Government Spending to Output Ratio" G/Y 
    ];
    header = ["Variable", "Value"],
    title = "RANK Steady State Moments",
    formatters = ft_printf("%.4f"),
)

## ------------------------------------------------------------------------------------------
## Linearize the full model, find state-space representation 
## ------------------------------------------------------------------------------------------

lr_full = linearize_full_model(sr_full, m_par);

# save the linearization
jldsave(paths["bld_example"] * "/linearresults.jld2", true; lr_full);

# No sparse state-space representation required for RANK, but we assign to standard variable names
sr_reduc = sr_full;
lr_reduc = lr_full;

# save the reduction
jldsave(paths["bld_example"] * "/reduction.jld2", true; sr_reduc, lr_reduc);

## ------------------------------------------------------------------------------------------
## Estimation
## ------------------------------------------------------------------------------------------

if e_set.estimate_model == true
    @printf "\n"
    @printf "Estimation...\n"

    # Use unified find_mode instead of the deprecated estimation_prep
    er_mode, posterior_mode, smoother_mode, sr_mode, lr_mode, m_par_mode =
        find_mode(sr_reduc, lr_reduc, m_par, e_set)

    # Adjust starting values for MCMC sampling to proportional 1% steps
    rank_start_vals = er_mode.par_final
    step_sizes = (abs.(rank_start_vals) .* 0.01) .+ 1e-4
    variances = step_sizes .^ 2
    hessian_diag = 1.0 ./ variances 
    @set! er_mode.hessian_final = Matrix(Diagonal(hessian_diag))
        
    smoother_mode = (0.0, 0.0, smoother_mode[3], 0.0, 0.0, smoother_mode[6], 0.0)

    jldsave(
        e_set.save_mode_file,
        true;
        posterior_mode,
        smoother_mode,
        sr_mode,
        lr_mode,
        er_mode,
        m_par_mode,
        e_set,
    )

    # Use unified sample_posterior instead of montecarlo
    sr_mc,
    lr_mc,
    er_mc,
    m_par_mc,
    draws_raw,
    posterior,
    accept_rate,
    par_final,
    hessian_sym,
    smoother_output = sample_posterior(sr_mode, lr_mode, er_mode, m_par_mode, e_set)

    smoother_output = (0.0, 0.0, smoother_output[3], 0.0, 0.0, smoother_output[6], 0.0)

    jldsave(
        e_set.save_posterior_file,
        true;
        sr_mc,
        lr_mc,
        er_mc,
        m_par_mc,
        draws_raw,
        posterior,
        accept_rate,
        par_final,
        hessian_sym,
        smoother_output,
        e_set,
    )

    @printf "Estimation... Done. \n"
end

## ------------------------------------------------------------------------------------------
## Post Estimation: Compute & Export IRFs
## ------------------------------------------------------------------------------------------

@printf "\n"
@printf "Compute IRFs...\n"

# Determine which parameter set to use for IRFs depending on whether estimation ran
sr_eval = e_set.estimate_model ? sr_mc : sr_reduc
lr_eval = e_set.estimate_model ? lr_mc : lr_reduc
m_par_eval = e_set.estimate_model ? m_par_mc : m_par

shock_names_rank = e_set.shock_names 

# Get indices of the shocks (using indexes_r for the RANK struct)
exovars_rank = [getfield(sr_eval.indexes_r, shock) for shock in shock_names_rank]

# Get standard deviations of the shocks to set identical impulse magnitudes
stds_rank = [getfield(m_par_eval, Symbol("σ_", shock)) for shock in shock_names_rank]

# Compute IRFs using the unified HANK function
IRFs_rank, IRFs_lvl_rank, SHOCKs_rank = compute_irfs(
    exovars_rank,
    lr_eval.State2Control,
    lr_eval.LOMstate,
    sr_eval.XSS,
    sr_eval.indexes_r;       
    init_val = stds_rank,
    distribution = false   
)

# Create the index dictionary for plotting later
idx_dict_rank = Dict{Symbol, Int}(
    name => getfield(sr_eval.indexes_r, name) 
    for name in fieldnames(typeof(sr_eval.indexes_r)) 
    if getfield(sr_eval.indexes_r, name) isa Int
)

all_rank_vars = collect(keys(idx_dict_rank))

# Save to output folder
jldsave(paths["bld_example"] * "/IRFs_RANK_export.jld2", true; 
    IRFs = IRFs_rank, 
    IRFs_order = SHOCKs_rank, 
    idx_dict_rank,
    all_rank_vars
)

@printf "\n"
@printf "Done.\n"
println("Total Runtime: ", round((time() - global_start_time) / 60; digits=2), " minutes")