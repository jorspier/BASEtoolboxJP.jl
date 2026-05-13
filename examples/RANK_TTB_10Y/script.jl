"""
Main script for estimation of RANK model that has time to build constraint and 10-year public investment process.
    Equivalent to HANK baseline_TTB_10Y
"""
#------------------------------------------------------------------------------
# Header: load module 
#------------------------------------------------------------------------------
# make sure that your pwd is set to the folder containing script and BASEforHANK
# otherwise adjust the load path

cd(dirname(@__FILE__))

# pre-process user inputs for model setup
include("3_NumericalBasics/PreprocessInputs.jl")

include("BASEforHANK.jl")
using .BASEforHANK

# set BLAS threads to the number of Julia threads.
# prevents BLAS from grabbing all threads on a machine
BASEforHANK.LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

#------------------------------------------------------------------------------
# initialize parameters to priors to select coefficients of DCTs of Vm, Vk
# that are retained 
#------------------------------------------------------------------------------
m_par  = ModelParameters()
priors = collect(metaflatten(m_par, prior)) # model parameters
par_prior = mode.(priors)
m_par  = BASEforHANK.Flatten.reconstruct(m_par, par_prior)
e_set  = BASEforHANK.e_set;

# load starting values
@load BASEforHANK.e_set.estimation_start_file start_vector
m_par = BASEforHANK.Flatten.reconstruct(m_par, start_vector[1:length(start_vector)-length(e_set.meas_error_input)])

# Fix seed for random number generation
BASEforHANK.Random.seed!(e_set.seed)

# Calculate Steady State
sr_full = compute_steadystate(m_par)
# jldsave("7_Saves/steadystate.jld2", true; sr_full) # true enables compression
# @load "7_Saves/steadystate.jld2" sr_full

#------------------------------------------------------------------------------
# compute and display steady-state moments
#------------------------------------------------------------------------------
K       = exp.(sr_full.XSS[sr_full.indexes.KSS])
Bgov    = exp.(sr_full.XSS[sr_full.indexes.BgovSS])
Y       = exp.(sr_full.XSS[sr_full.indexes.YSS])
G       = exp.(sr_full.XSS[sr_full.indexes.GSS])

println("Steady State Moments:")
println("Capital to Output Ratio: ", K/Y /4.0)
println("Government Debt to Output Ratio: ", Bgov/Y /4.0)
println("Government Consumption to Output Ratio: ", G/Y)

## linearize the full model
lr_full = linearize_full_model(sr_full, m_par)
# jldsave("7_Saves/linearresults.jld2", true; lr_full)
# @load "7_Saves/linearresults.jld2" lr_full

# No sparse state-space representation required for RANK
sr_reduc = sr_full 
lr_reduc = lr_full 

# initialize everything needed for MCMC estimation
er_start, posterior_start, smoother_start, sr_start, lr_start, m_par_start =
    estimation_prep(sr_reduc, lr_reduc, m_par)

# Test likelihood at starting values if MCMC does not find acceptable params
# test_par = vcat(par_prior, fill(0.01, 7))
# log_like, prior_like, post_like, alarm, _ = BASEforHANK.likeli(test_par, sr_reduc, lr_reduc, er_start, m_par_start, e_set)
# println("--- MCMC DIAGNOSTIC RESULTS ---")
# println("1. Model Stable (BK)   : ", !alarm)
# println("2. Prior Likelihood    : ", prior_like)
# println("3. Data Log-Likelihood : ", log_like)
# println("4. Total Posterior     : ", post_like)

# conduct MCMC estimation
sr_mc, lr_mc, er_mc, m_par_mc, draws_raw, posterior, accept_rate,
par_final, hessian_sym, smoother_output = montecarlo(sr_start, lr_start, er_start, m_par_start)

# Only relevant output for later plotting will be saved
smoother_output = (0.0,0.0,smoother_output[3],0.0,0.0, smoother_output[6],0.0)

# Stores mcmc results in file e_set.save_posterior_file 
jldsave(BASEforHANK.e_set.save_posterior_file, true;
        sr_mc, lr_mc, er_mc, m_par_mc, draws_raw, posterior, accept_rate,
        par_final, hessian_sym, smoother_output, e_set)

# ------
## Post Estimation: Compute IRFs for RANK model using the HANK function
# ----- 
# Extract the shock names
shock_names_rank = e_set.shock_names 

# Get indices of the shocks (using indexes_r for the RANK struct)
exovars_rank = [getfield(sr_mc.indexes_r, shock) for shock in shock_names_rank]

# Get standard deviations of the shocks to set identical impulse magnitudes
stds_rank = [getfield(m_par_mc, Symbol("σ_", shock)) for shock in shock_names_rank]

# Compute IRFs
# distribution = false bypasses all the DCT/Grid unpacking
IRFs_rank, IRFs_lvl_rank, SHOCKs_rank = compute_irfs(
    exovars_rank,
    lr_mc.State2Control,
    lr_mc.LOMstate,
    sr_mc.XSS,
    sr_mc.indexes_r;       
    init_val = stds_rank,
    distribution = false    
)

# Export IRFs to plot with HANK together
# Create the index dictionary for plotting later
idx_dict_rank = Dict{Symbol, Int}(
    name => getfield(sr_mc.indexes_r, name) 
    for name in fieldnames(typeof(sr_mc.indexes_r)) 
    if getfield(sr_mc.indexes_r, name) isa Int
)

# Extract all variables for the export
all_rank_vars = collect(keys(idx_dict_rank))

# Save to output folder
jldsave("7_Saves/IRFs_RANK_export.jld2", true; 
    IRFs = IRFs_rank, 
    IRFs_order = SHOCKs_rank, 
    idx_dict_rank,
    all_rank_vars
)

exit()