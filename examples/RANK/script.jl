#------------------------------------------------------------------------------
# Header: load module
#------------------------------------------------------------------------------
# make sure that your pwd is set to the folder containing script and BASEforHANK
# otherwise adjust the load path

cd("examples/RANK")

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
println("Capital to Output Ratio: ", K/Y)
println("Government Debt to Output Ratio: ", Bgov/Y)
println("Government Consumption to Output Ratio: ", G/Y)

# linearize the full model
lr_full = linearize_full_model(sr_full, m_par)
# jldsave("7_Saves/linearresults.jld2", true; lr_full)
# @load "7_Saves/linearresults.jld2" lr_full

# No sparse state-space representation required for RANK
sr_reduc = sr_full 
lr_reduc = lr_full 

# initialize everything needed for MCMC estimation
er_start, posterior_start, smoother_start, sr_start, lr_start, m_par_start =
    estimation_prep(sr_reduc, lr_reduc, m_par)

# conduct MCMC estimation
sr_mc, lr_mc, er_mc, m_par_mc, draws_raw, posterior, accept_rate,
par_final, hessian_sym, smoother_output = montecarlo(sr_start, lr_start, er_start, m_par_start)

# Only relevant output for later plotting will be saved
smoother_output = (0.0,0.0,smoother_output[3],0.0,0.0, smoother_output[6],0.0)

# Stores mcmc results in file e_set.save_posterior_file 
jldsave(BASEforHANK.e_set.save_posterior_file, true;
        sr_mc, lr_mc, er_mc, m_par_mc, draws_raw, posterior, accept_rate,
        par_final, hessian_sym, smoother_output, e_set)

exit()