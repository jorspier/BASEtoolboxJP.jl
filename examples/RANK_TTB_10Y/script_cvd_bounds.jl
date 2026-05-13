#------------------------------------------------------------------------------
# Header: load module
#------------------------------------------------------------------------------
# make sure that your pwd is set to the folder containing script and BASEforHANK
# otherwise adjust the load path
cd("examples/RANK")

# pre-process user inputs for model setup
include("3_NumericalBasics/PreprocessInputs.jl")
using BenchmarkTools, LinearAlgebra

include("BASEforHANK.jl")
using .BASEforHANK

# set BLAS threads to the number of Julia threads.
# prevents BLAS from grabbing all threads on a machine
BLAS.set_num_threads(Threads.nthreads())
# Fix seed for random number generation
e_set = BASEforHANK.e_set;
BASEforHANK.Random.seed!(e_set.seed)
#------------------------------------------------------------------------------
# Load Mode and Posterior 
#------------------------------------------------------------------------------

@load "7_Saves/RANK_chain.jld2" sr_mc lr_mc er_mc m_par_mc draws_raw posterior accept_rate par_final hessian_sym smoother_output e_set


##############################################################################################
# Graphical Model Output, functions not integrated in package
###############################################################################################
using Plots, VegaLite, DataFrames, FileIO, StatsPlots, CategoricalArrays, Flatten, Statistics, PrettyTables, Colors


### Credible intervals for VD ### # Doesn't work with multiple models yet!
models = ((sr_mc, lr_mc, BASEforHANK.e_set, m_par_mc, draws_raw),) # needs to be adjusted if more than one model
model_names = ["RANK"]
shocks = [
    :Z, :ZI, :μ, :μw, :A, :Rshock, :Gshock, :Auth
]
variables = [:Ygrowth, :Cgrowth, :Igrowth, :N, :wgrowth, :RB, :π,]

VDcredint = compute_vardecomp_bounds(models, variables, model_names; n_replic=1000)
jldsave(string("7_Saves/VDcred.jld2"), true; VDcredint)

nice_var_names = ["output growth", "consumption growth", "investment growth", "employment",
    "wage growth", "nominal rate", "inflation", "uncertainty", "tax progressivity",
    "T10 wealth share", "T10 income share"]
nice_s_names = ["tfp", "inv.-spec. tech.", "price markup", "wage markup", "risk premium",
    "mon. policy", "structural deficit", "tax progr.", "income risk"]

open(string("../../tempTables/Estimates/vcd_cred_table_RANK.tex"), "w") do f

    write(f, "Variable ")
    for i in nice_s_names
        write(f, "\t & $i")
    end
    write(f, "\t \\\\ \n")
    var_count = 0
    for k in variables
        var_count += 1
        write(f, "\t ", nice_var_names[var_count], " &")
        for i in shocks
            k1 = round.(VDcredint[(VDcredint[:, :shock].==i).&(VDcredint[:, :variable].==k), :point_estimate], digits=1)[1]
            write(f, "\t $k1 &")
        end
        write(f, "\t \\\\ \n")
        for i in shocks
            k2 = round.(VDcredint[(VDcredint[:, :shock].==i).&(VDcredint[:, :variable].==k), :lower_bound], digits=1)[1]
            k3 = round.(VDcredint[(VDcredint[:, :shock].==i).&(VDcredint[:, :variable].==k), :upper_bound], digits=1)[1]
            write(f, "\t & (" * rpad(k2, 3, '0') * ", " * rpad(k3, 3, '0') * ")")
        end
        write(f, "\t \\\\ \n")
    end
end

exit()