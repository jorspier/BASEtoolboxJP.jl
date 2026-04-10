# This file defines the sets of aggregate shocks, states (inluding shocks), and controls
# The code checks that the number of equations in the aggregate model is equal to the number
# of aggregate variables excluding the distributional summary statistics. The latter are not
# contained in the aggregate model code as they are parameter free but change whenever the
# distribution changes and do not show up in any aggregate model equation.

n_rep = 1 # Number of n_rep of some model equations (e.g. countries, industries)

# List of aggregate shocks, without duplication (e.g. across countries or industries)
shock_names = [:ZI, :μ, :μw, :A, :Rshock, :Gshock, :Tprogshock, :Sshock, :TFP, :Auth] 

# List of aggregate states, without duplication of names (e.g. across countries or industries)
# Duplicated names are created below
state_names = [
    "A",
    #"Z",
    "ZI",
    "RB",
    "μ",
    "μw",
    "σ",
    "Ylag",
    "Bgovlag",
    "Tlag",
    "Ilag",
    "wFlag",
    "qlag",
    "Clag",
    "Tbarlag",
    "Tproglag",
    "qΠlag",
    "Gshock",
    "Tprogshock",
    "Rshock",
    "Sshock",
    "TFP",
    #"GI",
    #"Sp",
    "KG",
    "Auth",
    "Auth_lag1",
    "Auth_lag2",
    "Auth_lag3",
    "Auth_lag4",
    "Auth_lag5",
    "Auth_lag6",
    "Auth_lag7",
    "Auth_lag8",
    "Auth_lag9",
    "Auth_lag10",
    "Auth_lag11",
    "Auth_lag12",
    "Auth_lag13",
    "Auth_lag14",
    "Auth_lag15",
    "Auth_lag16",
    "Auth_lag17",
    "Auth_lag18",
    "Auth_lag19",
    "Auth_lag20",
    "Auth_lag21",
    "Auth_lag22",
    "Auth_lag23",
    "Auth_lag24",
    "Auth_lag25",
    "Auth_lag26",
    "Auth_lag27",
    "Auth_lag28",
    "Auth_lag29",
    "Auth_lag30",
    "Auth_lag31",
    "Auth_lag32",
    "Auth_lag33",
    "Auth_lag34",
    "Auth_lag35",
    "Auth_lag36",
    "Auth_lag37",
    "Auth_lag38",
    "Auth_lag39",
    "GI_lag1",
    "GI_lag2",
    "GI_lag3",
    "GI_lag4",
    #"GI_lag5",
    #"GI_lag6",
    #"GI_lag7",
    #"GI_lag8",
    #"GI_lag9",
    #"GI_lag10",
    #"GI_lag11",
    #"GI_lag12"
]

# List of (the subset) of aggregate states, that need to be duplicated (e.g. across countries or industries)
dup_state_names = Vector{String}(undef, 0)
#[
#     "A",
#]

# List cross-sectional controls / distributional summary variables (no equations in aggregate model expected)
# if these need to be duplicated, do by hand !
distr_names = ["GiniC", "GiniW", "TOP10Ishare", "TOP10Inetshare", "TOP10Wshare", "sdlogy"]

control_names = [
    "RK_before_taxes",
    "RK",
    "wF",
    "K",
    "π",
    "πw",
    "Y",
    "C",
    "q",
    "N",
    "mc",
    "mcw",
    "u",
    "qΠ",
    "Π_F",
    "RL",
    "RD",
    "RRL",
    "RRD",
    "Bgov",
    "Hprog",
    "Htilde",
    "Tbar",
    "T",
    "I",
    "B",
    "BD",
    "BY",
    "TY",
    "wH",
    "G",
    "Tlev",
    "Tprog",
    "Tc",
    "Tk",
    "Ttr_1",
    "Ttr_2",
    "TR",
    "Ygrowth",
    "Bgovgrowth",
    "Igrowth",
    "wgrowth",
    "Cgrowth",
    "Tgrowth",
    "GIgrowth",
    "LP",
    "LPXA",
    "Π_U",
    "Π_E",
    "TotalAssets",
    "τprog",
    "Z",
    "GI"
]

# List of (the subset) of aggregate states, that need to be duplicated (e.g. across countries or industries)
dup_control_names = Vector{String}(undef, 0)#
#["",
#     "LP",
#]

# Delete duplicated state and shock names and create duplicated variable names of state and control names

state_names = setdiff(state_names, dup_state_names)
for j = 1:n_rep # Create names of duplicated state variables (e.g. across countries or industries)
    if j == 1
        aux = ""
    else
        aux = string(j)
    end
    append!(state_names, dup_state_names .* aux)
end

shock_states = string.(shock_names)
dup_shock_states = intersect(string.(shock_names), dup_state_names)
shock_states = setdiff(shock_states, dup_shock_states)
for j = 1:n_rep # Create names of duplicated shock variables (e.g. across countries or industries)
    if j == 1
        aux = ""
    else
        aux = string(j)
    end
    append!(shock_states, dup_shock_states .* aux)
end
shock_names = Symbol.(shock_states)

control_names = setdiff(control_names, dup_control_names)
for j = 1:n_rep # Create names of duplicated control variables (e.g. across countries or industries)
    if j == 1
        aux = ""
    else
        aux = string(j)
    end
    append!(control_names, dup_control_names .* aux)
end

# All controls in one array
control_names = [distr_names; control_names]
# All names in one array
aggr_names = [state_names; control_names]

args_hh_prob_names = [
    "wH",
    "N",
    "Hprog",
    "q",
    "RRL",
    "RRD",
    "RK",
    "Tlev",
    "Tprog",
    "Tbar",
    "Tc",
    "Tk",
    "Ttr_1",
    "Ttr_2",
    "Π_E",
    "Π_U",
    "Htilde",
    "σ",
]

# ascii names used for cases where unicode doesn't work, e.g., file saves
unicode2ascii(x) =
    replace.(
        replace.(
            replace.(replace.(replace.(x, "τ" => "tau"), "σ" => "sigma"), "π" => "pi"),
            "μ" => "mu",
        ),
        "ρ" => "rho",
    )

state_names_ascii = unicode2ascii(state_names)
control_names_ascii = unicode2ascii(control_names)
aggr_names_ascii = [state_names_ascii; control_names_ascii]
