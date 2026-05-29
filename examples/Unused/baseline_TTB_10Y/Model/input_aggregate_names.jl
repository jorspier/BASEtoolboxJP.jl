# This file defines the sets of aggregate shocks, states (inluding shocks), and controls
# The code checks that the number of equations in the aggregate model is equal to the number
# of aggregate variables excluding the distributional summary statistics. The latter are not
# contained in the aggregate model code as they are parameter free but change whenever the
# distribution changes and do not show up in any aggregate model equation.

n_rep = 1 # Number of n_rep of some model equations (e.g. countries, industries)

# List of aggregate shocks, without duplication (e.g. across countries or industries)
shock_names = [:TFP, :ZI, :μ, :μw, :A, :Rshock, :Gshock, :GI_NS, :Tprogshock, :Sshock]

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
    "KG",
    "GI_NS",
    "GI_lag1",
    "GI_lag2",
    "GI_lag3",
    "GI_lag4",
    "GI_news_1",
    "GI_news_2",
    "GI_news_3",
    "GI_news_4",
    "GI_news_5",
    "GI_news_6",
    "GI_news_7",
    "GI_news_8",
    "GI_news_9",
    "GI_news_10",
    "GI_news_11",
    "GI_news_12",
    "GI_news_13",
    "GI_news_14",
    "GI_news_15",
    "GI_news_16",
    "GI_news_17",
    "GI_news_18",
    "GI_news_19",
    "GI_news_20",
    "GI_news_21",
    "GI_news_22",
    "GI_news_23",
    "GI_news_24",
    "GI_news_25",
    "GI_news_26",
    "GI_news_27",
    "GI_news_28",
    "GI_news_29",
    "GI_news_30",
    "GI_news_31",
    "GI_news_32",
    "GI_news_33",
    "GI_news_34",
    "GI_news_35",
    "GI_news_36",
    "GI_news_37",
    "GI_news_38",
    "GI_news_39",
    "GI_news_40",
]

# List of (the subset) of aggregate states, that need to be duplicated (e.g. across countries or industries)
dup_state_names = Vector{String}(undef, 0)
#[
#     "A",
#]

# List cross-sectional controls / distributional summary variables (no equations in aggregate model expected)
# if these need to be duplicated, do by hand !
distr_names = ["GiniC", "GiniI", "GiniInet", "GiniW", "TOP10Ishare", "TOP10Inetshare", "TOP10Wshare", "sdlogy"]

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
