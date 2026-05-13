# This file defines the sets of aggregate shocks, states (inluding shocks), and controls
# The code checks that the number of equations in the aggregate model is equal to the number 
# of aggregate variables excluding the distributional summary statistics. The latter are not 
# contained in the aggregate model code as they are parameter free but change whenever the 
# distribution changes and do not show up in any aggregate model equation.

shock_names = [
    :TFP, :ZI, :μ, :μw, :A, :Rshock, :Gshock, :Auth
]

state_names = [
    "A", 
    "TFP", 
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
    "wlag", 
    "qlag", 
    "Clag", 
    "av_tax_ratelag", 
    "τproglag",
    "qΠlag",
    "Gshock", 
    "Tprogshock", 
    "Rshock", 
    "Sshock",
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
]

# List cross-sectional controls / distributional summary variables (no equations in aggregate model expected)
distr_names   = []

control_names = [
    "r", "w", "K", "π" ,"πw", "Y" ,"C", "q",  "N", "mc", "mcw", "u","qΠ","firm_profits","RL","Bgov",
    "Ht", "av_tax_rate", "T", "I", "B","BD", "BY","TY", "mcww", "G", "τlev", "τprog",
     "Ygrowth", "Bgovgrowth", "Igrowth", "wgrowth", "Cgrowth", "Tgrowth", "LP", "LPXA", "unionprofits", "profits",
     "Z", "GI", "GIgrowth"
]

# All controls in one array
control_names       = [distr_names; control_names]
# All names in one array
aggr_names          = [state_names; control_names]

args_hh_prob_names = []

# ascii names used for cases where unicode doesn't work, e.g., file saves
unicode2ascii(x)    = replace.(replace.(replace.(replace.(replace.(x,"τ"=>"tau"), "σ" => "sigma"),"π"=>"pi"),"μ"=>"mu"),"ρ"=>"rho")

state_names_ascii   = unicode2ascii(state_names)
control_names_ascii = unicode2ascii(control_names)
aggr_names_ascii    = [state_names_ascii; control_names_ascii]