#------------------------------------------------------------------------------
# THIS FILE CONTAINS THE "AGGREGATE" MODEL EQUATIONS, I.E. EVERYTHING  BUT THE 
# HOUSEHOLD PLANNING PROBLEM. THE lATTER IS DESCRIBED BY ONE EGM BACKWARD STEP AND 
# ONE FORWARD ITERATION OF THE DISTRIBUTION.
#
# AGGREGATE EQUATIONS TAKE THE FORM 
# F[EQUATION NUMBER] = lhs - rhs
#
# EQUATION NUMBERS ARE GENEREATED AUTOMATICALLY AND STORED IN THE INDEX STRUCT
# FOR THIS THE "CORRESPONDING" VARIABLE NEEDS TO BE IN THE LIST OF STATES 
# OR CONTROLS.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# AUXILIARY VARIABLES ARE DEFINED FIRST
#------------------------------------------------------------------------------
    # ιΠ =  (1.0 ./ 40.0 - 1.0 ./ 800.0) .* m_par.shiftΠ .+ 1.0 ./ 800.0 
    # ωΠ = ιΠ ./ m_par.ιΠ .* m_par.ωΠ


    # Elasticities and steepness from target markups for Phillips Curves
    η                       = μ / (μ - 1.0)                                 # demand elasticity
    κ                       = η * (m_par.κ / m_par.μ) * (m_par.μ - 1.0)     # implied steepness of phillips curve
    ηw                      = μw / (μw - 1.0)                               # demand elasticity wages
    κw                      = ηw * (m_par.κw / m_par.μw) * (m_par.μw - 1.0) # implied steepness of wage phillips curve

    # Capital Utilization
    MPK_SS                  = exp(XSS[indexes.rSS]) - 1.0 + m_par.δ_0       # stationary equil. marginal productivity of capital
    δ_1                     = MPK_SS                                        # normailzation of utilization to 1 in stationary equilibrium
    δ_2                     = δ_1 .* m_par.δ_s                              # express second utilization coefficient in relative terms
    # Auxiliary variables
    Kserv                   = K * u                                         # Effective capital
    MPKserv                 = mc .* Z .* m_par.α .* (Kserv ./ N) .^(m_par.α - 1.0)      # marginal product of Capital
    depr                    = m_par.δ_0 + δ_1 * (u - 1.0) + δ_2 / 2.0 * (u - 1.0)^2.0   # depreciation

    Wagesum                 = N * w                                         # Total wages in economy t
    WagesumPrime            = NPrime * wPrime                               # Total wages in economy t+1

    YREACTION = Ygrowth                                  # Policy reaction function to Y

    # distr_y = sum(distrSS, dims=(1, 2))

    # tax progressivity variabels used to calculate e.g. total taxes
    tax_prog_scale = (m_par.γ + m_par.τ_prog) / ((m_par.γ + τprog))                        # scaling of labor disutility including tax progressivity
    incgross = profits .+ mcw .* w .* N  # capital liquidation Income (q=1 in steady state)
    # incgross[end] = (n_par.grid_y[end] .* profits)                         # gross profit income
    inc = τlev .* ((profits .^ (1.0 .- τprog)) .+ ((mcw .* w .* N) .^ (1.0 .- τprog)))                                 # capital liquidation Income (q=1 in steady state)
    #inc[:, :, end] .= τlev .* (n_par.mesh_y[:, :, end] .* profits) .^ (1.0 - τprog)             # profit income net of taxes
    taxrev = incgross .- inc                                                 # tax revenues
    
    
    ############################################################################
    #           Error term calculations (i.e. model starts here)          #
    ############################################################################

    #-------- States -----------#
    # Error Term on exogeneous States
    # Shock processes
    F[indexes.Gshock]       = log.(GshockPrime)         - m_par.ρ_Gshock * log.(Gshock)     # primary deficit shock
    F[indexes.Tprogshock]   = log.(TprogshockPrime)     - m_par.ρ_Pshock * log.(Tprogshock) # tax shock

    F[indexes.Rshock]       = log.(RshockPrime)         - m_par.ρ_Rshock * log.(Rshock)     # Taylor rule shock
    F[indexes.Sshock]       = log.(SshockPrime)         - m_par.ρ_Sshock * log.(Sshock)     # uncertainty shock

    # Stochastic states that can be directly moved (no feedback)
    F[indexes.A]            = log.(APrime)              - m_par.ρ_A * log.(A)               # (unobserved) Private bond return fed-funds spread (produces goods out of nothing if negative)
    F[indexes.TFP]          = log.(TFPPrime)            - m_par.ρ_TFP * log.(TFP)           # TFP
    F[indexes.ZI]           = log.(ZIPrime)             - m_par.ρ_ZI * log.(ZI)             # Investment-good productivity

    F[indexes.μ]            = log.(μPrime./m_par.μ)     - m_par.ρ_μ * log.(μ./m_par.μ)      # Process for markup target
    F[indexes.μw]           = log.(μwPrime./m_par.μw)   - m_par.ρ_μw * log.(μw./m_par.μw)   # Process for w-markup target

    # Endogeneous States (including Lags)
    F[indexes.σ]            = log.(σPrime)              - (m_par.ρ_s * log.(σ) + (1.0 - m_par.ρ_s) *
                                m_par.Σ_n * log(Ygrowth) + log(Sshock))                     # Idiosyncratic income risk (contemporaneous reaction to business cycle)

    F[indexes.Ylag]         = log(YlagPrime)    - log(Y)
    F[indexes.Bgovlag]      = log(BgovlagPrime)    - log(Bgov)
    F[indexes.Ilag]         = log(IlagPrime)    - log(I)
    F[indexes.wlag]         = log(wlagPrime)    - log(w)
    F[indexes.Tlag]         = log(TlagPrime)    - log(T)
    F[indexes.qlag]         = log(qlagPrime)    - log(q)
    F[indexes.Clag]         = log(ClagPrime)    - log(C)
    F[indexes.av_tax_ratelag] = log(av_tax_ratelagPrime) - log(av_tax_rate)
    F[indexes.τproglag]     = log(τproglagPrime) - log(τprog)
    F[indexes.qΠlag]         = log(qΠlagPrime)    - log(qΠ)

    # Growth rates
    F[indexes.Ygrowth]      = log(Ygrowth)      - log(Y/Ylag)
    F[indexes.Tgrowth]      = log(Tgrowth)      - log(T/Tlag)
    F[indexes.Bgovgrowth]   = log(Bgovgrowth)      - log(Bgov/Bgovlag)
    F[indexes.Igrowth]      = log(Igrowth)      - log(I/Ilag)
    F[indexes.wgrowth]      = log(wgrowth)      - log(w/wlag)
    F[indexes.Cgrowth]      = log(Cgrowth)      - log(C/Clag)
    F[indexes.GIgrowth]     = log(GIgrowth)     - log((GI/GI_lag1))

    #  Taylor rule and interest rates
    F[indexes.RB]           = log(RBPrime) - XSS[indexes.RBSS] -
                            ((1 - m_par.ρ_R) * m_par.θ_π) .* log(π) -
                            ((1 - m_par.ρ_R) * m_par.θ_Y) .* log(YREACTION) -
                            m_par.ρ_R * (log.(RB) - XSS[indexes.RBSS])  - log(Rshock)

    # Tax rule
    F[indexes.τprog]        = log(τprog) - m_par.ρ_P * log(τproglag)  - 
                            (1.0 - m_par.ρ_P) *(XSS[indexes.τprogSS]) - 
                            (1.0 - m_par.ρ_P) * m_par.γ_YP * log(YREACTION) -
                            (1.0 - m_par.ρ_P) * m_par.γ_BP * (log(Bgovgrowth)) - 
                            log(Tprogshock)

    TaxAux = dot(1.0, taxrev)
    IncAux = dot(1.0, incgross)
    F[indexes.τlev] = av_tax_rate - TaxAux ./ IncAux  # Union profits are taxed at average tax rate
    F[indexes.T] = log(T) - log(TaxAux + av_tax_rate * unionprofits)
                            

    F[indexes.av_tax_rate]  = log(av_tax_rate) - m_par.ρ_τ * log(av_tax_ratelag)  - 
                                (1.0 - m_par.ρ_τ) * XSS[indexes.av_tax_rateSS] -
                                (1.0 - m_par.ρ_τ) * m_par.γ_Yτ * log(YREACTION) -
                                (1.0 - m_par.ρ_τ) * m_par.γ_Bτ * (log(Bgovgrowth))
    
    # Government Investment Rule
    # Spending the authorized investment linearly over 40 periods
    F[indexes.Auth] = (log(AuthPrime)) - 
        (m_par.ρ_GI * log(Auth) + (1.0 - m_par.ρ_GI) * XSS[indexes.AuthSS])
    F[indexes.Auth_lag1] = (log(Auth_lag1Prime)) - (log(Auth))
    F[indexes.Auth_lag2] = (log(Auth_lag2Prime)) - (log(Auth_lag1))
    F[indexes.Auth_lag3] = (log(Auth_lag3Prime)) - (log(Auth_lag2))
    F[indexes.Auth_lag4] = (log(Auth_lag4Prime)) - (log(Auth_lag3))
    F[indexes.Auth_lag5] = (log(Auth_lag5Prime)) - (log(Auth_lag4))
    F[indexes.Auth_lag6] = (log(Auth_lag6Prime)) - (log(Auth_lag5))
    F[indexes.Auth_lag7] = (log(Auth_lag7Prime)) - (log(Auth_lag6))
    F[indexes.Auth_lag8] = (log(Auth_lag8Prime)) - (log(Auth_lag7))
    F[indexes.Auth_lag9] = (log(Auth_lag9Prime)) - (log(Auth_lag8))
    F[indexes.Auth_lag10] = (log(Auth_lag10Prime)) - (log(Auth_lag9))
    F[indexes.Auth_lag11] = (log(Auth_lag11Prime)) - (log(Auth_lag10))
    F[indexes.Auth_lag12] = (log(Auth_lag12Prime)) - (log(Auth_lag11))
    F[indexes.Auth_lag13] = (log(Auth_lag13Prime)) - (log(Auth_lag12))
    F[indexes.Auth_lag14] = (log(Auth_lag14Prime)) - (log(Auth_lag13))
    F[indexes.Auth_lag15] = (log(Auth_lag15Prime)) - (log(Auth_lag14))
    F[indexes.Auth_lag16] = (log(Auth_lag16Prime)) - (log(Auth_lag15))
    F[indexes.Auth_lag17] = (log(Auth_lag17Prime)) - (log(Auth_lag16))
    F[indexes.Auth_lag18] = (log(Auth_lag18Prime)) - (log(Auth_lag17))
    F[indexes.Auth_lag19] = (log(Auth_lag19Prime)) - (log(Auth_lag18))
    F[indexes.Auth_lag20] = (log(Auth_lag20Prime)) - (log(Auth_lag19))
    F[indexes.Auth_lag21] = (log(Auth_lag21Prime)) - (log(Auth_lag20))
    F[indexes.Auth_lag22] = (log(Auth_lag22Prime)) - (log(Auth_lag21))
    F[indexes.Auth_lag23] = (log(Auth_lag23Prime)) - (log(Auth_lag22))
    F[indexes.Auth_lag24] = (log(Auth_lag24Prime)) - (log(Auth_lag23))
    F[indexes.Auth_lag25] = (log(Auth_lag25Prime)) - (log(Auth_lag24))
    F[indexes.Auth_lag26] = (log(Auth_lag26Prime)) - (log(Auth_lag25))
    F[indexes.Auth_lag27] = (log(Auth_lag27Prime)) - (log(Auth_lag26))
    F[indexes.Auth_lag28] = (log(Auth_lag28Prime)) - (log(Auth_lag27))
    F[indexes.Auth_lag29] = (log(Auth_lag29Prime)) - (log(Auth_lag28))
    F[indexes.Auth_lag30] = (log(Auth_lag30Prime)) - (log(Auth_lag29))
    F[indexes.Auth_lag31] = (log(Auth_lag31Prime)) - (log(Auth_lag30))
    F[indexes.Auth_lag32] = (log(Auth_lag32Prime)) - (log(Auth_lag31))
    F[indexes.Auth_lag33] = (log(Auth_lag33Prime)) - (log(Auth_lag32))
    F[indexes.Auth_lag34] = (log(Auth_lag34Prime)) - (log(Auth_lag33))
    F[indexes.Auth_lag35] = (log(Auth_lag35Prime)) - (log(Auth_lag34))
    F[indexes.Auth_lag36] = (log(Auth_lag36Prime)) - (log(Auth_lag35))
    F[indexes.Auth_lag37] = (log(Auth_lag37Prime)) - (log(Auth_lag36))
    F[indexes.Auth_lag38] = (log(Auth_lag38Prime)) - (log(Auth_lag37))
    F[indexes.Auth_lag39] = (log(Auth_lag39Prime)) - (log(Auth_lag38))

    # Construction pipeline of public capital
    F[indexes.GI_lag1] = (log(GI_lag1Prime)) - (log(GI))
    F[indexes.GI_lag2] = (log(GI_lag2Prime)) - (log(GI_lag1))
    F[indexes.GI_lag3] = (log(GI_lag3Prime)) - (log(GI_lag2))
    F[indexes.GI_lag4] = (log(GI_lag4Prime)) - (log(GI_lag3))

    # Linear Spending Equation
    F[indexes.GI] = (log(GI)) - (log((1/40) * 
        (Auth + Auth_lag1 + Auth_lag2 + Auth_lag3 + Auth_lag4 + 
        Auth_lag5 + Auth_lag6 + Auth_lag7 + Auth_lag8 + Auth_lag9 + 
        Auth_lag10 + Auth_lag11 + Auth_lag12 + Auth_lag13 + Auth_lag14 + 
        Auth_lag15 + Auth_lag16 + Auth_lag17 + Auth_lag18 + Auth_lag19 + 
        Auth_lag20 + Auth_lag21 + Auth_lag22 + Auth_lag23 + Auth_lag24 + 
        Auth_lag25 + Auth_lag26 + Auth_lag27 + Auth_lag28 + Auth_lag29 + 
        Auth_lag30 + Auth_lag31 + Auth_lag32 + Auth_lag33 + Auth_lag34 + 
        Auth_lag35 + Auth_lag36 + Auth_lag37 + Auth_lag38 + Auth_lag39)))


    # Finished public capital
    F[indexes.KG] = (log(KGPrime)) - (log((1.0 - m_par.δ_KG) * KG + GI_lag4))

    # Effective TFP
    F[indexes.Z] = (log(Z)) - 
        (log(TFP) + 
        m_par.η_KG * (log(KG) - XSS[indexes.KGSS])
    )

    # --------- Controls ------------
    # Deficit rule
    F[indexes.π]            = log(BgovgrowthPrime) + m_par.γ_B * (log(Bgov)- XSS[indexes.BgovSS])  -
                                m_par.γ_Y * log(YREACTION)  - m_par.γ_π * log(π) - log(Gshock) - 
                                (m_par.γ_GI * exp(XSS[indexes.GISS] - XSS[indexes.BgovSS])) * (log(GI) -XSS[indexes.GISS])

    F[indexes.G]            = log(G) - log(BgovPrime + T - RB / π * Bgov - GI)             # Government Budget Constraint

    # Phillips Curve to determine equilibrium markup, output, factor incomes 
    F[indexes.mc]           = (log.(π)- XSS[indexes.πSS]) - κ *(mc - 1 ./ μ ) -
                                m_par.β * ((log.(πPrime) - XSS[indexes.πSS]) .* YPrime ./ Y) 
                            
    # Wage Phillips Curve 
    F[indexes.mcw]          = (log.(πw)- XSS[indexes.πwSS]) - (κw * (mcw - 1 ./ μw) +
                                m_par.β * ((log.(πwPrime) - XSS[indexes.πwSS]) .* WagesumPrime ./ Wagesum))
    # worker's wage = mcw * firm's wage

    # Wage Dynamics
    F[indexes.πw]           = log.(w ./ wlag) - log.(πw ./ π)                   # Definition of real wage inflation

    # Capital utilisation
    F[indexes.u]            = MPKserv  -  q * (δ_1 + δ_2 * (u - 1.0))           # Optimality condition for utilization

    # Prices
    F[indexes.r]            = log.(r) - log.(1 + MPKserv * u - q * depr )       # rate of return on capital

    F[indexes.mcww]         = log.(mcww) - log.(mcw * w)                        # wages that workers receive

    F[indexes.w]            = log.(w) - log.(wage(Kserv, Z * mc, N, m_par))     # wages that firms pay

    F[indexes.unionprofits] = log.(unionprofits)  - log.(w.*N .* (1.0 - mcw))  # profits of the monopolistic unions

    # F[indexes.profits]      = log.(profits)  - log.(Y .* (1.0 - mc) .+ q .* (KPrime .- (1.0 .- depr) .* K) .- I)  # profits of the monopolistic resellers

    F[indexes.firm_profits] = log.(firm_profits) - log.(Y .* (1.0 - mc) .+ q .* (KPrime .- (1.0 .- depr) .* K) .- I)    # profits: + price setting profits + investment profits missing, but =0 on the margin
    F[indexes.profits]      = log.(profits)  - log.((1.0 .- m_par.ωΠ) .* firm_profits .+ m_par.ιΠ .* (qΠ .- 1.0)) # distributed profits to entrepreneurs
    F[indexes.qΠ]           = log.(RBPrime ./ πPrime) .- log.(((qΠPrime .- 1.0).* (1 - m_par.ιΠ) .+ m_par.ωΠ .* firm_profitsPrime) ./ (qΠ .- 1.0) )

    F[indexes.RL]           = log.(RL) - log.((RB .* Bgov .+ π .* ((qΠ .- 1.0) .* (1 - m_par.ιΠ) .+ m_par.ωΠ .* firm_profits)) ./ B)

    F[indexes.Bgov]         = log.(B) - log.(Bgov + (qΠlag .- 1.0))                                 # total liquidity demand


    F[indexes.q]            = 1.0 - ZI * q * (1.0 - m_par.ϕ / 2.0 * (Igrowth - 1.0)^2.0 - # price of capital investment adjustment costs
                            m_par.ϕ * (Igrowth - 1.0) * Igrowth)  -
                            m_par.β * ZIPrime * qPrime * m_par.ϕ * (IgrowthPrime - 1.0) * (IgrowthPrime)^2.0
    
    # Asset market premia
    F[indexes.LP]           = log.(LP)                  - (log((q + r - 1.0)/qlag) - log(RB / π))                   # Ex-post liquidity premium           
    F[indexes.LPXA]         = log.(LPXA)                - (log((qPrime + rPrime - 1.0)/q) - log(RBPrime / πPrime))  # ex-ante liquidity premium

    # Aggregate Quantities
    F[indexes.I]            = KPrime .-  K .* (1.0 .- depr)  .- ZI .* I .* (1.0 .- m_par.ϕ ./ 2.0 .* (Igrowth -1.0).^2.0)           # Capital accumulation equation
    F[indexes.N]            = log.(N) - log.(((1.0 - τprog) * τlev * (mcw .* w).^(1.0 - τprog)).^(1.0 / (m_par.γ + τprog)) .* Ht)   # labor supply
    F[indexes.Y]            = log.(Y) - log.(Z .* N .^(1.0 .- m_par.α) .* Kserv .^m_par.α)                                          # production function
    F[indexes.C]            = log.(Y .- G .- GI .- I .- BD*m_par.Rbar .+ (A .- 1.0) .* RB .* B ./ π) .- log(C)                            # Resource constraint

    # Error Term on prices/aggregate summary vars (logarithmic, controls), here difference to SS value averages
    F[indexes.BY]           = log.(BY)    - log.(B/Y)                                                               # Bond to Output ratio
    F[indexes.TY]           = log.(TY)    - log.(T/Y)                                                               # Tax to output ratio
    
    # Distribution summary statistics used in this file (using the steady state distrubtion in case). 
    # Lines here generate a unit derivative (distributional summaries do not change with other aggregate vars).
    F[indexes.K]            = log.(K)     - XSS[indexes.KSS]                                                        # Capital market clearing
    F[indexes.BD]           = log.(BD)    - XSS[indexes.BDSS]                                                       # IOUs            
    F[indexes.B]            = log.(B)     - XSS[indexes.BSS]                                                        # Bond market clearing
    
    # Add distributional summary stats that do change with other aggregate controls/prices and with estimated parameters
    Htact                   = 1.0 #dot(distr_y[1:end-1],(n_par.grid_y[1:end-1]/n_par.H).^((m_par.γ + m_par.τ_prog)/(m_par.γ + τprog)))
    F[indexes.Ht]           = log.(Ht)    - log.(Htact)

    # other dsitributional statistics not used in other aggregate equations and not changing with parameters, 
    # but potentially with other aggregate variables are NOT included here. They are found in FSYS.

