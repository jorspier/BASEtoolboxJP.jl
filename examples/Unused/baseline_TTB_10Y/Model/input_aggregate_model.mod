#=

This file contains the aggregate model equations. That is, everything but the household
planning problem which is described by one EGM backward step and one forward iteration of
the distribution.

Model equations take the form F[equation number] = (lhs) - (rhs)

Equation numbers are generated automatically and stored in the index struct. For this the
corresponding variable needs to be in the list of states or controls.

=#

## ----------------------------------------------------------------------------------------
## Auxiliary variables
## ----------------------------------------------------------------------------------------

# Policy reaction function to Y
YREACTION = Ygrowth

## Taxation -------------------------------------------------------------------------------

# Mass of households in each productivity state, distribution is (nb, nk, nh)
distr_h = get_PDF_h(distr)

## Profit shares --------------------------------------------------------------------------

# Shares regarding entrepreneurs selling shares of their profits
ιΠ = (1.0 / 40.0 - 1.0 / 800.0) * m_par.shiftΠ + 1.0 / 800.0
ωΠ = ιΠ / m_par.ιΠ * m_par.ωΠ

# Slopes of the Phillips curve ------------------------------------------------------------

# Demand elasticity
η = μ / (μ - 1.0)

# Implied steepness of phillips curve
κ = η * (m_par.κ / m_par.μ) * (m_par.μ - 1.0)

# Demand elasticity wages
ηw = μw / (μw - 1.0)

# Implied steepness of wage phillips curve
κw = ηw * (m_par.κw / m_par.μw) * (m_par.μw - 1.0)

## Capital Utilization --------------------------------------------------------------------

# Normalize utilization to 1 in stationary equilibrium
δ_1 = exp(XSS[indexes.RK_before_taxesSS]) - 1.0 + m_par.δ_0

# Express second utilization coefficient in relative terms
δ_2 = δ_1 * m_par.δ_s

# Effective capital
Kserv = K * u

# Marginal product of capital
MPKserv = interest(mc, Z, Kserv, N, m_par) + m_par.δ_0

# Depreciation
depr = m_par.δ_0 + δ_1 * (u - 1.0) + δ_2 / 2.0 * (u - 1.0)^2.0

## ----------------------------------------------------------------------------------------
## Aggregate equations
## ----------------------------------------------------------------------------------------

## Lagged variables -----------------------------------------------------------------------

F[indexes.Ylag] = (log(YlagPrime)) - (log(Y))
F[indexes.Bgovlag] = (log(BgovlagPrime)) - (log(Bgov))
F[indexes.Ilag] = (log(IlagPrime)) - (log(I))
F[indexes.wFlag] = (log(wFlagPrime)) - (log(wF))
F[indexes.Tlag] = (log(TlagPrime)) - (log(T))
F[indexes.qlag] = (log(qlagPrime)) - (log(q))
F[indexes.Clag] = (log(ClagPrime)) - (log(C))
F[indexes.Tbarlag] = (log(TbarlagPrime)) - (log(Tbar))
F[indexes.Tproglag] = (log(TproglagPrime)) - (log(Tprog))
F[indexes.qΠlag] = (log(qΠlagPrime)) - (log(qΠ))

## Growth rates ---------------------------------------------------------------------------

F[indexes.Ygrowth] = (log(Ygrowth)) - (log(Y / Ylag))
F[indexes.Tgrowth] = (log(Tgrowth)) - (log(T / Tlag))
F[indexes.Bgovgrowth] = (log(Bgovgrowth)) - (log(Bgov / Bgovlag))
F[indexes.Igrowth] = (log(Igrowth)) - (log(I / Ilag))
F[indexes.wgrowth] = (log(wgrowth)) - (log(wF / wFlag))
F[indexes.Cgrowth] = (log(Cgrowth)) - (log(C / Clag))
F[indexes.GIgrowth] = (log(GIgrowth)) - (log(GI / GI_lag1))

## Fiscal policy --------------------------------------------------------------------------

# Deficit rule, see equation 33 in BBL
F[indexes.π] =
    (log(BgovgrowthPrime)) - (
        -m_par.γ_B * (log(Bgov) - XSS[indexes.BgovSS]) +
        m_par.γ_Y * log(YREACTION) +
        m_par.γ_π * log(π) +
        log(Gshock) +
        (m_par.γ_GI * exp(XSS[indexes.GISS] - XSS[indexes.BgovSS])) * (log(GI) - XSS[indexes.GISS])
    ) 

# Average tax rate, see equation 34 in BBL (here simplified)
F[indexes.Tbar] =
    (log(Tbar .- 1.0)) - (
        m_par.ρ_τ * log(Tbarlag .- 1.0) +
        (1.0 - m_par.ρ_τ) * (log(exp(XSS[indexes.TbarSS]) .- 1.0)) +
        (1.0 - m_par.ρ_τ) * m_par.γ_Yτ * log(YREACTION) +
        (1.0 - m_par.ρ_τ) * m_par.γ_Bτ * (log(Bgov) - log(Bgovlag))
    )
# This variable needs to be set for the package!

# Progressivity of labor tax, see equation 33a in BBL
F[indexes.Tprog] =
    (log(Tprog .- 1.0)) - (
        m_par.ρ_P * log(Tproglag .- 1.0) +
        (1.0 - m_par.ρ_P) * (log(exp(XSS[indexes.TprogSS]) .- 1.0)) +
        (1.0 - m_par.ρ_P) * m_par.γ_YP * log(YREACTION) +
        (1.0 - m_par.ρ_P) * m_par.γ_BP * (log(Bgov) - XSS[indexes.BgovSS]) +
        log(Tprogshock)
    )
# This variable needs to be set for the package!

# Auxiliary variable for the estimation
F[indexes.τprog] = (log(τprog)) - (log(Tprog .- 1.0))

# Level of labor tax, see equation 35 in BBL (typos!), this determines Tlev
F[indexes.Tlev] =
    (Tbar .- 1.0) - (av_labor_tax_rate(
        n_par,
        m_par,
        wH * N / Hprog,
        (Tlev .- 1.0),
        (Tprog .- 1.0),
        Π_E,
        Htilde,
        distr_h,
    ))
# This variable needs to be set for the package!

# Government budget constraint, see below equation 35 in BBL
F[indexes.G] = (log(G)) - (log(BgovPrime + T - RB / π * Bgov - GI)) # investment added

# Total goverment tax revenues, see below equation 35 in BBL
F[indexes.T] =
    (log(T)) - (log(
        (Tbar .- 1.0) * (wH * N + Π_E + Π_U) +
        (Tc .- 1.0) * C +
        (Tk .- 1.0) * (RK_before_taxes .- 1.0) * K - (TR .- 1.0),
    ))

# VAT rate (gross)
F[indexes.Tc] = (log(Tc)) - (XSS[indexes.TcSS])
# This variable needs to be set for the package!

# Capital income tax rate (gross)
F[indexes.Tk] = (log(Tk)) - (XSS[indexes.TkSS])
# This variable needs to be set for the package!

# Flat (lump-sum) tax level (gross)
F[indexes.Ttr_1] = (log(Ttr_1)) - (XSS[indexes.Ttr_1SS])
F[indexes.Ttr_2] = (log(Ttr_2)) - (XSS[indexes.Ttr_2SS])
@write_args_hh_prob()
F[indexes.TR] =
    (log(TR)) - (log(transfer_scheme(n_par, m_par, args_hh_prob; distr_h = distr_h)))
# This variable needs to be set for the package!

# Tax shock
F[indexes.Tprogshock] = (log(TprogshockPrime)) - (m_par.ρ_Tprogshock * log(Tprogshock))

# Primary deficit shock (only consumption)
F[indexes.Gshock] = (log(GshockPrime)) - (m_par.ρ_Gshock * log(Gshock))

## Monetary policy ------------------------------------------------------------------------

# Taylor rule, see equation 32 in BBL
F[indexes.RB] =
    (log(RBPrime)) - (
        XSS[indexes.RBSS] +
        ((1 - m_par.ρ_R) * m_par.θ_π) * log(π) +
        ((1 - m_par.ρ_R) * m_par.θ_Y) * log(YREACTION) +
        m_par.ρ_R * (log(RB) - XSS[indexes.RBSS]) +
        log(Rshock)
    )

# Monetary policy shock
F[indexes.Rshock] = (log(RshockPrime)) - (m_par.ρ_Rshock * log(Rshock))

## Labor market ---------------------------------------------------------------------------

# Idiosyncratic income risk (contemporaneous reaction to business cycle)
F[indexes.σ] =
    (log(σPrime)) -
    ((m_par.ρ_s * log(σ) + (1.0 - m_par.ρ_s) * m_par.Σ_n * log(Ygrowth) + log(Sshock)))
# This variable needs to be set for the package!

# Uncertainty shock
F[indexes.Sshock] = (log(SshockPrime)) - (m_par.ρ_Sshock * log(Sshock))

# Wage Phillips Curve
F[indexes.mcw] =
    (log(πw) - XSS[indexes.πwSS]) - (
        κw * (mcw - 1 / μw) +
        m_par.β * ((log(πwPrime) - XSS[indexes.πwSS]) * (NPrime * wFPrime) / (N * wF))
    )

# Definition of real wage inflation
F[indexes.πw] = (log(wF / wFlag)) - (log(πw / π))

# Process for wF-markup target
F[indexes.μw] = (log(μwPrime / m_par.μw)) - (m_par.ρ_μw * log(μw / m_par.μw))

# Wages that households receive
F[indexes.wH] = (log(wH)) - (log(mcw * wF))
# This variable needs to be set for the package!

# Union profits
F[indexes.Π_U] = (log(Π_U)) - (log(profits_U(wF, wH, N)))
# This variable needs to be set for the package!

# Labor supply
F[indexes.N] =
    (log(N)) - (log(
        labor_supply(
            wH,
            Hprog,
            (Tlev .- 1.0),
            (Tprog .- 1.0),
            (Tc .- 1.0),
            m_par,
            wH * N / Hprog + Π_E,
            m_par.scale_prog,
        ),
    ))
# This variable needs to be set for the package!

# Hours-weighted average labor productivity, normalized, see equation 19b in BBL
F[indexes.Hprog] =
    (log(Hprog)) - (log(
        dot(
            distr_h[1:(end - 1)],
            (n_par.grid_h[1:(end - 1)] / Htilde) .^ scale_Hprog((Tprog .- 1.0), m_par),
        ),
    ))
# This variable needs to be set for the package!

## Production -----------------------------------------------------------------------------

# Price Phillips Curve
F[indexes.mc] =
    (log(π) - XSS[indexes.πSS]) -
    (κ * (mc - 1 / μ) + m_par.β * ((log(πPrime) - XSS[indexes.πSS]) * YPrime / Y))

# Process for markup target
F[indexes.μ] = (log(μPrime / m_par.μ)) - (m_par.ρ_μ * log(μ / m_par.μ))

# Rate of return on capital
F[indexes.RK_before_taxes] = (log(RK_before_taxes)) - (log(1 + MPKserv * u - q * depr))
# This variable needs to be set for the package!

# Rate of return on capital, net of capital taxes
F[indexes.RK] = (log(RK)) - (log((RK_before_taxes .- 1.0) * (1.0 .- (Tk .- 1.0)) .+ 1.0))
# This variable needs to be set for the package!

# Wages that firms pay
F[indexes.wF] = (log(wF)) - (log(wage(mc, Z, Kserv, N, m_par)))

# Firm profits
F[indexes.Π_F] = (log(Π_F)) - (log(Y * (1.0 - mc) + q * (KPrime - (1.0 - depr) * K) - I))

# Distributed profits to entrepreneurs
F[indexes.Π_E] = (log(Π_E)) - (log((1.0 - ωΠ) * Π_F + ιΠ * (qΠ - 1.0)))
# This variable needs to be set for the package!

# Price of capital investment
F[indexes.q] =
    (log(1.0)) - (log(
        ZI *
        q *
        (1.0 - m_par.ϕ / 2.0 * (Igrowth - 1.0)^2.0 - m_par.ϕ * (Igrowth - 1.0) * Igrowth) +
        m_par.β * ZIPrime * qPrime * m_par.ϕ * (IgrowthPrime - 1.0) * (IgrowthPrime)^2.0,
    ))
# This variable needs to be set for the package!

# Capital accumulation equation
F[indexes.I] =
    (log(KPrime)) -
    (log(K * (1.0 - depr) + ZI * I * (1.0 - m_par.ϕ / 2.0 * (Igrowth - 1.0) .^ 2.0)))

# Production function
F[indexes.Y] = (log(Y)) - (log(output(Z, Kserv, N, m_par)))

# Capital utilisation: optimality condition for utilization
F[indexes.u] = (log(MPKserv)) - (log(q * (δ_1 + δ_2 * (u - 1.0))))

# Investment-good productivity
F[indexes.ZI] = (log(ZIPrime)) - (m_par.ρ_ZI * log(ZI))

## Government investment: News shock cascade -------

# Single news shock process, ρ=0, σ calibrated to 12.5bn/Y_SS
F[indexes.GI_NS] = (log(GI_NSPrime)) - (m_par.ρ_GI_NS * log(GI_NS))

# Each news variable is independently driven by the same shock
# and passes its value forward each period
# At t=0: all 40 variables are elevated by GI_NS simultaneously
# At t>0: GI_NS=SS=0 in logs, cascade drains one step per period
F[indexes.GI_news_1]  = log(GI_news_1Prime)  - log(GI_news_2)  - log(GI_NS)
F[indexes.GI_news_2]  = log(GI_news_2Prime)  - log(GI_news_3)  - log(GI_NS)
F[indexes.GI_news_3]  = log(GI_news_3Prime)  - log(GI_news_4)  - log(GI_NS)
F[indexes.GI_news_4]  = log(GI_news_4Prime)  - log(GI_news_5)  - log(GI_NS)
F[indexes.GI_news_5]  = log(GI_news_5Prime)  - log(GI_news_6)  - log(GI_NS)
F[indexes.GI_news_6]  = log(GI_news_6Prime)  - log(GI_news_7)  - log(GI_NS)
F[indexes.GI_news_7]  = log(GI_news_7Prime)  - log(GI_news_8)  - log(GI_NS)
F[indexes.GI_news_8]  = log(GI_news_8Prime)  - log(GI_news_9)  - log(GI_NS)
F[indexes.GI_news_9]  = log(GI_news_9Prime)  - log(GI_news_10) - log(GI_NS)
F[indexes.GI_news_10] = log(GI_news_10Prime) - log(GI_news_11) - log(GI_NS)
F[indexes.GI_news_11] = log(GI_news_11Prime) - log(GI_news_12) - log(GI_NS)
F[indexes.GI_news_12] = log(GI_news_12Prime) - log(GI_news_13) - log(GI_NS)
F[indexes.GI_news_13] = log(GI_news_13Prime) - log(GI_news_14) - log(GI_NS)
F[indexes.GI_news_14] = log(GI_news_14Prime) - log(GI_news_15) - log(GI_NS)
F[indexes.GI_news_15] = log(GI_news_15Prime) - log(GI_news_16) - log(GI_NS)
F[indexes.GI_news_16] = log(GI_news_16Prime) - log(GI_news_17) - log(GI_NS)
F[indexes.GI_news_17] = log(GI_news_17Prime) - log(GI_news_18) - log(GI_NS)
F[indexes.GI_news_18] = log(GI_news_18Prime) - log(GI_news_19) - log(GI_NS)
F[indexes.GI_news_19] = log(GI_news_19Prime) - log(GI_news_20) - log(GI_NS)
F[indexes.GI_news_20] = log(GI_news_20Prime) - log(GI_news_21) - log(GI_NS)
F[indexes.GI_news_21] = log(GI_news_21Prime) - log(GI_news_22) - log(GI_NS)
F[indexes.GI_news_22] = log(GI_news_22Prime) - log(GI_news_23) - log(GI_NS)
F[indexes.GI_news_23] = log(GI_news_23Prime) - log(GI_news_24) - log(GI_NS)
F[indexes.GI_news_24] = log(GI_news_24Prime) - log(GI_news_25) - log(GI_NS)
F[indexes.GI_news_25] = log(GI_news_25Prime) - log(GI_news_26) - log(GI_NS)
F[indexes.GI_news_26] = log(GI_news_26Prime) - log(GI_news_27) - log(GI_NS)
F[indexes.GI_news_27] = log(GI_news_27Prime) - log(GI_news_28) - log(GI_NS)
F[indexes.GI_news_28] = log(GI_news_28Prime) - log(GI_news_29) - log(GI_NS)
F[indexes.GI_news_29] = log(GI_news_29Prime) - log(GI_news_30) - log(GI_NS)
F[indexes.GI_news_30] = log(GI_news_30Prime) - log(GI_news_31) - log(GI_NS)
F[indexes.GI_news_31] = log(GI_news_31Prime) - log(GI_news_32) - log(GI_NS)
F[indexes.GI_news_32] = log(GI_news_32Prime) - log(GI_news_33) - log(GI_NS)
F[indexes.GI_news_33] = log(GI_news_33Prime) - log(GI_news_34) - log(GI_NS)
F[indexes.GI_news_34] = log(GI_news_34Prime) - log(GI_news_35) - log(GI_NS)
F[indexes.GI_news_35] = log(GI_news_35Prime) - log(GI_news_36) - log(GI_NS)
F[indexes.GI_news_36] = log(GI_news_36Prime) - log(GI_news_37) - log(GI_NS)
F[indexes.GI_news_37] = log(GI_news_37Prime) - log(GI_news_38) - log(GI_NS)
F[indexes.GI_news_38] = log(GI_news_38Prime) - log(GI_news_39) - log(GI_NS)
F[indexes.GI_news_39] = log(GI_news_39Prime) - log(GI_news_40) - log(GI_NS)
F[indexes.GI_news_40] = log(GI_news_40Prime) - XSS[indexes.GISS]  - log(GI_NS)

# Current GI equals current-period announcement
F[indexes.GI] = log(GI) - log(GI_news_1)

# Construction pipeline of public capital
F[indexes.GI_lag1] = (log(GI_lag1Prime)) - (log(GI))
F[indexes.GI_lag2] = (log(GI_lag2Prime)) - (log(GI_lag1))
F[indexes.GI_lag3] = (log(GI_lag3Prime)) - (log(GI_lag2))
F[indexes.GI_lag4] = (log(GI_lag4Prime)) - (log(GI_lag3))

# Finished public capital
F[indexes.KG] = (log(KGPrime)) - (log((1.0 - m_par.δ_KG) * KG + GI_lag4))

# Law of motion for the TFP shock
F[indexes.TFP] = (log(TFPPrime)) - (m_par.ρ_TFP * log(TFP))

# Effective TFP
F[indexes.Z] = (log(Z)) - 
    (log(TFP) + 
     m_par.η_KG * (log(KG) - XSS[indexes.KGSS])
    )

## Asset markets --------------------------------------------------------------------------

# Asset pricing equation for tradable stocks
F[indexes.qΠ] =
    (log(RBPrime / πPrime)) -
    (log(((qΠPrime - 1.0) * (1 - ιΠ) + ωΠ * Π_FPrime) / (qΠ - 1.0)))

# Return on liquid assets
F[indexes.RL] =
    (log(RL)) - (log(A * ((RB * Bgov + π * ((qΠ - 1.0) * (1 - ιΠ) + ωΠ * Π_F)) / B)))

# Return on liquid debt
F[indexes.RD] = (log(RD)) - (log(RRD .* π))

# Total liquidity demand
F[indexes.Bgov] = (log(B)) - (log(Bgov + (qΠlag - 1.0)))

# Ex-post liquidity premium
F[indexes.LP] = (log(LP)) - (log((q + RK - 1.0) / qlag) - log(RB / π))

# Ex-ante liquidity premium
F[indexes.LPXA] = (log(LPXA)) - (log((qPrime + RKPrime - 1.0) / q) - log(RBPrime / πPrime))

# Private bond return fed-funds spread (produces goods out of nothing if negative)
F[indexes.A] = (log(APrime)) - (m_par.ρ_A * log(A))

# Real rates on liquid assets
F[indexes.RRL] = (log(RRL)) - (log(RL / π))
# This variable needs to be set for the package!

# Real rates on liquid debt
F[indexes.RRD] = (log(RRD)) - (log(borrowing_rate_ss(RRL, m_par)))
# This variable needs to be set for the package!

## Market clearing ------------------------------------------------------------------------

# Resource constraint
F[indexes.C] =
    (log(Y - G - GI - I - BD * (RRD .- RRL) + (1.0 - 1.0 ./ A) * RL * B / π)) - (log(C))

## Additional definitions -----------------------------------------------------------------

# Bond to Output ratio
F[indexes.BY] = (log(BY)) - (log(B / Y))

# Tax to output ratio
F[indexes.TY] = (log(TY)) - (log(T / Y))

# Total assets by accounting identity
F[indexes.TotalAssets] = (log(TotalAssets)) - (log(qlag * K + B))

## ----------------------------------------------------------------------------------------
## Closing the aggregate model, see documentation for details
## ----------------------------------------------------------------------------------------

#=

Do not delete the following lines of code!

These equations are overwritten in FSYS by the corresponding aggregation equations of
households' decisions. Here, they are simply set to close the aggregate model. This is a
trick that is exploited in the estimation when only the derivatives with respect to
aggregates is needed. These derivatives are still correct since the left-hand-side of the
equations are the same in both the purely aggregate as well as the complete model.

=#

# Scaling factor for individual productivity
F[indexes.Htilde] = (log(Htilde)) - (XSS[indexes.HtildeSS])
# This variable needs to be set for the package!

# Capital market clearing
F[indexes.K] = (log(K)) - (XSS[indexes.KSS])

# Bond market clearing
F[indexes.B] = (log(B)) - (XSS[indexes.BSS])

# IOUs
F[indexes.BD] = (log(BD)) - (XSS[indexes.BDSS])

## ----------------------------------------------------------------------------------------
## Other distributional statistics
## ----------------------------------------------------------------------------------------

#=

# TO BE EXPLAINED! CURRENTLY, THESE ARE HARD-CODED IN FSYS, FIX THIS!

=#

# other distributional statistics not used in other aggregate equations and not changing
# with parameters, but potentially with other aggregate variables are NOT included here.
# They are found in FSYS.
