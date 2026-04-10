# Steady state values that are given already:
# - outputs of `Ksupply`, in particular: KSS, BSS
# - all variables in args_hh_prob_names
# - all variables in distr_names

## Shocks
ZSS = m_par.Z
ZISS = 1.0
μSS = m_par.μ
μwSS = m_par.μw
ASS = 1.0
RshockSS = 1.0
GshockSS = 1.0
TprogshockSS = 1.0
SshockSS = 1.0
TFPSS = 1.0

## Growth rates
YgrowthSS = 1.0
BgovgrowthSS = 1.0
IgrowthSS = 1.0
wgrowthSS = 1.0
CgrowthSS = 1.0
TgrowthSS = 1.0
GIgrowthSS = 1.0

## Further assumptions (partly also used in args_hh_prob)
mcSS = 1.0 ./ μSS
mcwSS = 1.0 ./ μwSS
πSS = 1.0
πwSS = 1.0
uSS = 1.0

## Variables (that are not already defined)

# nominal interest rates
RBSS = m_par.RRB .* πSS
RLSS = RBSS
RDSS = RRDSS .* πSS

# production side
wFSS = wage(mcSS, ZSS, KSS, NSS, m_par)
YSS = output(ZSS, KSS, NSS, m_par)
ISS = m_par.δ_0 * KSS
Π_FSS = (1.0 - mcSS) .* YSS

# government investment
GISS = m_par.GI_share * YSS
# SpSS = GISS / m_par.ϕ_GI
AuthSS = GISS
KGSS = GISS / m_par.δ_KG

# Lags for government investment and authorizations
Auth_lag1SS = AuthSS
Auth_lag2SS = AuthSS
Auth_lag3SS = AuthSS
Auth_lag4SS = AuthSS
Auth_lag5SS = AuthSS
Auth_lag6SS = AuthSS
Auth_lag7SS = AuthSS
Auth_lag8SS = AuthSS
Auth_lag9SS = AuthSS
Auth_lag10SS = AuthSS
Auth_lag11SS = AuthSS
Auth_lag12SS = AuthSS
Auth_lag13SS = AuthSS
Auth_lag14SS = AuthSS
Auth_lag15SS = AuthSS
Auth_lag16SS = AuthSS
Auth_lag17SS = AuthSS
Auth_lag18SS = AuthSS
Auth_lag19SS = AuthSS
Auth_lag20SS = AuthSS
Auth_lag21SS = AuthSS
Auth_lag22SS = AuthSS
Auth_lag23SS = AuthSS
Auth_lag24SS = AuthSS
Auth_lag25SS = AuthSS
Auth_lag26SS = AuthSS
Auth_lag27SS = AuthSS
Auth_lag28SS = AuthSS
Auth_lag29SS = AuthSS
Auth_lag30SS = AuthSS
Auth_lag31SS = AuthSS
Auth_lag32SS = AuthSS
Auth_lag33SS = AuthSS
Auth_lag34SS = AuthSS
Auth_lag35SS = AuthSS
Auth_lag36SS = AuthSS
Auth_lag37SS = AuthSS
Auth_lag38SS = AuthSS
Auth_lag39SS = AuthSS

GI_lag1SS = GISS
GI_lag2SS = GISS
GI_lag3SS = GISS
GI_lag4SS = GISS
GI_lag5SS = GISS
GI_lag6SS = GISS
GI_lag7SS = GISS
GI_lag8SS = GISS
GI_lag9SS = GISS
GI_lag10SS = GISS
GI_lag11SS = GISS
GI_lag12SS = GISS

# financial market
LPSS = RKSS / (RBSS / πSS)
LPXASS = LPSS
BDSS = -aggregate_asset(distrSS, :b, n_par, 0.0)
qΠSS = (m_par.ωΠ .* Π_FSS) ./ (RBSS / πSS .- 1 .+ m_par.ιΠ) + 1.0
BgovSS = BSS .- qΠSS .+ 1.0
TotalAssetsSS = BSS + qSS * KSS

# fiscal side
RK_before_taxesSS = ((RKSS - 1.0) ./ (1.0 - (TkSS - 1.0))) + 1.0
TRSS = transfer_scheme(n_par, m_par, args_hh_prob; distr_h = distrSS.h)

# other definitions
τprogSS = TprogSS .- 1.0

# jointly determine C, T, G (interacted through consumption tax)
# resource constaint, plugged in government budget constraint and tax revenues
CSS =
    (
        YSS - ISS - (RRDSS .- RRLSS) * BDSS + (RRLSS - 1.0) * BgovSS - (
            (TbarSS .- 1.0) .* (wHSS .* NSS + Π_ESS + Π_USS) +
            (TkSS .- 1.0) .* (RK_before_taxesSS .- 1.0) .* KSS - (TRSS .- 1.0)
        )
    ) ./ (1.0 + (TcSS .- 1.0))

# tax revenues
TSS =
    (TbarSS .- 1.0) .* (wHSS .* NSS + Π_ESS + Π_USS) +
    (TcSS .- 1.0) .* CSS +
    (TkSS .- 1.0) .* (RK_before_taxesSS .- 1.0) .* KSS - (TRSS .- 1.0)

# government spending from budget constraint
GSS = TSS - (RRLSS - 1.0) * BgovSS - GISS

# remaining aggregates
BYSS = BSS / YSS
TYSS = TSS / YSS

## Lags
YlagSS = YSS
BgovlagSS = BgovSS
TlagSS = TSS
IlagSS = ISS
wFlagSS = wFSS
qlagSS = qSS
ClagSS = CSS
TbarlagSS = TbarSS
TproglagSS = TprogSS
qΠlagSS = qΠSS
