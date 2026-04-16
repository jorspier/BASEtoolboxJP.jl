# Set aggregate steady state variabel values
ASS       = 1.0
TFPSS     = 1.0
ZISS      = 1.0
μSS       = m_par.μ
μwSS      = m_par.μw
τprogSS   = m_par.τ_prog
τlevSS    = m_par.τ_lev

σSS          = 1.0
τprog_obsSS  = 1.0
GshockSS     = 1.0
RshockSS     = 1.0
TprogshockSS = 1.0

SshockSS  = 1.0
# rSS       = 1.0 + interest(KSS,1.0 / m_par.μ, NSS, m_par)
RBSS      = m_par.RB
LPSS      = 1 + rSS - RBSS
LPXASS    = 1 + rSS - RBSS
ISS       = m_par.δ_0 * KSS

πSS       = 1.0
πwSS      = 1.0

BDSS      = eps()
BSS       = m_par.BhhBAR .* YSS

# Calculate taxes and government expenditures
TSS       = dot(1.0, taxrev) + av_tax_rateSS*((1.0 .- 1.0 ./ m_par.μw).*wSS.*NSS)

# println("BSS/YSS: ", BSS/YSS)

# println("StockshareSS: ",(qΠSS_fnc(YSS,m_par.RB,m_par) .- 1.0)/BSS)
BgovSS        = BSS .- qΠSS_fnc(YSS,m_par.RB,m_par) .+ 1.0
# println("BgovSS/YSS: ", BgovSS/YSS)
GSS           = TSS - (m_par.RB./m_par.π-1.0)*BgovSS - GISS
# println("GSS/YSS: ",GSS/YSS)

mcSS      = 1.0 ./ m_par.μ


firm_profitsSS = (1.0 - mcSS) .* YSS
# println("firm_profitsSS: ", firm_profitsSS)
qΠSS      = qΠSS_fnc(YSS,RBSS,m_par)
qΠlagSS   = qΠSS
RLSS      = m_par.RB

CSS       = (YSS - m_par.δ_0 * KSS - GSS - GISS- m_par.Rbar*BDSS)
LMULTSS = ((CSS) - (NSS)^(1+m_par.γ) / (1+m_par.γ))^(-m_par.ξ)

qSS       = 1.0
mcwSS     = 1.0 ./ m_par.μw
mcwwSS    = wSS * mcwSS
uSS       = 1.0
unionprofitsSS = (1.0 - mcwSS) .* wSS .* NSS

BYSS   = BSS / YSS
TYSS   = TSS / YSS
TlagSS = TSS

YlagSS = YSS
BgovlagSS = BgovSS
GlagSS = GSS
IlagSS = ISS
wlagSS = wSS
qlagSS = qSS
ClagSS = CSS
av_tax_ratelagSS = av_tax_rateSS
τproglagSS       = τprogSS

YgrowthSS = 1.0
BgovgrowthSS = 1.0
IgrowthSS = 1.0
wgrowthSS = 1.0
CgrowthSS = 1.0
TgrowthSS = 1.0
HtSS      = 1.0

# Government Investment
GISS = m_par.GI_share * YSS
GI_lag1SS = GISS
GI_lag2SS = GISS
GI_lag3SS = GISS
GI_lag4SS = GISS

KGSS = GISS / m_par.δ_KG

AuthSS = GISS
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

