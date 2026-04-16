library(rdbnomics)
library(dplyr)
library(tidyr)
library(lubridate)
library(RJDemetra)
library(readr)
library(wid)

### ---------- Download raw data ------------
## GDP (chain linked values 2020; season & calendar adjusted)
# gdp_agg <- rdb(ids = "Eurostat/namq_10_gdp/Q.CLV20_MEUR.SCA.B1GQ.DE")

# gdp_agg <- gdp_agg %>%
#     rename(gdp = value) %>%
#     select(period, gdp) %>%
#     mutate(gdp = gdp * 1e6) # from million to euro

## Government Consumption
# (real) cons_gov_agg <- rdb(ids = "Eurostat/namq_10_gdp/Q.CLV20_MEUR.SCA.P3_S13.DE")

# current prices in million euro, season & calendar adjusted
cons_gov_nom <- rdb(ids = "Eurostat/namq_10_gdp/Q.CP_MEUR.SCA.P3_S13.DE")

cons_gov_nom <- cons_gov_nom %>%
    rename(cons_gov = value) %>%
    select(period, cons_gov) %>%
    mutate(cons_gov = cons_gov * 1e6) # from million to euro

## Private Consumption (HH & non-profits)
# (real) cons_priv_agg <- rdb(ids = "Eurostat/namq_10_gdp/Q.CLV20_MEUR.SCA.P31_S14_S15.DE")

cons_priv_nom <- rdb(ids = "Eurostat/namq_10_gdp/Q.CP_MEUR.SCA.P31_S14_S15.DE")

cons_priv_nom <- cons_priv_nom %>%
    rename(cons_priv = value) %>%
    select(period, cons_priv) %>%
    mutate(cons_priv = cons_priv * 1e6) # from million to euro

## Total Investments
# Gross fixed capital formation (chain linked values 2020; season & calendar adjusted)
#inv_tot_agg <- rdb(ids = "Eurostat/namq_10_gdp/Q.CLV20_MEUR.SCA.P51G.DE")
# inv_tot_agg <- rdb(ids = "OECD/DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR/Q.Y.DEU.S1.S1.P51G._Z._T._Z.XDC.L.N.T0102")

inv_tot_nom <- rdb(ids = "OECD/DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR/Q.Y.DEU.S1.S1.P51G._Z._T._Z.XDC.V.N.T0102")

inv_tot_nom <- inv_tot_nom %>%
    rename(inv_tot = value) %>%
    select(period, inv_tot) %>%
    mutate(inv_tot = inv_tot * 1e6) # from million to euro

## Government Investments
# inv_gov_agg <-rdb(ids = "OECD/DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR/Q.Y.DEU.S13.S1.P51G._Z._T._Z.XDC.L.N.T0102")

inv_gov_nom <- rdb(ids = "OECD/DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR/Q.Y.DEU.S13.S1.P51G._Z._T._Z.XDC.V.N.T0102")

inv_gov_nom <- inv_gov_nom %>%
    rename(inv_gov = value) %>%
    select(period, inv_gov) %>%
    mutate(inv_gov = inv_gov * 1e6) # from million to euro

## Private Investments
inv_priv_nom = inv_tot_nom$inv_tot - inv_gov_nom$inv_gov
inv_priv_nom <- data.frame(period = inv_tot_nom$period, inv_priv = inv_priv_nom)

## Output (consumption + investment; no net exports)
gdp_nom <- cons_priv_nom$cons_priv + cons_gov_nom$cons_gov + inv_tot_nom$inv_tot
gdp_nom <- data.frame(period = inv_tot_nom$period, gdp = gdp_nom) %>%
    filter(period <= as.Date("2025-07-01")) # drop 2025 Q4 which is missing for some consumption

## Wages
# wage & salary income (season, not calendar adjusted; current prices)
# 2025 missing
#wages_nom <- rdb(ids = "ECB/MNA/Q.Y.DE.W2.S1.S1.D.D11._Z._T._Z.EUR.V.N")

wages_nom <- rdb(ids = "Eurostat/namq_10_a10/Q.CP_MEUR.NSA.TOTAL.D11.DE")

wages_nom <- wages_nom %>%
    rename(wages = value) %>%
    select(period, wages) %>%
    mutate(wages = wages * 1e6) # from million to euro

## Hours Worked
# Total employment (season & calendar adjusted), all NACE activities, in thousand
hours_raw <- rdb(ids = "Eurostat/namq_10_a10_e/Q.THS_HW.TOTAL.SCA.EMP_DC.DE")

hours_raw <- hours_raw %>%
    rename(hours = value) %>%
    select(period, hours) %>%
    mutate(hours = hours * 1000) # from thousand to hour

## GDP Deflator
# implicit deflator GDP (2020=100; season & calendar adjusted)
deflator_raw <- rdb(ids = "Eurostat/namq_10_gdp/Q.PD20_EUR.SCA.B1GQ.DE")

deflator_raw <- deflator_raw %>%
    rename(deflator = value) %>%
    select(period, deflator) 

## Interest Rate 
#3 months EURIBOR (only from 1994)
euribor_raw <- rdb(ids = "ECB/FM/Q.U2.EUR.RT.MM.EURIBOR3MD_.HSTA")

euribor_raw <- euribor_raw %>%
    rename(interest = value) %>%
    select(period, interest) %>%
    mutate(interest = interest / 100) # from percentage to decimal

## Population
# Total population (season & calendar adjusted), in thousand
# pop_tsd <- rdb(ids = "Eurostat/namq_10_pe/Q.THS_PER.SCA.POP_NC.DE")

# Working-age population (15-64)
pop_url <- "https://rplumber.ilo.org/data/indicator?id=POP_XWAP_SEX_AGE_NB_Q&ref_area=DEU&sex=SEX_T&classif1=AGE_YTHADULT_Y15-64&timefrom=1991&timeto=2025&type=label&format=.csv"
pop_tsd <- read_csv(pop_url)

pop <- pop_tsd %>%
    rename(population = obs_value) %>%
    mutate(population = population * 1000,
        period = yq(time)) %>% # from thousand to person
    select(period, population)

# fill missing quarters before 2005 (currently same value from Q2 onwards)
pop <- pop %>%
    complete(period = seq.Date(as.Date("1991-01-01"), max(pop$period), by = "quarter")) %>%
    fill(population, .direction = "down") %>%
    fill(population, .direction = "up")


## Top 10% income share
T10Ishare <- download_wid(indicator = "sptinc", 
                            perc = "p90p100",
                            areas = "DE",
                            pop = "j", # equal split adults
                            ages = "999") %>%
                            filter(year >= 1991) %>%
                            select(year, value) %>%
                            rename(T10Ishare = value)

# convert to quarterly structure with same value for all quarters in a year
T10Ishare <- T10Ishare %>%
    mutate(year = as.Date(paste0(year, "-01-01"))) %>%
    # complete(year = seq.Date(as.Date("1991-01-01"), as.Date("2025-10-01"), by = "quarter")) %>%
    # fill(T10Ishare, .direction = "down") %>%
    # fill(T10Ishare, .direction = "up") %>%
    rename(period = year)

## Top 10% wealth share
T10Wshare <- download_wid(indicator = "shweal", 
                            perc = "p90p100",
                            areas = "DE", 
                            pop = "j", # equal split adults
                            ages = "999") %>%
                            filter( year >= 1991) %>%
                            select(year, value) %>%
                            rename(T10Wshare = value)

T10Wshare <- T10Wshare %>%
    mutate(year = as.Date(paste0(year, "-01-01"))) %>%
    # complete(year = seq.Date(as.Date("1991-01-01"), as.Date("2025-10-01"), by = "quarter")) %>%
    # fill(T10Wshare, .direction = "down") %>%
    # fill(T10Wshare, .direction = "up") %>%
    rename(period = year)


### --------- Real values ------------
## Merge dataset
master <- gdp_nom %>%
    left_join(cons_gov_nom, by = "period") %>%
    left_join(cons_priv_nom, by = "period") %>%
    left_join(inv_gov_nom, by = "period") %>%
    left_join(inv_priv_nom, by = "period") %>%
    left_join(wages_nom, by = "period") %>%
    left_join(hours_raw, by = "period") %>%
    left_join(euribor_raw, by = "period") %>%
    left_join(deflator_raw, by = "period") %>%
    left_join(T10Ishare, by = "period") %>%
    left_join(T10Wshare, by = "period") %>%
    left_join(pop, by = "period")

master_real <- master %>%
    mutate(across(c(gdp, cons_gov, cons_priv, inv_gov, inv_priv)
        , ~ .x / (deflator / 100))) %>%
    mutate(wages = (wages / hours) / (deflator / 100)) %>%
    select(period, gdp, cons_gov, cons_priv, inv_gov, inv_priv, 
        wages, hours, interest, population, deflator, T10Ishare, T10Wshare)


### ---------- Seasonal & calendar adjustment ------------
# Convert unadjusted wages to a quarterly ts object
start_year <- year(min(master_real$period))
start_qtr  <- quarter(min(master_real$period))

wages_ts <- ts(
    master_real$wages,
    start     = c(start_year, start_qtr),
    frequency = 4
)

# X-13ARIMA-SEATS from unadjusted series 
# RSA5c is the Eurostat standard spec for quarterly national
# accounts: automatic ARIMA selection + trading-day (TD7) +
# Easter regressors + full seasonal and calendar adjustment
# in a single joint estimation step.
spec  <- x13_spec(
  spec               = "RSA5c",
  tradingdays.option = "None",
  easter.enabled     = TRUE,
  easter.duration    = 8
)

model <- x13(wages_ts, spec)
summary(model)

# Extract the seasonally & calendar adjusted series
wages_sca_ts <- model$final$series[, "sa"]

# Convert back to a data frame 
wages_sca <- data.frame(
    period = master_real$period,
    wages  = as.numeric(wages_sca_ts)
)

# Plot raw vs adjusted to verify the adjustment looks sensible
plot(wages_ts, col = "grey60", lwd = 1.5,
     main = "Wages: unadjusted vs seasonally & calendar adjusted",
     ylab = "Million EUR", xlab = "")
lines(wages_sca_ts, col = "steelblue", lwd = 2)
legend("topleft", legend = c("Unadjusted", "SCA"),
       col = c("grey60", "steelblue"), lwd = c(1.5, 2), bty = "n")

# Replace by adjusted wages
master_real$wages <- wages_sca$wages

## Write as per capita values
master_pc <- master_real %>%
    mutate(across(c(gdp, cons_gov, cons_priv, inv_gov, inv_priv, wages, hours)
        , ~ .x / population)) %>%
    select(-population) %>%
    rename(
        gdp_pc = gdp,
        cons_gov_pc = cons_gov,
        cons_priv_pc = cons_priv,
        inv_gov_pc = inv_gov,
        inv_priv_pc = inv_priv,
        wages_pc = wages,
        hours_pc = hours
    )

## Adjusting into growth rates
master_growth <- master_pc %>%
    mutate(
        Ygrowth  = (log(gdp_pc)       - lag(log(gdp_pc))),
        GCgrowth = (log(cons_gov_pc)  - lag(log(cons_gov_pc))),
        Cgrowth  = (log(cons_priv_pc) - lag(log(cons_priv_pc))),
        GIgrowth = (log(inv_gov_pc)   - lag(log(inv_gov_pc))),
        Igrowth  = (log(inv_priv_pc)  - lag(log(inv_priv_pc))),
        wgrowth  = (log(wages_pc)     - lag(log(wages_pc))),
        N        = log(hours_pc),
        pi       = log(deflator)      - lag(log(deflator)),
        RB       = interest / 4,
        TOP10Ishare = log(T10Ishare),
        TOP10Wshare = log(T10Wshare)
    ) %>%
    select(period, Ygrowth, GCgrowth, Cgrowth, GIgrowth, Igrowth, wgrowth, N, pi, RB, TOP10Ishare, TOP10Wshare)

## Compte TS averages
master_growth_avg <- master_growth %>%
    summarise(
        Ygrowth_avg = mean(Ygrowth, na.rm = TRUE),
        GCgrowth_avg = mean(GCgrowth, na.rm = TRUE),
        Cgrowth_avg = mean(Cgrowth, na.rm = TRUE),
        GIgrowth_avg = mean(GIgrowth, na.rm = TRUE),
        Igrowth_avg = mean(Igrowth, na.rm = TRUE),
        wgrowth_avg = mean(wgrowth, na.rm = TRUE),
        N_avg       = mean(N, na.rm = TRUE),
        pi_avg      = mean(pi, na.rm = TRUE),
        RB_avg      = mean(RB, na.rm = TRUE),
        TOP10Ishare_avg = mean(TOP10Ishare, na.rm = TRUE),
        TOP10Wshare_avg = mean(TOP10Wshare, na.rm = TRUE)
    )

master_growth_stationary <- master_growth %>%
    mutate(
        Ygrowth = Ygrowth - master_growth_avg$Ygrowth_avg,
        GCgrowth = GCgrowth - master_growth_avg$GCgrowth_avg,
        Cgrowth = Cgrowth - master_growth_avg$Cgrowth_avg,
        GIgrowth = GIgrowth - master_growth_avg$GIgrowth_avg,
        Igrowth = Igrowth - master_growth_avg$Igrowth_avg,
        wgrowth = wgrowth - master_growth_avg$wgrowth_avg,
        N       = N - master_growth_avg$N_avg,
        pi      = pi - master_growth_avg$pi_avg,
        RB      = RB - master_growth_avg$RB_avg,
        TOP10Ishare = TOP10Ishare - master_growth_avg$TOP10Ishare_avg,
        TOP10Wshare = TOP10Wshare - master_growth_avg$TOP10Wshare_avg
    ) %>%
    select(period, Ygrowth, GCgrowth, Cgrowth, GIgrowth, Igrowth, wgrowth, N, pi, RB, TOP10Ishare, TOP10Wshare)

## Report important averages
# Top 10% shares
T10Iavg <- mean(master_pc$T10Ishare, na.rm = TRUE)
T10Wavg <- mean(master_pc$T10Wshare, na.rm = TRUE)

# Government spending shares
GCshare_avg <- mean(master_pc$cons_gov_pc / master_pc$gdp_pc, na.rm = TRUE)
GIshare_avg <- mean(master_pc$inv_gov_pc / master_pc$gdp_pc, na.rm = TRUE)

## Tax progrssivity P = (AMTR - ATR)/(1 - ATR)


# Income risk


# save as csv
write_csv(master_growth_stationary, "examples/baseline_TTB_10Y/Data/GER_growth.csv", na = "NaN")