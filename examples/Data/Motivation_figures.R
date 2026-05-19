data = read.csv("gov_10a_main__custom_20758448_linear_2_0.csv")

library(tidyverse)
library(patchwork)
library(extrafont)

gfcf <- data %>%
   filter(na_item == "P51G", unit == "PC_GDP", TIME_PERIOD >= "1995") %>%
   select(geo, TIME_PERIOD, gfcf = OBS_VALUE)

cfc <- data %>%
   filter(na_item == "P51C", unit == "PC_GDP", TIME_PERIOD >= "1995") %>%
   select(geo, TIME_PERIOD, cfc = OBS_VALUE)

net_capital <- inner_join(gfcf, cfc, by = c("geo", "TIME_PERIOD")) %>%
   mutate(net_capital = (gfcf - cfc))

country_colors <- case_when(
   unique(net_capital$geo) == "DE" ~ "red",
   unique(net_capital$geo) == "FR" ~ "blue",
   unique(net_capital$geo) == "ES" ~ "orange",
   unique(net_capital$geo) == "EU27_2020" ~ "black",
   TRUE                            ~ "grey60"
   )
names(country_colors) <- unique(net_capital$geo)

ggplot(net_capital, 
   aes(x = TIME_PERIOD, y = net_capital, color = geo)) +
   geom_line(linewidth = 1) +
   geom_point(size = 1.5) +
   scale_color_manual(values = country_colors) +
   scale_y_continuous(labels = scales::comma) +
   labs(
      title = "Change in Net Capital Formation by Country",
      subtitle = "Gross Fixed Capital Formation minus Consumption of Fixed Capital",
      x = "Year",
      y = "Change in Net Capital",
      color = "Country"
   ) +
   theme_bw() +
   theme(
      plot.title    = element_text(face = "bold"),
      legend.position = "none"
   )



net_capital_agg <- net_capital %>%
   group_by(geo) %>%
   arrange(TIME_PERIOD) %>%
   mutate(
      index = cumsum(net_capital),
      index = index - first(index) + 100
   ) %>%
   ungroup()


NetCapital <- ggplot(net_capital_agg, 
       aes(x = TIME_PERIOD, y = index, color = geo)) +
   geom_line(linewidth = 1) +
   scale_color_manual(values = country_colors,
                      breaks = c("DE", "FR", "ES", "EU27_2020"),
                      labels = c("Germany", "France", "Spain", "EU27 average")) +
   scale_y_continuous(labels = scales::comma) +
   scale_x_continuous(expand =c(0, 0.02)) +
   labs(
      #title    = "Net Public Capital Formation for European Countries",
      # subtitle = "Index: 1995 = 100",
      x        = "Year",
      y        = "Index (1995 = 100)",
      #color    = "Country",
      #caption  = "Source: own calculation, Eurostat" 
   ) +
   theme_classic(base_size = 18) +
   theme(
      #plot.title    = element_text(face = "bold"),
      text = element_text(family = "Times New Roman"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey"),
      #panel.background = element_blank(),
      legend.position = c(0.2, 0.85),
      legend.text  = element_text(size = 18),
      legend.title = element_blank()
   )

ggsave("NetCapital.png", width = 12, height = 6, dpi = 600)


### GDP growth
gdp = read.csv("tec00115__custom_20758773_linear_2_0.csv")

gdp_pc <- gdp %>%
   filter(TIME_PERIOD >= "2019", unit == "CLV_PCH_PRE")

p_gdp <- ggplot(gdp_pc,
       aes(x=TIME_PERIOD, y = OBS_VALUE, color = geo)) +
   geom_line(linewidth = 1) +
   #geom_point(size = 1.5) +
   scale_color_manual(values = country_colors,
                      breaks = c("DE", "FR"),
                      labels = c("Germany", "France")) +
   scale_y_continuous(labels = scales::percent_format(scale = 1)) +
   scale_x_continuous(expand =c(0, 0.02)) +
   labs(
      title = "Real GDP Growth",
      x     = "Year",
      y     = "Growth Rate"
   ) +
   theme_classic(base_size = 16) +
   theme(
      #plot.title    = element_text(face = "bold"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey"),
      #panel.background = element_blank(),
      legend.position = "none",
      legend.title = element_blank()
   )

p_cap + p_gdp + plot_layout(widths = c(2, 1)) +
   plot_annotation(title = "Historically low net public investments & Missing recovery after COVID",
                   theme = theme(plot.title = element_text(face = "bold", size = 18)),
                   caption = "Source: own calculations, Eurostat")
   
ggsave("cap_gdp_combined.png", width = 14, height = 6, dpi = 300)
   

