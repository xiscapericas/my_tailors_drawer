## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(ggplot2)

# Data --------------------------------------------------------------------
temp_bcn <- fread('2019_x2_barcelona_zoo.csv')
temp_bcn[, date := lubridate::dmy(DATA_LECTURA)]
temp_bcn[, `:=` (
  day = lubridate::day(date), 
  month = as.factor(lubridate::month(date, label = T, abbr = F)), 
  year = as.factor(year(date))
)]

# Graph -------------------------------------------------------------------
ggplot(temp_bcn[year %in% c(2016, 2017, 2018, 2019)], aes(x = month, y = day, fill = TM)) + 
  geom_tile(color= "white",size=0.1) + 
  scale_fill_viridis(name="Hrly Temps C",option ="C") + 
  facet_grid(~year) + 
  scale_y_continuous(trans = "reverse", breaks = temp_bcn[, unique(day)]) + 
  scale_x_discrete(guide = guide_axis(n.dodge=3)) +
  theme_bw() + labs(title = 'Temperatura BCN 2016-2019', caption = 'By Xisca Pe') + 
  xlab('Mes') + ylab('Dia') + 
  theme(legend.position = 'right',
        axis.title.x = element_text(vjust = -1), 
        panel.grid = element_blank(), 
        plot.title = element_text(hjust = 0.5, size = 20))
  
