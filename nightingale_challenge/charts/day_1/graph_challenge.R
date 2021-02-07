## 30DGraphChallenge



# Packages ----------------------------------------------------------------
library(data.table) 
library(gtrendsR)
library(ggplot2)
paleta_colores <- c('#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600', '#51D231') #https://learnui.design/tools/data-color-picker.html#palette



# Datos -------------------------------------------------------------------
concursantes <- c('gerard', 'hugo', 'nia', 'eva', 'anaju', 'flavio', 'maialen', 'samantha', 'bruno')
concursantes_dt <- rbindlist(lapply(concursantes, function(concursante) {
  as.data.table(
    gtrends(paste0(concursante, ' ot 2020'), time = "today 1-m", geo = 'ES')$interest_over_time
  )
}))


# Graph -------------------------------------------------------------------
concursantes_dt[, date_f := lubridate::ymd(date)]
ggplot(concursantes_dt, aes(x = date, y = hits, fill = keyword)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(~keyword) + scale_fill_manual(values = paleta_colores) + theme_bw() + 
  theme(legend.position =  'none', plot.title = element_text(hjust = 0.5, size = 20)) + 
  xlab('Fecha') + ylab('Hits') + ggtitle('Búsquedas OT2020')




