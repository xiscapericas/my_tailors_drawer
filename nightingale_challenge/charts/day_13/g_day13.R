## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(dygraphs)
library(data.table)
library(viridis)
paleta_colores <- c('#003f5c', '#bc5090', '#ffa600')

# Auxiliar ----------------------------------------------------------------
FormatYAxis <- function(tx) {
  div <- findInterval(as.numeric(gsub("\\,", "", tx)), c(0, 1e3, 1e6, 1e9, 1e12))  # modify this if negative numbers are possible
  res_format <- paste(round( as.numeric(gsub("\\,","",tx))/10^(3*(div-1)), 2), c("","K","M","B","T")[div] )
  return(res_format)
}

# Data --------------------------------------------------------------------
bicing <- fread('2019_01_Gener_BICING_US.csv')
## Aggregate by day 
bicing[, date := as.Date(dateTime)]
usage_by_date <- bicing[, lapply(.SD, sum), .SDcols = setdiff(colnames(bicing), c('dateTime', 'date', 'error')), by = date]
usageby_date_m <- melt(usage_by_date, id.vars = 'date', measure.vars = setdiff(colnames(usage_by_date), 'date'))

# Graph -------------------------------------------------------------------

ggplot(usageby_date_m, aes(x = date, y = value, color = variable)) + 
  geom_line(size = 1.5) + geom_point(size = 3) + facet_wrap(variable ~ ., scales = 'free') + 
  geom_ribbon(aes(x= date, ymax= value, fill = variable), ymin=0, alpha=0.3) + 
  theme_bw() + scale_color_manual(values = paleta_colores) + 
  scale_fill_manual(values = paleta_colores) + 
  theme(
    legend.position="none", panel.background = element_rect(fill = 'white'),
    plot.title = element_text(hjust = 0.5, size = 20)
  ) + 
  labs(title = 'Bicing usage 2019', caption = 'By Xisca Pe') + 
  xlab('Date') + ylab('# bicycles') + 
  scale_y_continuous(labels = FormatYAxis)




