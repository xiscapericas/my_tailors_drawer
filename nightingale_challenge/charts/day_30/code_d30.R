## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(HistData) #Nightingale Data
library(ggplot2)

# Data --------------------------------------------------------------------
data("Nightingale")
## The rates are: (X / Army) * 12000 ==> annual rate of mortality per 1000 in each month
## Source: https://pointedanalytics.wordpress.com/2013/07/14/plotting-coxcombs-using-ggplot2/

## Filter the data 
nightingale <- as.data.table(Nightingale)[, date := as.Date(Date)][
  year(date) >= 1854 & date < as.Date('1855-04-01')][, labels := Month]

## Adding labels
nightingale[1, labels := paste0(Month, ' ', Year)]
nightingale[10, labels := paste0(Month, ' ', Year)]


## Estimate radius 
EstimateRadius <- function(x, y) {
  return(sqrt((1000 * x/y)/pi))
}

fields <- c('Disease', 'Wounds', 'Other')
nightingale <- cbind(
  nightingale, 
  setnames(nightingale[, lapply(.SD, function(x) {
    EstimateRadius(x, Army)}), .SDcols = fields], c(paste0('r', fields))))[, id := 1:.N]

## Melt 
nightingale_m <- melt(nightingale, 
     id.vars = c('Month', 'labels', 'id'), 
     measure.vars = c(paste0('r', fields)))[, variable := as.character(variable)][order(id, variable)]

nightingale_m[duplicated(labels), labels := '']
colors_paleta <-  c('#B3BDBF', '#9C8D8C','#E8B9B4')

# Graph -------------------------------------------------------------------
ggplot(nightingale_m, 
       aes(x = id, y = value, fill = variable, label = labels, order = variable)) + 
  geom_bar(stat = 'identity', position = 'identity', width = 1, alpha = 0.8) + 
  scale_fill_manual(values = colors_paleta) + 
  coord_polar(start = 11, theta = 'x') + 
  geom_text(vjust=-0.25, size=3) + 
  theme_bw() + 
  theme(
    panel.background = element_blank(), 
    axis.text = element_blank(), 
    panel.border = element_blank(), 
    axis.ticks = element_blank(), 
    axis.title = element_blank(), 
    legend.position = 'bottom', 
    plot.title = element_text(hjust = 0.5, vjust = -20, 
                              size = 20, family = 'Arial', face = 'bold'), 
    plot.caption = element_text(family = 'Arial'), 
    legend.title = element_blank()
  ) + 
  labs(title = 'Diagram of the causes of mortality \n in the army in the east', 
       subtitle = 'April 1854 to March 1855',
       caption = 'Florence Nightingale *')
