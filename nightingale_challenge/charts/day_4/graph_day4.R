## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(ggplot2)

# Auxiliar ----------------------------------------------------------------
FormatYAxis <- function(tx) {
  
  if (min(tx, na.rm = T) > 1000) {
    div <- findInterval(as.numeric(gsub("\\,", "", tx)), c(0, 1e3, 1e6, 1e9, 1e12))  # modify this if negative numbers are possible
    res_format <- paste(round( as.numeric(gsub("\\,","",tx))/10^(3*(div-1)), 2), c("","K","M","B","T")[div] )
  } else {
    res_format <- paste0("$", prettyNum(tx, big.mark=","))
  }
  return(res_format)
}

# Get Data ----------------------------------------------------------------
avocado <- fread('avocado.csv') #https://www.kaggle.com/neuromusic/avocado-prices
avocado[, date := as.Date(Date)]
avocado_ag <- avocado[, .(
                          mean_price = mean(AveragePrice), 
                          total_volume = sum(`Total Volume`), 
                          total_bags = sum(`Total Bags`)
                        ), by = .(date, type)]

avocado_m <- melt(avocado_ag, id.vars = c('date', 'type'), 
                  measure.vars = setdiff(colnames(avocado_ag), c('date', 'type')))

# Graph -------------------------------------------------------------------
ggplot(avocado_m, aes(x = date, y = value, color = type)) + geom_line(alpha = 0.7) + 
  facet_wrap(type ~ variable, scales = 'free_y') + 
  theme_bw() + theme(legend.position = 'none', plot.title = element_text(hjust = 0.5, size = 20), 
                     axis.title.y = element_blank()) + 
  ggtitle('Avocados price, volume and bags sold evolution') + 
  scale_color_manual(values = c('#AA3939', '#378B2E')) + xlab('Date') + ylab('Value') + 
  scale_y_continuous(labels = FormatYAxis)
  

