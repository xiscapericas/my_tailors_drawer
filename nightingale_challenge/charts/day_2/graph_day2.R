## 30DGraphChallenge

# Packages ----------------------------------------------------------------
## rCharts 
#require(devtools)
#install_github('rCharts', 'ramnathv')
library(rCharts)
library(data.table)

# Data --------------------------------------------------------------------
atur <- rbindlist(lapply(list.files('data/'), function(file) {
  fread(paste0('data/', file))
}))
atur_per_sexe_data <- atur[, .(total = sum(`Nombre`)), by = .(Any, Mes, Sexe)]
## Format dates 
atur_per_sexe_data[, mes_any := as.character(as.Date(paste0(Any, '-', ifelse(nchar(Mes) > 1, Mes, paste0(0, Mes)), '-01')))]

# Graph -------------------------------------------------------------------
atur_per_sexe_data_dc <- dcast.data.table(atur_per_sexe_data, mes_any ~ Sexe, value.var = 'total')
m1 <- mPlot(x = "mes_any", y = c("Dones", "Homes"), type = "Line", data = atur_per_sexe_data_dc)
m1$set(pointSize = 2, lineWidth = 2, lineColors=c('red', 'blue'), 
       events = c('2012-01-01', '2014-01-01', '2016-01-01', '2018-01-01'), eventLineColors = 'gray', eventStrokeWidth = 0.1)
m1

