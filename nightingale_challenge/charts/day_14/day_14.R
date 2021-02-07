## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table) 
library(plotly)

# Data --------------------------------------------------------------------
disney_companies_sample <- as.data.table(read.csv("~/Desktop/graph_challenge/day_14/disney_companies_sample.txt", stringsAsFactors = F))

# Graph -------------------------------------------------------------------
p1 <- disney_companies_sample[, plot_ly(
  type = 'treemap', 
  ids = ids, 
  labels = labels, 
  parents = parents
  #marker=list(colorscale='Picnic')
)]
p1 <- p1 %>% layout(uniformtext=list(minsize=10, mode='hide'))
p1
