## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(tree)
library(pROC)
library(rpart)
library('rpart.utils')
library(plotly)
source('auxiliar.R')

# Get Data ----------------------------------------------------------------
heroes_data <- fread('datasets_26532_33799_heroes_information.csv')[
  Publisher %in% c('Marvel Comics', 'DC Comics')]

## Remove NA  
heroes_data[`Skin color` == '-', `Skin color` := NA]
setnames(heroes_data, gsub(' ','_', colnames(heroes_data)))
factor.vars <- c('Publisher', 'Gender', 'Alignment', 'Eye_color', 'Hair_color', 'Skin_color')
heroes_data[, (factor.vars) := lapply(.SD, function(x) as.factor(x)), .SDcols = factor.vars]

## Create decision tree 
htree <- heroes_data[, rpart(Publisher ~ Gender + Height + Weight + Alignment + Eye_color + Hair_color + Skin_color, 
                           method = 'class')]
## Predict 
results <- as.data.table(predict(htree, heroes_data))
setnames(results, paste0('prediction_', colnames(results)))
heroes_data <- cbind(heroes_data, results)
heroes_data[Publisher == 'Marvel Comics', is_marvel := as.factor('1')]
heroes_data[Publisher != 'Marvel Comics', is_marvel := as.factor('0')]

## AUC 
roc <- heroes_data[, roc(response = is_marvel, predictor = `prediction_Marvel Comics`)]
auc(roc) ## 0.48 :( 

# Graph -------------------------------------------------------------------
nodes_source_target <- GenerateNodeSourceValues(htree) #https://stackoverflow.com/questions/52202266/decision-tree-using-rpart-to-produce-a-sankey-diagram
nodes <- nodes_source_target[[1]]
source <- nodes_source_target[[2]]
target <- nodes_source_target[[3]]

p <- plot_ly(
  type = "sankey",
  orientation = "h",
  node = list(
    label = nodes,
    pad = 15, thickness = 20,
    line = list(color = "black",width = 0.5)
  ),
  link = list(
    source = source,
    target = target,
    value=values[-1]
  )
) %>% 
  layout(
    title = "Heroes Classifier Sankey Diagram",
    font = list(size = 10))

## Result 
p
