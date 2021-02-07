## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(GGally)

# Data --------------------------------------------------------------------
mcdonalds <- fread('../day_25/datasets_910_1662_menu.csv')
## Size 
mcdonalds[, size := as.numeric(gsub("(.* oz \\()([0-9]{2,4})(.*)", '\\2', `Serving Size`))]

## McFlurry 
mcflurry <- mcdonalds[grepl('McFlurry', Item)]
mcflurry_sub <- mcflurry[, .(item = Item, size, calories = Calories, 
                             total_fat = `Total Fat`, cholesterol = Cholesterol, 
                             sugars = Sugars, protein = Protein)]

## Conversion 
mcflurry_sub[, item := as.factor(item)]

## McDonalds paleta 
paleta_colores <- c('#ffc72c', '#fbb52a', '#ec7624', '#da291c', 
                    '#b8281d', '#90271e', '#74261e', '#55261e', 
                    '#27251f')

# Graph -------------------------------------------------------------------
ggparcoord(mcflurry_sub,
           columns = 2:6, groupColumn = 1, order = "anyClass",
           showPoints = TRUE, 
           title = "Parallel Coordinate Plot for McFlurry",
           alphaLines = 0.8) + 
  theme_bw() + 
  scale_color_manual(values = paleta_colores) + 
  xlab('Characteristic') + ylab('Norm Value') + 
  theme(legend.position = 'right', 
        plot.title = element_text(hjust = 0.5, size = 20, family = 'Arial', face = 'bold'), 
        plot.caption = element_text(family = 'Arial'), 
        legend.title = element_blank()) + 
  labs(caption = 'By Xisca Pe')

