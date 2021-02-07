## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table) 
library(ggplot2)

# Get Data ----------------------------------------------------------------
mcdonalds <- fread('../day_25/datasets_910_1662_menu.csv')
paleta_colores <- c('#ffc72c', '#fbb52a', '#ec7624', '#da291c', 
                    '#b8281d', '#90271e', '#74261e', '#55261e', 
                    '#27251f')

# Graph -------------------------------------------------------------------
ggplot(mcdonalds, aes(x = Category, y = Calories, fill = Category)) + 
  geom_violin(trim=FALSE) +
  geom_boxplot(width=0.1, fill="white") +
  theme_bw() + 
  coord_flip() + 
  scale_fill_manual(values=paleta_colores) + 
  theme(legend.position = 'bottom', 
        plot.title = element_text(hjust = 0.5, size = 20, family = 'Arial', face = 'bold'), 
        plot.caption = element_text(family = 'Arial')) + 
  labs(title = 'McDonalds: Calories per Product Category', 
       caption = 'By Xisca Pe')



## Help: 
mcdonalds[which.max(Calories), .(Item, Calories)]
## Item Calories
## 1: Chicken McNuggets (40 piece)     1880
