## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(ggplot2)
library(ggridges)

# Data --------------------------------------------------------------------
pokemon <- fread('pokemon.csv')
pokemon_attack_type <- pokemon[, .(attack = `Attack`, primary_type = as.factor(`Type 1`), name = Name, generation = as.factor(Generation))]

# Graph -------------------------------------------------------------------
mycolor <- rev(c("#33ccff", "#bfbfbf", "#808080", "#8000ff", "#ccb3ff", "#ffccff",
             "#ccffff", "#996633", "#009933", "#4d0099", "#ffffcc", "#ff6600",
             "#cc9900", "#ff66ff", "#ffff00", "#ff0066", "#000000", "#00cc99"))

ggplot(pokemon_attack_type, aes(x = attack, y = primary_type, fill = primary_type)) +
  geom_density_ridges(alpha = 0.7) + xlab('Attack Stat') + ylab('Primary type') + 
  theme_ridges() + scale_fill_manual(values = mycolor) + 
  theme(legend.position = "none", axis.title.x = element_text(hjust = 0.5), 
        axis.title.y = element_text(hjust = 0.5), 
        plot.title = element_text(hjust = 0.5, size = 20)) + 
  labs(title = 'Attack distribution per pokemon type', caption = 'By Xisca Pe')

