## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(ggplot2)
library(ape)
library(ggdendro)
mycolor <- rev(c("#33ccff", "#bfbfbf", "#808080", "#8000ff", "#ccb3ff", "#ffccff",
                 "#ccffff", "#996633", "#009933", "#4d0099", "#ffffcc", "#ff6600",
                 "#cc9900", "#ff66ff", "#ffff00", "#ff0066", "#000000", "#00cc99"))

# Data --------------------------------------------------------------------
pokemons <- fread('../day_7/PokemonDatabase.csv')[!duplicated(`Pokemon Name`)]

## Create numeric for type 
pokemons[, id := 1]
pokemons_c <- dcast.data.table(pokemons, `Pokemon Name` ~ `Primary Type`, fun.aggregate = sum, value.var = 'id')[pokemons, on = 'Pokemon Name']

## Create small dataset 
features <- c('Pokemon Height', 'Pokemon Weight', 'Base Happiness', 'Health Stat', 'Attack Stat', 'Defense Stat', 'Speed Stat', 'Base Stat Total')
pokemons_c <- pokemons[`Pokedex Number` %in% seq(1,101,1), c('Pokemon Name', features, pokemons[, unique(`Primary Type`)]), with = F][!duplicated(`Pokemon Name`)]

## Introduce NA for wrong data 
numeric_vars <- setdiff(colnames(pokemons_c), c('Pokemon Name'))
sapply(numeric_vars, function(x) {
  pokemons_c[!grep('[0-9]', get(x)), (x) := NA]
})
## Convert to numeric 
pokemons_c[, (numeric_vars) := lapply(.SD, function(x) as.numeric(x)), .SDcols = numeric_vars]


## Distance 
pokemons_c <- as.data.frame(pokemons_c)
rownames(pokemons_c) <- pokemons_c$`Pokemon Name`
pokemons_c$`Pokemon Name` <- NULL
dd <- dist(scale(pokemons_c), method = "euclidean")


# Graph -------------------------------------------------------------------
hc <- hclust(dd, method = "ward.D2")
## All functions: https://atrebas.github.io/post/2019-06-08-lightweight-dendrograms/
hcdata <- dendro_data_k(hc, pokemons[, uniqueN(`Primary Type`)])
p <- plot_ggdendro(hcdata,
                   fan         = TRUE,
                   scale.color = sample(c(mycolor, 'white')),
                   label.size  = 4,
                   nudge.label = 0.02,
                   expand.y    = 0.4)

p + theme_void() + 
  theme(panel.background = element_rect(fill = '#a1d5f7'), 
        plot.title = element_text(hjust = 0.5, size = 15)) + 
  labs(title = 'Pokemon Generation One Clustering', 
       caption = 'By Xisca Pe')
