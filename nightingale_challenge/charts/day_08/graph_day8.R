## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(plotly)

# Data --------------------------------------------------------------------
pokemon <- fread('../day_7/pokemon.csv')
pokemon_attack_type <- pokemon[, .(attack = `Attack`, defense = `Defense`, total = Total, primary_type = as.factor(`Type 1`), name = Name, generation = as.factor(Generation))]

# Graph -------------------------------------------------------------------
fig <- plot_ly(data = pokemon_attack_type, 
        x=~attack,y=~defense, z=~total, 
        type = "contour",
        contours = list(showlabels = TRUE), 
        colorscale = 'Hot')
layout(fig, title = "Pokemon Attack, Defense and Total power relation",  showlegend = T,
       xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = T),
       yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = T))


