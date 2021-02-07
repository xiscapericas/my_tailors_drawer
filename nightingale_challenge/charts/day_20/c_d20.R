## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table) 
library(igraph)
set.seed(8592)

# Get Data ----------------------------------------------------------------
## Create correlation matrix 
pokemon_corr_matrix <- as.matrix(dist(scale(pokemons_c[1:30,]), method = "euclidean"))

## Set 0 
pokemon_corr_matrix[is.na(pokemon_corr_matrix)] <- 0
pokemon_corr_matrix[pokemon_corr_matrix > (max(pokemon_corr_matrix)*0.5)] <- 0

## Colors 
colors_list <- rev(c("#33ccff", "#bfbfbf", "#808080", "#8000ff", "#ccb3ff", "#ffccff",
                 "#ccffff", "#996633", "#009933", "#4d0099", "#ffffcc", "#ff6600",
                 "#cc9900", "#ff66ff", "#ffff00", "#ff0066", "#000000", "#00cc99"))

my_color <- colors_list[as.numeric(as.factor(pokemons$`Primary Type`))]

# Graph -------------------------------------------------------------------
# Make an Igraph object from this matrix:
network <- graph_from_adjacency_matrix(pokemon_corr_matrix, weighted=T, mode="undirected", diag=F)

# ## Graph  
par(bg="white", mar=c(0,0,0,0), bty= 'l')
plot(network, 
     # === vertex
     vertex.color=my_color,   
     vertex.frame.color = "black",               
     vertex.shape="circle",                       
     vertex.size=20,     
     
     # === vertex label
     vertex.label.color="black",
     vertex.label.family="Arial",                  # Font family of the label (e.g.“Times”, “Helvetica”)
     vertex.label.font=2,                          # Font: 1 plain, 2 bold, 3, italic, 4 bold italic, 5 symbol
     vertex.label.cex=1,                           # Font size (multiplication factor, device-dependent)
     vertex.label.dist=0,                          # Distance between the label and the vertex
     vertex.label.degree=0 ,                       # The position of the label in relation to the vertex (use pi)
     
     # === Edge
     edge.color="gray",                           # Edge color
     edge.width=1,                                 # Edge width, defaults to 1
     edge.arrow.size=1,                            # Arrow size, defaults to 1
     edge.arrow.width=1,                           # Arrow width, defaults to 1
     edge.lty="dashed",                             # Line type, could be 0 or “blank”, 1 or “solid”, 2 or “dashed”, 3 or “dotted”, 4 or “dotdash”, 5 or “longdash”, 6 or “twodash”
     edge.curved=0    
)

## Text and Legend 
text(0,0,"Pokemon Network", col="black", cex=3)
legend(x=-1.5, y=-1.05, 
       #legend=unique(pokemons$`Primary Type`), 
       legend = intersect(unique(pokemons$`Primary Type`), pokemons[1:30, unique(`Primary Type`)]),
       col = unique(colors_list[as.numeric(as.factor(pokemons$`Primary Type`))]) , 
       bty = "n", pch=20 , pt.cex = 2, cex = 1,
       text.col="black" , horiz = T)
