## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(plotly)
library(colormap)

# Get Data ----------------------------------------------------------------
cha_death <- fread('character-deaths.csv')
## Death per Book
death_per_book <- cha_death[!duplicated(Name)][!is.na(`Book of Death`), .N, by = `Book of Death`][, prop := N / sum(N)]
death_per_book <- death_per_book[order(-N)][, `:=` (
  ymax = cumsum(prop), 
  ymin = c(0, cumsum(prop)[1:(nrow(death_per_book)-1)]))
][, `:=` (
  label_position = (ymax + ymin) / 2, 
  label = paste0('Book ', `Book of Death`)
)][, book_of_death := as.factor(`Book of Death`)]

# Graph -------------------------------------------------------------------
## Colors 
mycolor <- colormap(colormap=colormaps$velocity_green, nshades=max(death_per_book[, .N]))
mycolor <- sample(mycolor, length(mycolor))


fig <- plot_ly(data = death_per_book[order(prop)][, .(prop, book_of_death, label)], 
               labels = ~book_of_death, values = ~prop, 
               textposition = 'inside', hoverinfo = "label", 
               marker = list(colors = mycolor, line = list(color = '#FFFFFF', width = 1)))
fig <- add_pie(fig, hole = 0.6)
layout(fig, title = "GOT % Deaths per Book",  showlegend = T,
       xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
       yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))




