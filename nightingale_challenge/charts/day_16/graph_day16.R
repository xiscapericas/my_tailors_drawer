## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(waffle)
library(ggplot2)
library(colormap)
library(dplyr)
library(hrbrthemes)
library(extrafont)

# Get Data ----------------------------------------------------------------
cha_death <- fread('../day_6/character-deaths.csv')
## Death per Book
death_per_book <- cha_death[!duplicated(Name)][!is.na(`Book of Death`), .N, by = `Book of Death`][, prop := N / sum(N)][order(`Book of Death`)]

# Graph -------------------------------------------------------------------
### https://github.com/hrbrmstr/waffle
mycolor <- colormap(colormap=colormaps$viridis, nshades=max(death_per_book[, .N]))
mycolor <- sample(mycolor, length(mycolor))

ggplot(death_per_book, aes(label = `Book of Death`, values = N)) + 
  geom_pictogram(n_rows = 10, aes(colour = `Book of Death`), flip = TRUE, make_proportional = TRUE) +
  scale_color_manual(
    name = NULL,
    values = mycolor, 
    labels = death_per_book[, `Book of Death`]
  ) +
  scale_label_pictogram(
    name = NULL,
    values = c("skull"),
    labels =death_per_book[, `Book of Death`]
  ) + 
  coord_equal() + 
  theme_bw() + 
  theme(
    panel.border = element_blank(), 
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(), 
    axis.line = element_blank(), 
    axis.ticks = element_blank(), 
    legend.position = 'bottom', 
    plot.title = element_text(hjust = 0.5, size = 20, family = 'Arial', face = 'bold'), 
    plot.caption = element_text(family = 'Arial')
  ) + 
  theme_enhance_waffle() + 
  labs(title = 'Deaths in GOT', 
       caption = 'By Xisca Pe')
  
