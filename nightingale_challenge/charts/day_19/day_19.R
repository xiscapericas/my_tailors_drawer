## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(streamgraph)
library(colormap)

# Get Data ----------------------------------------------------------------
anime <- fread('dataanime.csv')
anime[, start_date := as.Date(`Start airing`)]
anime <- anime[, .(primary_genre = unlist(strsplit(Genres, ','))[1]), by = Title][anime, on = 'Title']
anime[, year := as.integer(year(start_date))]
anime_by_year_genre <- as.data.frame(anime[, .N, by = .(year, primary_genre)][!is.na(year)])


# Graph -------------------------------------------------------------------
mycolor <- colormap(colormap=colormaps$cubehelix, nshades=uniqueN(anime_by_year_genre$primary_genre))
mycolor <- sample(mycolor, length(mycolor))

streamgraph(anime_by_year_genre, 'primary_genre', 'N', 'year', 
            interactive = TRUE) %>%
  sg_axis_x(5, "year", "%Y") %>%
  sg_fill_manual(mycolor) %>%
  sg_legend(show=TRUE, label="Genres: ") 


