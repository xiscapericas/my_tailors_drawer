## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(readxl)
library(circlize)

# Get Data ----------------------------------------------------------------
eurovision <- as.data.table(read_xlsx('../day_9/eurovision_song_contest_1975_2019.xlsx'))
setnames(eurovision, make.names(colnames(eurovision)))
top_five <- c('France', 'Germany', 'Italy', 'Spain', 'United Kingdom')

## Average Points from Jury on final
points_from_to <- eurovision[X.semi...final == 'f' & Jury.or.Televoting == 'J', .(
  avg_votes = mean(Points)
), by = .(From.country, To.country)][From.country %in% top_five & To.country %in% top_five]

## Colors 
grid_colors <- c('#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600')
names(grid_colors) <- points_from_to[, unique(From.country)]

# Graph -------------------------------------------------------------------

## Convert data to matrix 
data_fd <- dcast.data.table(points_from_to, From.country ~ To.country, value.var = 'avg_votes')
data_m <- as.matrix(data_fd[, c(setdiff(colnames(data_fd), c('From.country'))), with = F])
rownames(data_m) <- data_fd$From.country
colnames(data_m)

## Chord Diagram
chordDiagram(data_m, transparency = 0.5, 
             big.gap = 10, 
             grid.col = grid_colors)
