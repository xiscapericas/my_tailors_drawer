## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(sunburstR)

# Get Data ----------------------------------------------------------------
anime <- fread('../day_19/dataanime.csv')[!duplicated(Title)]
anime_sub <- anime[,.(title = Title, type = Type, genres = Genres, start_date = as.Date(`Start airing`))][, month_sd := month(start_date)]
## Filter date 
anime_sub <- anime_sub[start_date >= as.Date('2018-01-01') & start_date < as.Date('2019-01-01')]
anime_sub[, quarter := ifelse(month_sd >= 1 & month_sd < 4, 'Q1', 
                   ifelse(month_sd >= 4 & month_sd < 7, 'Q2', 
                          ifelse(month_sd >= 7 & month_sd < 10,'Q3',
                                 'Q4')))]
anime_sub <- anime_sub[, .(primary_genre = unlist(strsplit(genres, ','))[1]), by = title][anime_sub, on = 'title']
count_anime_per_group <- anime_sub[!is.na(quarter), .(animes_count = .N), by = .(quarter, type, primary_genre)]
count_anime_per_group[, path := paste0(quarter, '-', type, '-', primary_genre)]

# Graph -------------------------------------------------------------------
sunburst(data.frame(xtabs(animes_count ~ path, count_anime_per_group)))

