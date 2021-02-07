## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(openxlsx)
library(data.table)
library(ggplot2)

# Data --------------------------------------------------------------------
euro_data <- as.data.table(read.xlsx('~/Desktop/graph_challenge/day_9/eurovision_song_contest_1975_2019.xlsx'))
setnames(euro_data, '(semi-).final', 'semi_final')

## Process 
spain_top_friends <- euro_data[semi_final == 'f' & To.country == 'Spain' & Jury.or.Televoting == 'J'][,sum(Points), by= From.country][order(-V1)][1:10, From.country]
spain_points <- euro_data[semi_final == 'f' & To.country == 'Spain' & Jury.or.Televoting == 'J'][
  From.country %in% spain_top_friends][,.(
    year = Year, from_country = From.country, points = Points
  )]
# Adding missing years 
all_year <- spain_points[, unique(year)]
missing <- rbindlist(lapply(spain_points[, unique(from_country)], function(country) {
  if (spain_points[from_country == country, .N] != spain_points[, uniqueN(year)]) {
      data.table(
        year = all_year[!all_year %in% spain_points[from_country == country, year]]
      )[, `:=` (
        points = 0, 
        from_country = country
      )]
  } 
}))
spain_points_w_m <- rbind(spain_points, missing)

# Graph -------------------------------------------------------------------
p1 <- ggplot(spain_points_w_m, aes(x=year, y=points, fill=from_country)) + 
  geom_area(alpha = 0.7, size=0.2, colour="black") +
  scale_fill_brewer(palette="Paired") +
  theme_bw() + 
  xlab('Year') + ylab('Points') + 
  theme(axis.title.x = element_text(hjust = 0.5), 
        axis.title.y = element_text(hjust = 0.5), 
        plot.title = element_text(hjust = 0.5, size = 20)) + 
  labs(title = 'Eurovision points to Spain', caption = 'By Xisca Pe') + 
  guides(fill=guide_legend(title="Voting country"))

