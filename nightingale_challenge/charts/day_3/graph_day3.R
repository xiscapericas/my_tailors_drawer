## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(ggplot2)
library(plotly)
library(viridis) #https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html


# Auxiliar ----------------------------------------------------------------
comprss <- function(tx) { 
  div <- findInterval(as.numeric(gsub("\\,", "", tx)), c(0, 1e3, 1e6, 1e9, 1e12))  # modify this if negative numbers are possible
  paste(round( as.numeric(gsub("\\,","",tx))/10^(3*(div-1)), 2), c("","K","M","B","T")[div] )
}

# Get Data ----------------------------------------------------------------
google_play <- fread('googleplaystore.csv')
## Format Size 
google_play[, size_num := as.numeric(gsub('[A-Za-z]', '', Size))]
## Aggregate
more_reviews_app <-google_play[order(-Reviews)][!is.na(size_num)][!duplicated(App)][sample(25)]
more_reviews_app[, text := paste("App: ", App, 
                             "\nReviews Avg: ", comprss(Reviews), 
                             "\nSize Avg: ", size_num, 
                             "\nRating Avg: ", Rating, sep="")]


# Graph -------------------------------------------------------------------
### Source: https://www.r-graph-gallery.com/bubble_chart_interactive_ggplotly.html
p1 <- ggplot(more_reviews_app, aes(x = Reviews, y = size_num, size = Rating, color = App, text = text)) + 
  geom_point(alpha = 0.7) + 
  scale_color_viridis(discrete = T, guide = F, option = 'A') + 
  theme(legend.position = 'none', plot.title = element_text(hjust = 0.5, size = 20)) + 
  scale_size_continuous(range = c(5,20)) + 
  theme_bw() + xlab('Reviews') + ylab('Size') + ggtitle('Google Play Apps')

pp <- ggplotly(p1, tooltip="text")
pp


