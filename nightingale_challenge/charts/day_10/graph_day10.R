## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(openxlsx)
library(data.table)
library(ggplot2)
library(gridExtra)

# Data --------------------------------------------------------------------
source('../day_9/graph_day9.R') ## Code day 8
own_paletta <- c('#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600', '#51D231', '#A1DC15') 

# Graph -------------------------------------------------------------------
main_p <- ggplot(spain_points_w_m, aes(x=year, y=points, fill=from_country)) + 
  geom_area(alpha = 0.7, size=0.2, colour="black") +
  theme_bw() + 
  xlab('Year') + ylab('Points') + 
  theme(axis.title.x = element_text(hjust = 0.5), 
        axis.title.y = element_text(hjust = 0.5), 
        plot.title = element_text(hjust = 0.5, size = 20)) + 
  guides(fill=guide_legend(title="Voting country"))

# Palettas RColor Brewer 
paletas <- c('Set3', 'Paired', 'Spectral', 'PiYG')
plots_l <- lapply(paletas, function(paleta) {
  main_p + scale_fill_brewer(palette = paleta) + ggtitle(paleta)
})
# Paletas Viridis 
options <- c('A', 'D')
plots_v <- lapply(options, function(option_i) {
  main_p + scale_fill_viridis(option = option_i, discrete = T) + ggtitle(paste0('Viridis ', option_i))
})

# Others 
pg <- main_p + scale_fill_grey(start = 0.8, end = 0.2) + ggtitle('Gray Scale') 
p_own <- main_p + scale_fill_manual(values = own_paletta) + ggtitle('Own')
plots_r <- list(pg, p_own)

# Final list 
plots_f <- c(plots_l, plots_v, plots_r)

# Grid 
n <- length(plots_f)
nCol <- floor(sqrt(n))
do.call("grid.arrange", c(plots_f, ncol=nCol))

