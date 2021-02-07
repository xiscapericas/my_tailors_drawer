## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(ggplot2)

# Data --------------------------------------------------------------------
cacao_data <- fread('flavors_of_cacao.csv') # load 
setnames(cacao_data, make.names(colnames(cacao_data))) # Rename 
cacao_per_loc <- cacao_data[Rating > 3.5, .N, by = Company.Location][order(-N)]
cacao_per_loc[, Company.Location := as.factor(Company.Location)]
my_category <- 'Spain'

# Graph -------------------------------------------------------------------
p1 <- ggplot(cacao_per_loc, aes(x = reorder(Company.Location, N), y = N)) + 
  geom_segment(aes(x = reorder(Company.Location, N), xend = Company.Location, y = 0, yend = N), 
               color=ifelse(cacao_per_loc$Company.Location %in% my_category, 'brown', 'gray'), 
               size=ifelse(cacao_per_loc$Company.Location %in% my_category, 3, 1.5)) +
  geom_point(color=ifelse(cacao_per_loc$Company.Location %in% my_category, 'brown', 'gray'),
             size=ifelse(cacao_per_loc$Company.Location %in% my_category, 5, 3)) + 
  theme_bw() + 
  coord_flip() + 
  xlab('Company location country') + 
  ylab('# of cacaos') +  
  theme(axis.title.x = element_text(hjust = 0.5), 
        axis.title.y = element_text(hjust = 0.5), 
        plot.title = element_text(hjust = 0.5, size = 20)) + 
  labs(title = 'Chocolate bar with rating > 3.5 per country', caption = 'By Xisca Pe') + 
  annotate("text", x= my_category, 
           y=cacao_per_loc[Company.Location == my_category, N] * 2, 
           label=cacao_data[Company.Location == my_category & Rating > 3.5, unique(Company...Maker.if.known.)], 
           color="brown", size= 5 , angle= 0, fontface="bold", hjust= 0)


p1
