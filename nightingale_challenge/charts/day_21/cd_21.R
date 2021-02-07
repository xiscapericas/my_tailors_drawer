## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table) 
library(ggplot2)

# Data --------------------------------------------------------------------
frase_gato <- fread('frase_gato.csv')
circle <- data.table(x = 0.5, y = 3.5)
box <- data.table(x = seq(0,3,1))[, `:=` (ymin = 0, ymax = 0.5)]

# Graph -------------------------------------------------------------------
ggplot(box) + 
  geom_ribbon(aes(ymin=ymin, ymax=ymax, x=x), fill = "ivory3") + 
  geom_text(data = frase_gato, aes( x=x, y=y, label=label, angle = angle), size = 8,  
                     fontface="bold", color = 'orange2') + xlim(0,3) + ylim(0,4) + 
  geom_point(aes(x=x, y=y), data=circle, size=30, color="deepskyblue2")  + 
  geom_hline(yintercept = 0.5, color = 'ivory3', size = 5) + 
  theme(panel.background = element_rect(color = 'black', fill = 'black'), 
        panel.grid = element_blank(), 
        panel.border = element_blank(), 
        axis.title = element_blank(), 
        axis.text = element_blank(), 
        axis.ticks = element_blank()) 


  
