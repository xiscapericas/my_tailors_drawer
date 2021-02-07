## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table)
library(ggplot2)
library(gganimate)
library(viridis)

# Get Data ----------------------------------------------------------------
r_packages <- fread('r_packages.csv', encoding = 'UTF-8') #https://cran.r-project.org/web/packages/available_packages_by_date.html
r_packages[, date := as.Date(Date, '%m/%d/%y')]
r_packages[, title := gsub("[^[:alnum:][:blank:]?&/\\-]", "", Title)]
r_packages[,description := tolower((title))]

## Aggregate by class 
packages_clas <- fread("packages_clas.csv")
sapply(1:nrow(packages_clas), function(ind) {
  print(packages_clas[ind, class])
  sapply(unlist(strsplit(packages_clas[ind, words], ',')), function(my_word) {
    print(my_word)
    r_packages[grepl(my_word, description), package_group := packages_clas[ind, class]]
  })
})

## Cumulative
r_packages[, ind := 1]
cumnum_packages <- r_packages[!is.na(package_group)][order(date), .(
  date, num_packages = sum(ind)
), by = .(date, package_group)][, .(cum_num = cumsum(num_packages), date), by = .(package_group)]



# Graph -------------------------------------------------------------------
p <- ggplot(cumnum_packages, aes(x = package_group, y = cum_num, fill = package_group)) + geom_bar(stat = 'identity', position = 'dodge') + 
      theme_bw() + coord_flip() + 
      transition_time(date) + 
      xlab('Tipo') + ylab('Package #') + 
      scale_fill_viridis(discrete = T, option = 'B') +
      theme(plot.title = element_text(hjust = 0.5, size = 20, family = 'Arial', face = 'bold'), 
            plot.caption = element_text(family = 'Arial')) + 
      labs(title = 'R-Packages Per Date: {frame_time}', 
           caption = 'By Xisca Pe', fill = 'Tipo')

## Animate & save 
animate(p, renderer = gifski_renderer(), duration = 10)
anim_save("output.gif")

