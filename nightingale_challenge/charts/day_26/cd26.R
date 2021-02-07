## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table) 
library(ggplot2)

# Get Data ----------------------------------------------------------------
cats <- fread('cats.csv')[, .(id, type, age, gender, size, coat, breed)] ## https://www.kaggle.com/ma7555/cat-breeds-dataset
most_freq_breeds <- cats[, .N, by = breed][order(-N)][1:5, breed]
## Compute 
cats_c <- cats[breed %in% most_freq_breeds, .N, by = .(breed, age)]

## X Axis 
age_data <- cats_c[, .(sample = sum(N)),  by = age][, per := round(sample / sum(sample) * 100)][order(-per)]
xpos <- data.table(
  xmax = cumsum(age_data[order(per)]$per[1:.N-1]), 
  age = age_data$age
)[, xmin := c(0, xmax[1:.N-1])][xmax == 101, xmax := 100]

## Y Axis 
cats_d <- cats[breed %in% most_freq_breeds, .(samples_per_s_a = .N), by = .(breed, age)][
  age_data, on = 'age'][, per_per_age := round(samples_per_s_a / sample * 100)]

ypos <- rbindlist(lapply(cats_d[, unique(age)], function(age_i) {
  data.table(
    ymax = cumsum(cats_d[age == age_i][order(breed)]$per_per_age[1:.N]), 
    breed = cats_d[age == age_i][order(breed)][, breed], 
    text = cats_d[age == age_i][order(breed)][, per_per_age]
  )[, ymin := c(0, ymax[1:.N-1])][ymax >= 99, ymax := 100][, age := age_i]
}))

## Xpos + Ypos 
yx_df <- xpos[ypos, on = 'age']

## Add text position 
yx_df[, `:=` (xtext = (xmax / 2) + (xmin / 2), 
              ytext = (ymax / 2) + (ymin / 2))]

# Graph -------------------------------------------------------------------
ggplot(yx_df, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = breed)) + 
  geom_rect(colour = I("grey")) + 
  geom_text(aes(x = xtext, y = ytext, label = paste0(text, '%')), color = 'black', size = 5) + 
  theme_minimal() + 
  scale_x_continuous(breaks = yx_df[, unique(xmax), by = age][, V1], labels = yx_df[, unique(xmax), by = age][, age]) + 
  xlab('Age') + ylab('Freq') + 
  scale_fill_brewer(palette = 'Set2') + 
  theme(plot.title = element_text(hjust = 0.5, size = 20, family = 'Arial', face = 'bold'), 
        plot.caption = element_text(family = 'Arial')) + 
  labs(title = 'Cats Breed per Age', 
       caption = 'By Xisca Pe')
