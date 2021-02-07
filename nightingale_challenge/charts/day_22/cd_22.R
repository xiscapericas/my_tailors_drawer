## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(rvest)
library(data.table)
library(tm) # data processing 
library(colormap)
library(wordcloud)

# Get Data ----------------------------------------------------------------
url_letras <- "https://www.letras.com/la-casa-azul/"
tmp <- read_html(url_letras)
tmp_div <- html_nodes(tmp, "a")

## Descargar lista de canciones
songs_path <- rbindlist(lapply(tmp_div, function(tmp_div_ind) {
  if (length(html_attrs(tmp_div_ind)) == 2) {
    data.table(
      title = html_attr(tmp_div_ind, 'title'), 
      url = html_attr(tmp_div_ind, 'href')
    )
  }
}))[!is.na(title)][grepl('la-casa-azul', url)]

## Descargar canciones 
letras <- rbindlist(lapply(1:nrow(songs_path), function(ind) {

  ## Importar Cancion 
  my_whole_url <- paste0('https://www.letras.com', songs_path[ind, url])
  letra_pg <- read_html(my_whole_url)
  article <- html_nodes(letra_pg, 'article')
  parrafos_c <- html_nodes(article, 'p')
  parrafos_t <- html_text(parrafos_c)
  
  ## Procesar Datos 
  parrafos_t <- tolower(gsub('([a-z])()([A-Z])','\\1 \\3', parrafos_t)) ## Clean spaces & tolower 
  parrafos_t <- gsub('[?|¿|!|¡]', '', parrafos_t) #Quitar signos interrogacion
  parrafos_t_l_n <- removeWords(parrafos_t, words = stopwords("spanish"))
  parrafos_t_l_n <- removePunctuation(parrafos_t_l_n)
  parrafos_t_l_n <- stripWhitespace(parrafos_t_l_n)
  
  ## Crear output 
  data.table(palabras = unlist(strsplit(parrafos_t_l_n, ' ')))[, title := songs_path[ind, title]]
  
}))

# Graph -------------------------------------------------------------------
## Word Cloud
pal_freq <- letras[palabras != '', .(freq = .N), by = palabras][order(-freq)]
pal_freq_t <- letras[, .N, by = .(palabras, title)][, .(best_song = title[which.max(N)]), by = palabras][pal_freq, on = 'palabras']
max_words <- 100

## Colors 
mycolor_list <- colormap(colormap=colormaps$freesurface_blue, nshades=pal_freq_t[order(-freq)][1:max_words, uniqueN(best_song)])
mycolor_list <- sample(mycolor_list, length(mycolor_list))
my_color <- mycolor_list[as.numeric(as.factor(pal_freq_t[order(-freq)][1:max_words]$best_song))]

## Layout 
layout(matrix(c(1, 2), nrow=2), heights=c(1, 4))
par(mar=rep(0, 4))
plot.new()
text(x=0.5, y=0.5, "La Casa Azul WordCloud", family = 'Arial', cex = 2, col = my_color[1])
## WordCloude
wordcloud(
  words = pal_freq_t$palabra, 
  freq = pal_freq_t$freq, 
  max.words = max_words, 
  random.order = F, 
  colors=my_color
)


