## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table) 
library(leaflet)
library(manipulateWidget)

# Get Data ----------------------------------------------------------------
puntos_mapa <- rbindlist(lapply(list.files(getwd()), function(file) {
  fread(file)[, file_n := file]
}))
puntos_mapa[, categoria := gsub('([A-Z][0-9]{3}_)(.*)(.csv)','\\2',file_n)]
pt_sub <- puntos_mapa[, .(equipment = EQUIPAMENT, latitude = LATITUD, longitud = LONGITUD, categoria)][!duplicated(equipment)]
pt_sub <- data.table(categoria = pt_sub[, unique(categoria)], 
           type = c('biblioteca', 'musica', 'infantil', 'punto_verde'))[pt_sub, on = 'categoria']
pt_sub[, type := as.factor(icon)]

# Graph -------------------------------------------------------------------

## ICONS 
libraryIcon <- makeIcon(
  iconUrl = "book_library_icon.jpg",
  iconWidth = 25, iconHeight = 25
)

musicIcon <- makeIcon(
  iconUrl = "computer-icons-musical-note-music-download-musical-theatre-musical-note.jpg",
  iconWidth = 25, iconHeight = 25
)

ptoVerdeIcon <- makeIcon(
  iconUrl = "png-transparent-recycling-symbol-logo-decal-plastic-recycling-symbol-angle-leaf-label.png",
  iconWidth = 25, iconHeight = 25
)

infantilIcon <- makeIcon(
  iconUrl = "children-icon-png-11552333605ypbqejojqg.png",
  iconWidth = 25, iconHeight = 25
)

bcnIcons <- iconList(
  biblioteca = libraryIcon, 
  musica = musicIcon, 
  punto_verde = ptoVerdeIcon, 
  infantil = infantilIcon
)

## Create maps 
maps_l <- lapply(pt_sub[, unique(categoria)], function(categoria_i) {
  my_icon <- bcnIcons[[pt_sub[categoria == categoria_i, unique(type)]]]
  leaflet(data = pt_sub[categoria == categoria_i]) %>% 
    addTiles() %>% 
    addMarkers(m, lng=~longitud, lat=~latitude, popup=~equipment, icon = my_icon) %>% 
    addProviderTiles(providers$CartoDB.Positron) %>%
    addProviderTiles(providers$Stamen.TonerLines) %>%
    addProviderTiles(providers$Stamen.TonerLabels)
})

## Combine 
manipulateWidget::combineWidgets(maps_l[[1]], maps_l[[2]], maps_l[[3]], maps_l[[4]], 
                                 title = 'Puntos de interes BCN')


