## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(data.table) 
library(leaflet)
library(geojsonio)

# Get Data ----------------------------------------------------------------
## GeoJson 
### Source: https://datahub.io/core/geo-countries
world <- geojsonio::geojson_read("countries.geojson", what = "sp")

## National Depth 
### Fuente: https://worldpopulationreview.com/countries/countries-by-national-debt/
national_debt <- fread('data.csv')
national_debt[, debt_gdp := value / 1000]

## Add info 
countries_world_gdp <- national_debt[, .(
  country, debt_gdp)][data.table(
    country =  world$ADMIN),  on = 'country'][
      is.na(debt_gdp), debt_gdp := 0]
world$debt <- countries_world_gdp[, debt_gdp]


## Colors 
countries_world_gdp[, cut_gdp := cut(debt_gdp, unique(quantile(debt_gdp, seq(0, 1, length=10))), include.lowest=T)]
bins <- c(countries_world_gdp[, min(debt_gdp), by = cut_gdp][order(V1), V1], Inf)
pal <- colorBin("OrRd", domain = world$debt, bins = bins)

## Labels 
labels <- sprintf(
  "<strong>%s</strong><br/>%g%% national dept to GDP ratio",
  world$ADMIN, world$debt
) %>% lapply(htmltools::HTML)


# Graph -------------------------------------------------------------------
m <- leaflet(world)
m %>% addPolygons(
  fillColor = ~pal(debt),
  weight = 2,
  opacity = 1,
  color = "white",
  dashArray = "3",
  fillOpacity = 0.7,
  highlight = highlightOptions(
    weight = 5,
    color = "#666",
    dashArray = "",
    fillOpacity = 0.7,
    bringToFront = TRUE),
    label = labels,
  labelOptions = labelOptions(
    style = list("font-weight" = "normal", padding = "3px 8px"),
    textsize = "15px",
    direction = "auto")) %>% 
  addLegend(pal = pal, values = ~debt, opacity = 0.7, title = NULL,
                                        position = "bottomright")
