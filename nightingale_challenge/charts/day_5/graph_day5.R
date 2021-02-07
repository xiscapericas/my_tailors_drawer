## 30DGraphChallenge

# Packages ----------------------------------------------------------------
library(rjson)
library(data.table)
library(igraph)
library(ggraph)
library(colormap)


# Data --------------------------------------------------------------------
lod_cloud_data_url <- "https://lod-cloud.net/lod-data.json"
lod_cloud_data <- rjson::fromJSON(file=lod_cloud_data_url)
## Format 
result <- rbindlist(lapply(names(lod_cloud_data), function(x) {
  print(x)
  if (length(lod_cloud_data[[x]]$links) > 0) {
    cbind(
      data.table(
        element = x, 
        domain = lod_cloud_data[[x]]$domain
      ), 
      setnames(as.data.table(matrix(unlist(lod_cloud_data[[x]]$links), 
                                    nrow=length(lod_cloud_data[[x]]$links), byrow=T), 
                             stringsAsFactors=FALSE), unique(names(unlist(lod_cloud_data[[x]]$links))))
    )
  }
}), fill = T)


# Graph -------------------------------------------------------------------
my_elements <- sample(result[domain != ''][, unique(element)], 8)
data_for_graph <- result[element %in% my_elements, .(element, target, domain)][order(element, domain)]

## Elements dt 
elements_dt <- data.table(element = unique(c(data_for_graph$element, data_for_graph$target)))
### With domain info 
element_d <- result[, .(element, domain)][elements_dt, on = 'element']
element_d <- element_d[!duplicated(element_d)]

## Graph 
graph <- graph_from_data_frame(d = data_for_graph, vertices = element_d, directed = F)

# Color 
mycolor <- colormap(colormap=colormaps$spring, nshades=max(element_d[, uniqueN(domain)]))
mycolor <- sample(mycolor, length(mycolor))

# Make the graph
ggraph(graph, layout="linear") + 
  geom_edge_arc(edge_colour="black", edge_alpha=0.2, edge_width=0.3, fold=TRUE) +
  geom_node_point(aes(color=as.factor(domain)), alpha=0.5, size = 3) +
  scale_color_manual(values=mycolor) + 
  geom_node_text(aes(label=name), angle=65, hjust=1, nudge_y = -1.1, size=2.3) +
  theme(
    legend.position="bottom", panel.background = element_rect(fill = 'white'),
    plot.title = element_text(hjust = 0.5, size = 20), 
    plot.margin=unit(c(0,0,0.4,0), "null"),
    panel.spacing=unit(c(0,0,3.4,0), "null")
  ) + 
  expand_limits(x = c(-1.2, 1.2), y = c(-5.6, 1.2)) + 
  guides(color=guide_legend(title="Domain")) + 
  ggtitle('Linked Open Data Arc Graph')
