# Title     : Node embeddings
# Objective : Visualize the resulted node embeddings
# Created by: Yuan Liao
# Created on: 2025-03-08

library(umap)
library(ggplot2)
library(ggpubr)
library(scico)
library(ggsci)
library(arrow)
library(scales)
library(ggExtra)
library(ggthemes)
library(shadowtext)
library(ggmap)
library(ggspatial)
library(boot)
library(sf)
options(scipen=10000)

# Example dataframe
set_id <- 10
df.io <- as.data.frame(read_parquet(paste0('dbs/embeddings/baseline_set', set_id, '_individual_original.parquet')))
df.i <- as.data.frame(read_parquet(paste0('dbs/embeddings/baseline_set', set_id, '_individual_umap.parquet')))
df.h <- as.data.frame(read_parquet(paste0('dbs/embeddings/baseline_set', set_id, '_hexagon_umap.parquet')))
df.hc <- as.data.frame(read_parquet(paste0('dbs/embeddings/baseline_set', set_id, '_hexagon_umap_clusters.parquet')))
# Visualize individuals/umap-based results----
g1 <- ggplot(df.i, aes(x = umap_1, y = umap_2, color = as.factor(deso_r))) +
  geom_point(size = 0.2, alpha=1) +
  scale_color_npg(name='DeSO') +
  theme_void()

g2 <- ggplot(df.i, aes(x = umap_1, y = umap_2, color = as.factor(group))) +
  geom_point(size = 0.2, alpha=1) +
  scale_color_npg(name='Group') +
  theme_void()

G1 <- ggarrange(g1, g2, ncol = 2, nrow = 1, labels = c('(a)', '(b)'))
ggsave(filename = paste0("figures/baseline_individual_umap_", set_id, ".png"), plot=G1,
       width = 15, height = 6, unit = "in", dpi = 300)

g3 <- ggplot(df.i, aes(x = longitude, y = latitude, color = as.factor(group_e))) +
  geom_point(size = 0.1, alpha=0.7) +
  theme_void() +
  labs(color='Group_e')

g4 <- ggplot(df.i, aes(x = longitude, y = latitude, color = as.factor(group))) +
  geom_point(size = 0.1, alpha=0.7) +
  scale_color_npg(name='Group') +
  theme_void()

G2 <- ggarrange(g3, g4, ncol = 2, nrow = 1, labels = c('(a)', '(b)'))
ggsave(filename = paste0("figures/baseline_individual_umap_clusters_", set_id, ".png"), plot=G2,
       width = 18, height = 6, unit = "in", dpi = 300)

# Visualize individuals/clusters based on original embeddings----
g3 <- ggplot(df.io, aes(x = longitude, y = latitude, color = as.factor(group_e))) +
  geom_point(size = 0.1, alpha=0.7) +
  theme_void() +
  labs(color='Group_e')

g4 <- ggplot(df.io, aes(x = longitude, y = latitude, color = as.factor(group))) +
  geom_point(size = 0.1, alpha=0.7) +
  scale_color_npg(name='Group') +
  theme_void()

G2 <- ggarrange(g3, g4, ncol = 2, nrow = 1, labels = c('(a)', '(b)'))
ggsave(filename = paste0("figures/baseline_individual_original_clusters_", set_id, ".png"), plot=G2,
       width = 18, height = 6, unit = "in", dpi = 300)

# Visualize hexagons ----
g5 <- ggplot(df.h, aes(x = umap_1, y = umap_2, color = as.factor(group))) +
  geom_point(size = 0.2, alpha=1) +
  scale_color_npg(name='Group') +
  theme_void()

ggsave(filename = paste0("figures/baseline_hexagon_umap_", set_id, ".png"), plot=g5,
       width = 6, height = 6, unit = "in", dpi = 300)

# Visualize hexagons / clusters ----
# Basemaps ----
ggmap::register_stadiamaps(key='1ffbd641-ab9c-448b-8f83-95630d3c7ee3')
z.level <- 11
# Stockholm
bbox <- c(17.6799476147,59.1174841345,18.4572303295,59.475092515)

names(bbox) <- c("left", "bottom", "right", "top")
stockholm_basemap <- get_stadiamap(bbox, maptype="stamen_toner_lines", zoom = z.level)
gdf <- st_transform(st_read(paste0('dbs/embeddings/baseline_set', set_id, '_hexagon_umap_clusters.shp')), 4326)

g6 <- ggmap(stockholm_basemap) +
  geom_sf(data = gdf, aes(fill=as.factor(group_e)),
          color = 'white', size=0.05, alpha=0.3, show.legend = T, inherit.aes = FALSE) +
  labs(title = 'Group by UMAP embeddings') +
  # scale_fill_locuszoom(name='Public transit group') +
  annotation_scale(location = "bl", width_hint = 0.3, text_cex = 0.5) +  # Add a scale bar
  annotation_north_arrow(
    location = "tr", which_north = "true",
    style = north_arrow_fancy_orienteering(text_size = 6),
    height = unit(0.8, "cm"),  # Adjust arrow height
    width = unit(0.8, "cm")    # Adjust arrow width
  ) +
  theme_void() +
  theme(plot.margin = margin(0.1,0.1,0.1,0, "cm"),
        legend.position = 'top',
        plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(nrow = 1))
ggsave(filename = paste0("figures/baseline_hexagon_umap_clusters", set_id, ".png"), plot=g6,
       width = 6, height = 6, unit = "in", dpi = 300)