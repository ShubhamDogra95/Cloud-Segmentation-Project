# Load necessary spatial packages
library(raster)
library(tidyverse)
library(sf)
library(rpart)
library(rpart.plot)
library(rasterVis)
library(mapview)
library(mapedit)
library(caret)
library(forcats)
library(terra)
library(rhdf5)
library(ggplot2)
library(sp)
library(dplyr)
library(rgdal)
library(gdalUtils)
library(htmltools)
library(tidyr)
library(plyr)
library(RColorBrewer)
library(grid)
library(prismatic)
library(patchwork)

# Set up the graphical parameters
myponga_img = "2024-04-08-00_00_2024-04-08-23_59_Sentinel-2_L1C_True_color"
pt_augusta_img = "2024-04-03-00_00_2024-04-03-23_59_Sentinel-2_L2A_True_color"
aldinga_img = "2024-04-03-00_00_2024-04-03-23_59_Sentinel-2_L2A_True_color"
whyalla_img = "2024-04-08-00_00_2024-04-08-23_59_Sentinel-2_L2A_True_color"

# Load the dataset
path_name = "whyalla"  # Subfolder name where the image is stored.
fname <- paste0("./", path_name, "/", whyalla_img,".tiff")

# Check existence image/filename in current work directory.
if (file.exists(fname)) {
  print("File Found, and Continue.")
} else {
  print("Filename Not found!, Check Current Folder Path and Filename!")
}



image_stack <- stack(fname)
print(image_stack)
names(image_stack)  # Change the column names if necessary

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ********************* 2. Ground Truth Training Set *********************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 2.1 Selecting the Cloud and Non Cloud polygons from satellite image
# -------------------------------------------------------------------
points <- viewRGB(image_stack, r = 1, g = 2, b = 3) %>% editMap()
points

# Save as clouds after first iteration
clouds <- points$finished$geometry %>% st_sf() %>% mutate(class = "clouds", id = 1)
clouds

# Repeat for non-clouds
points <- viewRGB(image_stack, r = 1, g = 2, b = 3) %>% editMap()
non_clouds <- points$finished$geometry %>% st_sf() %>% mutate(class = "non_clouds", id = 2)
non_clouds

# Combine cloud and non-cloud polygons to create final training set
training_points <- rbind(clouds, non_clouds)
class(training_points)
dim(training_points)

# Save training points to a shapefile
write_sf(training_points, paste0("./", path_name, "/shapes/", path_name, "_trainingPoints.shp"), driver = "ESRI shapefile")

# Load the saved shapefile
training_points <- st_read(paste0("./", path_name, "/shapes/", path_name, "_trainingPoints.shp"), quiet = TRUE)

# 2.2 Plotting the satellite image with the Cloud and Non Cloud polygons
# ----------------------------------------------------------------------
# Separate cloud and non-cloud polygons
shp_clouds <- subset(training_points, id == 1) %>% as('Spatial')
shp_non_clouds <- subset(training_points, id == 2) %>% as('Spatial')

# Plot the main satellite image
par(col.axis = "white", col.lab = "white", tck = 0)
plotRGB(image_stack, r = 1, g = 2, b = 3, stretch = "lin", axes = TRUE, main = paste0("SA Region - ", toupper(path_name), "\n RGB (Red, Green, Blue)"))
box(col = "white")

# Add polygons to the image
plot(shp_clouds, col = "#FF5805", add = TRUE)  # Add cloud polygons
plot(shp_non_clouds, col = "#72DC72", add = TRUE)  # Add non-cloud polygons

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ********** 3. Extracting Spectral Values *******************************************
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 3.1 Analyzing the Polygons for Training Points
# ----------------------------------------------
dim(training_points)
class(training_points)
class(training_points$geometry)  # Should be list-like
head(training_points)

my_rows <- nrow(training_points)
st_geometry(training_points[my_rows, ])[[1]]  # Access the last polygon
st_geometry(training_points[my_rows, ])[[1]][[1]]  # Access points in the polygon

# 3.2 Unpacking the Polygon Edge Values
# -------------------------------------
df_training_points = data.frame()

# Loop through the training points to extract edge values
for (i in 1:nrow(training_points)) {
  d1 = as.data.frame(i)
  z <- as.data.frame(st_geometry(training_points[i, ])[[1]][[1]])
  for (j in 1:nrow(z)) {
    d1['poly_no'] <- i
    d1['class'] <- training_points[[i, 1]]
    d1['id'] <- training_points[[i, 2]]
    d1['x'] <- z[[j, 1]]
    d1['y'] <- z[[j, 2]]
    df_training_points <- rbind(df_training_points, d1)
  }
}
df_training_points <- df_training_points[,-1]
df_training_points

# 3.3 Merging the Training Points and the Satellite Image Data
# ------------------------------------------------------------
# Convert training points to Spatial object
training_points2 <- as(training_points, 'Spatial')

# Extract raster values at training points
df <- raster::extract(image_stack, training_points2)
class(df)
head(df, 1)
sapply(df, dim)  # Display the list of polygons with their data points/pixels
count_polygons <- length(df)
df[count_polygons]  # Last polygon spectral reflectance values

# Combine polygon data frames into a single data frame
df2 <- data.frame()
for (i in seq_along(df)) {
  if (length(df[[i]]) > 0) {
    df1 <- as.data.frame(df[[i]])
    df1['polygon_no'] <- i  # Insertion index number list, i.e., polygon number
    df2 <- rbind(df2, df1)
  }
}

# Data frame with spectral values
dim(df2)
head(df2)
tail(df2)
colnames(df2)

# 3.4 Merging Class and ID with Spectral Values
# ---------------------------------------------
# Convert training points to data frame and add polygon numbers
training_points3 = as.data.frame(training_points2)
training_points3$polygon_no <- 1:nrow(training_points3)

# Merge the spectral values with class and ID information
df_merge = merge(df2, training_points3, by = 'polygon_no')
df_merge <- df_merge[, c(2:4, 1, 5:6)]
df_merge$image <- path_name

# 3.5 Consolidate a Train File from Multiple Images
# -------------------------------------------------
file_name = "train_file.rds"

# Check if the train file exists and append new data if it does
if (file.exists(file_name)) {
  print("File already exists, and new data added.")
  train_file <- readRDS(file_name)
  df.new <- rbind(df_merge, train_file)
  saveRDS(df.new, file_name)
} else {
  saveRDS(df_merge, file_name)
  print(paste0(file_name, " has now been saved for the first time!"))
}

# Summary of the train file
train_file <- readRDS(file_name)
train_file %>% group_by(image) %>% dplyr::summarise(counts = n())
train_file %>% group_by(image,class) %>%  dplyr::summarise(counts = n())
