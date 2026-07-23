#-------------------------------------------------------------------------------
# Introduction
#-------------------------------------------------------------------------------

# This script processes WorldView-3 satellite imagery (0.3 m spatial grain) acquired over 
# Kremetart area during February and July, and Mapungubwe National Park. 
# The workflow includes scaling reflectance values to 8-bit (0–255), combining 
# multi-seasonal imagery into a single 16-band raster, and resampling the imagery to multiple spatial 
# resolutions (0.9 m, 2.7 m, and 8.1 m). 
# The processed datasets are then exported and saved in the specified directory for further analysis.

# Load the required packages

library(tidyverse)
library(terra)

setwd("C:/Users/cp844/University of Exeter/TESSLab - Chafika Phiri Data/Objective 3 Manuscript/")

# WorldView-3 kremetart february imagery rescaling

#-------------- Loads image file for february

krem_feb_wv3_image <- terra::rast("Satellite Data/krem_july_image.tiff")

# get minimum and maximum values of the raster file

minum_values_feb_wv3 <- terra::global(krem_feb_wv3_image, "min", na.rm = TRUE)$min
maximum_values_feb_wv3 <- terra::global(krem_feb_wv3_image, "max", na.rm =TRUE)$max

# scaling our image to 8bit 0-255

krem_feb_wv3_8bit <- (krem_feb_wv3_image - minum_values_feb_wv3) / (maximum_values_feb_wv3 - minum_values_feb_wv3) * 255
krem_feb_wv3_8bit <- terra::clamp(round(krem_feb_wv3_8bit), 0, 255) # <-- ensures that the values in the image are between 0 to 255


# save

terra::writeRaster(krem_feb_wv3_8bit,
                   "Satellite Data/Data/kremetart_feb_wv3_8bit.tif",
                   datatype = "INT1U",
                   overwrite = TRUE)

# WorldView-3 kremetart july rescaling

#-------------- Load image file for july

krem_july_wv3_image <- terra::rast("Satellite Data/Kremetart_july_pansharped_WV3.tif")

# get minimum and maximum values of the raster file

minum_values_july_wv3 <- terra::global(krem_july_wv3_image, "min", na.rm = TRUE)$min
maximum_values_july_wv3 <- terra::global(krem_july_wv3_image, "max", na.rm =TRUE)$max

# scaling our image to 8bit 0-255

krem_july_wv3_8bit <- (krem_july_wv3_image - minum_values_july_wv3) / (maximum_values_july_wv3 - minum_values_july_wv3) * 255
krem_july_wv3_8bit <- terra::clamp(round(krem_july_wv3_8bit), 0, 255) # <-- ensures that the values in the image are between 0 to 255

# save

terra::writeRaster(krem_july_wv3_8bit,
                   "Satellite Data/Data/kremetart_july_wv3_8bit.tif",
                   datatype = "INT1U",
                   overwrite = TRUE)

# mapungubwe july WorldView-3 imagery rescaling

#-------------- load image file

maps_july_wv3_image <- terra::rast("Satellite Data/mapungubwe_rgb_july_pansharped_WV3.tif")

# get minimum and maximum values of the raster file

minum_values_july_wv3 <- terra::global(maps_july_wv3_image, "min", na.rm = TRUE)$min
maximum_values_july_wv3 <- terra::global(maps_july_wv3_image, "max", na.rm =TRUE)$max

# scaling our image to 8bit 0-255

july_wv3_8bit <- (maps_july_wv3_image - minum_values_july_wv3) / (maximum_values_july_wv3 - minum_values_july_wv3) * 255
july_wv3_8bit <- terra::clamp(round(july_wv3_8bit), 0, 255) # <-- ensures that the values in the image are between 0 to 255


# save

terra::writeRaster(july_wv3_8bit,
                   "Satellite Data/Data/maps_july_wv3_8bit.tif",
                   datatype = "INT1U",
                   overwrite = TRUE)

# combine the kremetart february and july imagery into a single 16 bands raster

# Load the multi-seasonal imagery

feb  <- rast("Satellite Data/Data/kremetart_feb_image_8bit.tif")
july <- rast("Satellite Data/Data/kremetart_july_image_8bit.tif")

# Check geometry matches (all three must be TRUE before stacking)

crs(feb) == crs(july)
ext(feb) == ext(july)
all(res(feb) == res(july))

# If any check failed, align July to Feb's grid:
# july <- resample(july, feb, method = "bilinear")

# Stack into 16 bands

combined <- c(feb, july)
names(combined) <- c(paste0("feb_b", 1:8), paste0("jul_b", 1:8))

# Save

writeRaster(
  combined,
  "Satellite Data/Data/kremetart_feb_july_16band.tif",
  overwrite = TRUE,
  datatype  = "INT1U",
  gdal      = c("COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER")
)

# Resample the images to multiple spatial resolutions (0.9 m, 2.7 m, and 8.1 m)

# load and read image

Kremetart_image_feb <- rast("Satellite Data/Data/kremetart_feb_image_8bit.tif")
Kremetart_image_feb_july <- rast("Satellite Data/Data/kremetart_feb_july_16band.tif")
#nlyr(Kremetart_image_feb_july)
#maps_image_july <- rast("Satellite Data/Data/mapungubwe_july_image_8bit.tif")

#res(Kremetart_image_july)
#res(maps_image_july)

# re-scale images from 0.3m to 0.9, 2.7 and 8.1m

Kremetart_0.9m <- aggregate(Kremetart_image_feb, fact = 3, fun = "mean")
Kremetart_2.7m <- aggregate(Kremetart_0.9m,        fact = 3, fun = "mean")
Kremetart_8.1m <- aggregate(Kremetart_2.7m,        fact = 3, fun = "mean")

maps_0.9m <- aggregate(Kremetart_image_feb_july, fact = 3, fun = "mean")
maps_2.7m <- aggregate(maps_0.9m,       fact = 3, fun = "mean")
maps_8.1m <- aggregate(maps_2.7m,       fact = 3, fun = "mean")

write_opts <- list(
  datatype = "INT1U",
  gdal     = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"),
  overwrite = TRUE
)

# save the raster files

do.call(writeRaster, c(list(round(Kremetart_0.9m), "Satellite Data/Data/kremetart_feb_0.9m.tif"), write_opts))
do.call(writeRaster, c(list(round(Kremetart_2.7m), "Satellite Data/Data/kremetart_feb_2.7m.tif"), write_opts))
do.call(writeRaster, c(list(round(Kremetart_8.1m), "Satellite Data/Data/kremetart_feb_8.1m.tif"), write_opts))
do.call(writeRaster, c(list(round(maps_0.9m),      "Satellite Data/Data/mapungubwe_feb_july_0.9m.tif"), write_opts))
do.call(writeRaster, c(list(round(maps_2.7m),      "Satellite Data/Data/mapungubwe_feb_july_2.7m.tif"), write_opts))
do.call(writeRaster, c(list(round(maps_8.1m),      "Satellite Data/Data/mapungubwe_feb_july_8.1m.tif"), write_opts))

# visualise

krem_8.1m <- rast("Satellite Data/Data/kremetart_july_8.1m.tif")

plotRGB(krem_8.1m, r=5, g=3, b=2, stretch="lin")
