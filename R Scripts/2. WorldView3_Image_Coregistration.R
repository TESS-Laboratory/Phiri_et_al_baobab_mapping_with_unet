# ------------------------------------------------------------------------------
# Introduction
# ------------------------------------------------------------------------------

# This script performs image co-registration of two WorldView-3 multispectral 
# satellite images acquired over Kremetart, South Africa, in February and July. 
# The workflow includes loading the images, handling missing data, 
# automatically estimating the optimal spatial alignment, and applying a 
# manual fine-tuning shift to accurately align the February image to the 
# July reference image.

# Load the required packages

library(terra)
library(RStoolbox)

# set the working directory

setwd("C:/Users/cp844/University of Exeter/TESSLab - Chafika Phiri Data/Objective 3 Manuscript/")

# Load images using

# This is the "target" image (February) that will be moved/shifted to align with the reference

krem_target <- terra::rast("Satellite Data/Kremetart_WV3/krem_ms_feb_mosaic.tiff")

# This is the "reference" image (July) that remains stationary - the target will be aligned to match this

krem_ref <- terra::rast("Satellite Data/Kremetart_WV3/krem_ms_july_mosaic.tiff")

# Replace NA (Not Available/missing) values with 0
# This is necessary because co-registration algorithms cannot handle NA values

krem_target[is.na(krem_target)] <- 0
krem_ref[is.na(krem_ref)] <- 0

# Print the images to view their properties (dimensions, resolution, bands, etc.)
#krem_target
#krem_ref

# Co-register the images to find the optimal alignment between images

coreg_result_krem_feb <- RStoolbox::coregisterImages(
  img = krem_target,           
  ref = krem_ref,             
  shift = 10,                  
  shiftInc = 1,                
  nSamples = 10000,           
  reportStats = TRUE,          
  verbose = TRUE,              
  nBins = 100                  
)

# Specify the exact pixel shifts needed based on the co-registered results

dx_pixels <- 4   
dy_pixels <- 1  

# Convert pixel shifts to map units (meters)
# WorldView-3 multispectral resolution is 1.2 meters per pixel

dx_m <- dx_pixels * 1.2  
dy_m <- dy_pixels * 1.2  

# Apply the shift to create the aligned image

krem_feb_aligned <- shift(krem_target, dx = dx_m, dy = dy_m)

# Visualise (RGB)

plotRGB(krem_ref, r=5, g=3, b=2, stretch="lin")
plotRGB(krem_feb_aligned, r=5, g=3, b=2, stretch="lin", add=TRUE)

# Save output image

writeRaster(coreg_result_maps_feb, 
            "Satellite Data/Kremetart_WV3/aligned_krem_february_image.tif", 
            overwrite=TRUE)
