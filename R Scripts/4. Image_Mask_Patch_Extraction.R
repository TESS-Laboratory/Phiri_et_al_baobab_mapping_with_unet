#-------------------------------------------------------------------------------
# Introduction
#-------------------------------------------------------------------------------

# This script extracts image patches and corresponding binary masks for U-Net 
# model training. Patch extraction is based on the centroids of polygons, which 
# are used as the centre points of the image patches. The workflow also handles 
# cases where patches extend beyond image boundaries by automatically shifting 
# them inward to ensure complete extraction.

# Load the required packages

library(tidyverse)
library(terra)
library(sf)
library(knitr)
library(tools)
library(geodl)
library(fs)
library(purrr)

# Load images and polygons

image_path    <- "My_Dataset/mapungubwe_feb_july_0.9m.tif"
polygons_path <- "My_Dataset/Wet_Kremetart_tree_polygons.gpkg"

img_dir  <- "Kremetart_feb_polygons/new_feb_july_256_8bands_0.9m/images"
mask_dir <- "Kremetart_feb_polygons/new_feb_july_256_8bands_0.9m/masks"

dir.create(img_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(mask_dir, recursive = TRUE, showWarnings = FALSE)

# Load and align

Kremetart_image         <- rast(image_path)
Kremetart_tree_polygons <- vect(polygons_path)

if (!crs(Kremetart_tree_polygons) == crs(Kremetart_image)) {
  Kremetart_tree_polygons <- project(Kremetart_tree_polygons, crs(Kremetart_image))
}

Kremetart_tree_polygons <- crop(Kremetart_tree_polygons, ext(Kremetart_image))

# Patch extraction 

extract_polygon_centroid_patches <- function(image, polygons,
                                             patch_size     = 256,
                                             img_dir, mask_dir,
                                             season_prefix  = "krem_feb_july_0.9m",
                                             target_classes = 1,
                                             class_field    = "Class") {
  
  class_values    <- values(polygons)[[class_field]]
  polygons_target <- polygons[class_values %in% target_classes, ]
  
  if (nrow(polygons_target) == 0) {
    stop("No polygons found for class(es): ",
         paste(target_classes, collapse = ", "),
         " in column '", class_field, "'.")
  }
  
  message("Found ", nrow(polygons_target),
          " polygons for class ", paste(target_classes, collapse = ", "))
  
  cents   <- centroids(polygons_target)
  img_res <- res(image)
  half_x  <- (patch_size / 2) * img_res[1]
  half_y  <- (patch_size / 2) * img_res[2]
  img_ext <- ext(image)
  
  saved <- shifted <- skipped <- 0L
  
  for (i in seq_len(nrow(cents))) {
    
    xy <- crds(cents[i, ])
    cx <- xy[1, 1]; cy <- xy[1, 2]
    
    patch_ext <- ext(cx - half_x, cx + half_x,
                     cy - half_y, cy + half_y)
    
    # Shift patch inward if it overshoots the image edge.
    shift_x <- if (patch_ext$xmin < img_ext$xmin) img_ext$xmin - patch_ext$xmin
    else if (patch_ext$xmax > img_ext$xmax) img_ext$xmax - patch_ext$xmax
    else 0
    shift_y <- if (patch_ext$ymin < img_ext$ymin) img_ext$ymin - patch_ext$ymin
    else if (patch_ext$ymax > img_ext$ymax) img_ext$ymax - patch_ext$ymax
    else 0
    
    if (shift_x != 0 || shift_y != 0) {
      patch_ext <- ext(patch_ext$xmin + shift_x, patch_ext$xmax + shift_x,
                       patch_ext$ymin + shift_y, patch_ext$ymax + shift_y)
      shifted <- shifted + 1L
      
      if (patch_ext$xmin < img_ext$xmin || patch_ext$xmax > img_ext$xmax ||
          patch_ext$ymin < img_ext$ymin || patch_ext$ymax > img_ext$ymax) {
        skipped <- skipped + 1L
        next
      }
    }
    
    img_patch <- crop(image, patch_ext)
    
    if (nrow(img_patch) != patch_size || ncol(img_patch) != patch_size) {
      skipped <- skipped + 1L
      next
    }
    
    mask_patch <- rasterize(polygons_target, img_patch,
                            field = 1, background = 0)
    
    fname     <- sprintf("%s_patch_%05d.tif", season_prefix, i)
    img_file  <- file.path(img_dir,  fname)
    mask_file <- file.path(mask_dir, fname)
    
    writeRaster(img_patch,  img_file,  overwrite = TRUE,
                filetype = "GTiff", datatype = "INT1U")
    writeRaster(mask_patch, mask_file, overwrite = TRUE,
                filetype = "GTiff", datatype = "INT1U")
    
    saved <- saved + 1L
  }
  
  message("Done. Saved ", saved,
          " patches; shifted ", shifted,
          "; skipped ", skipped, ".")
  
  invisible(list(saved = saved, shifted = shifted, skipped = skipped))
}

# Run the extraction function

extract_polygon_centroid_patches(
  image          = Kremetart_image,
  polygons       = Kremetart_tree_polygons,
  patch_size     = 256,
  img_dir        = img_dir,
  mask_dir       = mask_dir,
  season_prefix  = "krem_feb_july_0.9m",
  target_classes = 1,
  class_field    = "Class"
)

#-------------------------------------------------------------------------------
# Extract point-centred negatives image/mask patches for U-Net training to reduce
# class imbalance

# Load the images and the polygons and the points for the negatives

image_path    <- "My_Dataset/mapungubwe_feb_july_0.9m.tif"
polygons_path <- "My_Dataset/Wet_Kremetart_tree_polygons.gpkg"
points_path   <- "My_Dataset/Kremetart_negatives_examples.gpkg"

img_dir_pts  <- "Kremetart_Negative_points/feb_july_256_8bands_0.9/images"
mask_dir_pts <- "Kremetart_Negative_points/feb_july_256_8bands_0.9/masks"

dir.create(img_dir_pts,  recursive = TRUE, showWarnings = FALSE)
dir.create(mask_dir_pts, recursive = TRUE, showWarnings = FALSE)

# Load and align 

Kremetart_image         <- rast(image_path)
Kremetart_tree_polygons <- vect(polygons_path)
Kremetart_tree_points   <- vect(points_path)

if (!terra::same.crs(Kremetart_tree_polygons, Kremetart_image)) {
  Kremetart_tree_polygons <- project(Kremetart_tree_polygons, Kremetart_image)
}

Kremetart_tree_polygons <- crop(Kremetart_tree_polygons, ext(Kremetart_image))

# Patch extraction 

extract_point_centred_patches <- function(image, points, mask_polygons,
                                          patch_size          = 256,
                                          img_dir, mask_dir,
                                          season_prefix       = "feb_july_negative_0.9m",
                                          mask_target_classes = 1,
                                          mask_class_field    = "Class") {
  
  stopifnot(
    inherits(image, "SpatRaster"),
    inherits(points, "SpatVector"),
    inherits(mask_polygons, "SpatVector"),
    patch_size %% 2 == 0,
    mask_class_field %in% names(mask_polygons)
  )
  
  dir.create(img_dir,  recursive = TRUE, showWarnings = FALSE)
  dir.create(mask_dir, recursive = TRUE, showWarnings = FALSE)
  
  if (!terra::same.crs(points, image))        points        <- project(points, image)
  if (!terra::same.crs(mask_polygons, image)) mask_polygons <- project(mask_polygons, image)
  
  keep            <- values(mask_polygons)[[mask_class_field]] %in% mask_target_classes
  polygons_target <- mask_polygons[keep, ]
  
  if (nrow(polygons_target) == 0) {
    stop("No mask polygons match class(es): ",
         paste(mask_target_classes, collapse = ", "))
  }
  
  message("Processing ", nrow(points), " points against ",
          nrow(polygons_target), " mask polygons")
  
  img_res <- res(image)
  half_x  <- (patch_size / 2) * img_res[1]
  half_y  <- (patch_size / 2) * img_res[2]
  img_ext <- ext(image)
  xy      <- crds(points)
  
  saved <- shifted <- skipped <- failed <- 0L
  gdal_opts <- c("COMPRESS=LZW", "PREDICTOR=2")
  
  for (i in seq_len(nrow(points))) {
    
    result <- tryCatch({
      
      cx <- xy[i, 1]; cy <- xy[i, 2]
      patch_ext <- ext(cx - half_x, cx + half_x,
                       cy - half_y, cy + half_y)
      
      # Shift patch inward if it overshoots the image edge.
      shift_x <- if (patch_ext$xmin < img_ext$xmin) img_ext$xmin - patch_ext$xmin
      else if (patch_ext$xmax > img_ext$xmax) img_ext$xmax - patch_ext$xmax
      else 0
      shift_y <- if (patch_ext$ymin < img_ext$ymin) img_ext$ymin - patch_ext$ymin
      else if (patch_ext$ymax > img_ext$ymax) img_ext$ymax - patch_ext$ymax
      else 0
      
      was_shifted <- (shift_x != 0 || shift_y != 0)
      
      if (was_shifted) {
        patch_ext <- ext(patch_ext$xmin + shift_x, patch_ext$xmax + shift_x,
                         patch_ext$ymin + shift_y, patch_ext$ymax + shift_y)
        
        if (patch_ext$xmin < img_ext$xmin || patch_ext$xmax > img_ext$xmax ||
            patch_ext$ymin < img_ext$ymin || patch_ext$ymax > img_ext$ymax) {
          return("skipped")
        }
      }
      
      img_patch <- crop(image, patch_ext)
      
      if (nrow(img_patch) != patch_size || ncol(img_patch) != patch_size) {
        return("skipped")
      }
      
      mask_patch <- rasterize(polygons_target, img_patch,
                              field = 1L, background = 0L)
      
      fname <- sprintf("%s_point_%05d.tif", season_prefix, i)
      writeRaster(img_patch,  file.path(img_dir,  fname),
                  overwrite = TRUE, datatype = "INT1U", gdal = gdal_opts)
      writeRaster(mask_patch, file.path(mask_dir, fname),
                  overwrite = TRUE, datatype = "INT1U", gdal = gdal_opts)
      
      if (was_shifted) "shifted" else "saved"
      
    }, error = function(e) {
      warning("Point ", i, " failed: ", conditionMessage(e), call. = FALSE)
      "failed"
    })
    
    if (result == "saved")   saved   <- saved   + 1L
    if (result == "shifted") { saved <- saved + 1L; shifted <- shifted + 1L }
    if (result == "skipped") skipped <- skipped + 1L
    if (result == "failed")  failed  <- failed  + 1L
    
    if (i %% 25 == 0) message("  ...processed ", i, " / ", nrow(points))
  }
  
  message("Done. Saved: ", saved,
          " | Shifted: ", shifted,
          " | Skipped: ", skipped,
          " | Failed: ",  failed)
  
  invisible(list(saved = saved, shifted = shifted,
                 skipped = skipped, failed = failed))
}

# Run 

extract_point_centred_patches(
  image               = Kremetart_image,
  points              = Kremetart_tree_points,
  mask_polygons       = Kremetart_tree_polygons,
  patch_size          = 256,
  img_dir             = img_dir_pts,
  mask_dir            = mask_dir_pts,
  season_prefix       = "feb_july_negative_0.9m",
  mask_target_classes = 1,
  mask_class_field    = "Class"
)


# Plot 4 random image / mask pairs

set.seed(42)

# List all the image patches we just saved. 

img_files <- list.files(img_dir_pts, pattern = "\\.tif$", full.names = TRUE)

if (length(img_files) == 0) stop("No image patches found in ", img_dir)

# Pick 4 random patches 

n_pairs      <- min(4, length(img_files))
sample_files <- sample(img_files, n_pairs)

# Build matching mask paths.

mask_files <- file.path(mask_dir_pts, basename(sample_files))

# RGB bands

rgb_bands <- c(5, 3, 2)

# Set up a 4-row x 2-col plot grid: column 1 = image, column 2 = mask.

op <- par(mfrow = c(n_pairs, 2), mar = c(1.5, 1.5, 2, 1.5))

for (i in seq_len(n_pairs)) {
  
  # Load patch i's image and mask from disk.
  
  img  <- rast(sample_files[i])
  mask <- rast(mask_files[i])
  
  # Plot the RGB composite. 
  
  plotRGB(img,
          r = rgb_bands[1], g = rgb_bands[2], b = rgb_bands[3],
          stretch = "lin",
          mar = NULL,
          main = basename(sample_files[i]))
  
  # Plot the binary mask. 
  
  plot(mask,
       col    = c("grey90", "red"),
       type   = "classes",
       legend = FALSE,
       axes   = FALSE,
       main   = "mask (1 = tree)")
}

par(op)


## Copy files into new folders

#  Source directories 

src_img_dir  <- "/Kremetart_Negative_points/feb_july_256_8bands_0.9/images"
src_mask_dir <- "/Kremetart_Negative_points/feb_july_256_8bands_0.9/masks"

# Destination directories

dst_img_dir  <- "/Kremetart_feb_polygons/new_feb_july_256_8bands_0.9m/images"
dst_mask_dir <- "/Kremetart_feb_polygons/new_feb_july_256_8bands_0.9m/masks"

dir.create(dst_img_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(dst_mask_dir, recursive = TRUE, showWarnings = FALSE)

# Collect source files 

img_files  <- sort(list.files(src_img_dir,  pattern = "\\.tif$", full.names = TRUE))
mask_files <- sort(list.files(src_mask_dir, pattern = "\\.tif$", full.names = TRUE))

cat(sprintf("Images found : %d\n", length(img_files)))
cat(sprintf("Masks found  : %d\n", length(mask_files)))

# Safety check: counts must match before copying 

stopifnot("Image and mask counts do not match — check your folders." =
            length(img_files) == length(mask_files))

# Copy files 

n <- length(img_files)

for (i in seq_len(n)) {
  
  file.copy(img_files[i],
            file.path(dst_img_dir, basename(img_files[i])),
            overwrite = TRUE)
  
  file.copy(mask_files[i],
            file.path(dst_mask_dir, basename(mask_files[i])),
            overwrite = TRUE)
  
  if (i %% 50 == 0) {
    cat(sprintf("  Copied %d / %d pairs...\n", i, n))
  }
}

cat(sprintf("\nDone. %d image/mask pairs copied.\n", n))
cat(sprintf("Images -> %s\n", dst_img_dir))
cat(sprintf("Masks  -> %s\n", dst_mask_dir))

#  Count .tif files in each folder 

n_images <- length(list.files(dst_img_dir,  pattern = "\\.tif$"))
n_masks  <- length(list.files(dst_mask_dir, pattern = "\\.tif$"))

# Report 

cat(sprintf("Images : %d\n", n_images))
cat(sprintf("Masks  : %d\n", n_masks))

if (n_images == n_masks) {
  cat(sprintf("Paired : %d matched pairs — all good.\n", n_images))
} else {
  cat(sprintf("WARNING: counts differ by %d — check for missing files.\n",
              abs(n_images - n_masks)))
}


