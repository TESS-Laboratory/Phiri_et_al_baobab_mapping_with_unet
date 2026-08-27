#-------------------------------------------------------------------------------
# Introduction

# This script produces 256x256-pixel U-Net image/mask patches from the 16 rasters made by
# the WorldView-3 pre-processing script (4 products x 4 grains: 0.3/0.9/2.7/8.1m).
# "Positive" patches are centred on tree-polygon centroids; "negative" patches
# are centred on hand-placed non-tree points. Patch size in pixels is fixed
# across grains, so ground footprint grows with coarser resolution.
#-------------------------------------------------------------------------------

# Load required packages

library(terra)  

#-------------------------------------------------------------------------------
# Project set-up
#-------------------------------------------------------------------------------

setwd("C:/Users/cp844/University of Exeter/TESSLab - Chafika Phiri Data/Objective 3 Manuscript/")

raster_write_options <- c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
                          "BIGTIFF=IF_SAFER")

patch_size <- 256L                               # fixed pixel size at every grain
resolutions <- c("0.3m", "0.9m", "2.7m", "8.1m")  # must match resampled file-name suffixes
resampled_image_folder <- "Satellite Data/Data/Processed"  # holds the 16 input rasters
patches_root <- "Training Patches"               # top folder for all six patch collections

#-------------------------------------------------------------------------------
# Polygon files
#-------------------------------------------------------------------------------
kremetart_polygon_path          <- "Training Labels/Dry_Season_Kremetart_Tree_Polygons.gpkg"
kremetart_negative_points_path  <- "Training Labels/Kremetart_negatives_examples.gpkg"

mapungubwe_polygon_path         <- "Training Labels/Dry_Season_Mapungubwe_Tree_Polygons.gpkg"
mapungubwe_negative_points_path <- "Training Labels/Mapungubwe_Non_baobab_points.gpkg"

class_field    <- "Class"   # attribute column marking a tree polygon
target_classes <- 1         # value in that column meaning "tree"

#-------------------------------------------------------------------------------
# Job registries: which raster, which ground truth, which output folder
#-------------------------------------------------------------------------------
# One row per positive-patch job
positive_jobs <- data.frame(
  file_prefix   = c("kremetart_february_july", "kremetart_february",
                    "kremetart_july",          "mapungubwe_july"),
  output_folder = file.path(patches_root,
                            c("Kremetart_feb_july_patches", "Kremetart_feb_patches",
                              "Kremetart_july_patches",     "Mapungubwe_july_patches")),
  polygon_path  = c(kremetart_polygon_path, kremetart_polygon_path,
                    kremetart_polygon_path, mapungubwe_polygon_path),
  stringsAsFactors = FALSE
)

negative_jobs <- data.frame(
  file_prefix   = c("kremetart_february_july", "kremetart_february",
                    "kremetart_july",          "mapungubwe_july"),
  label         = c("feb_july", "feb", "july", "july"),
  output_folder = file.path(patches_root,
                            c("Kremetart_Negative_patches", "Kremetart_Negative_patches",
                              "Kremetart_Negative_patches", "Mapungubwe_Negative_patches")),
  points_path   = c(kremetart_negative_points_path, kremetart_negative_points_path,
                    kremetart_negative_points_path, mapungubwe_negative_points_path),
  polygon_path  = c(kremetart_polygon_path, kremetart_polygon_path,
                    kremetart_polygon_path, mapungubwe_polygon_path),
  stringsAsFactors = FALSE
)

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------

# Stops with one message listing every missing file, checked upfront
check_required_files_exist <- function(paths) {
  missing_paths <- paths[!file.exists(paths)]
  if (length(missing_paths) > 0) {
    stop("These required files were not found:\n  ",
         paste(missing_paths, collapse = "\n  "),
         call. = FALSE)
  }
  invisible(TRUE)
}

# Reprojects a vector onto a raster's CRS only if needed. Uses same.crs()
# rather than comparing crs() text, which can flag equivalent CRS as different.
align_vector_crs <- function(vector_data, reference_image) {
  if (!terra::same.crs(vector_data, reference_image)) {
    vector_data <- terra::project(vector_data, reference_image)
  }
  vector_data
}

# Shifts a patch extent inward so it fits inside the image if it overshoots
# an edge. Shared by both patch types (was duplicated in the earlier scripts).
shift_patch_inward <- function(patch_ext, image_ext) {
  
  shift_x <- if (patch_ext$xmin < image_ext$xmin) image_ext$xmin - patch_ext$xmin
  else if (patch_ext$xmax > image_ext$xmax) image_ext$xmax - patch_ext$xmax
  else 0
  shift_y <- if (patch_ext$ymin < image_ext$ymin) image_ext$ymin - patch_ext$ymin
  else if (patch_ext$ymax > image_ext$ymax) image_ext$ymax - patch_ext$ymax
  else 0
  
  shifted_ext <- terra::ext(patch_ext$xmin + shift_x, patch_ext$xmax + shift_x,
                            patch_ext$ymin + shift_y, patch_ext$ymax + shift_y)
  
  list(
    ext          = shifted_ext,
    was_shifted  = (shift_x != 0 || shift_y != 0),
    out_of_bounds = shifted_ext$xmin < image_ext$xmin || shifted_ext$xmax > image_ext$xmax ||
      shifted_ext$ymin < image_ext$ymin || shifted_ext$ymax > image_ext$ymax
  )
}

# Shared engine for both positive and negative patches: given any centre
# points, crops a patch_size window around each, shifts it inward if needed,
# rasterises mask_polygons to match, and writes both as 8-bit GeoTIFFs.
extract_patches_from_points <- function(image, centre_points, mask_polygons,
                                        patch_size, img_dir, mask_dir,
                                        file_stub, name_suffix = "patch") {
  
  dir.create(img_dir,  recursive = TRUE, showWarnings = FALSE)
  dir.create(mask_dir, recursive = TRUE, showWarnings = FALSE)
  
  img_res <- terra::res(image)
  half_x  <- (patch_size / 2) * img_res[1]
  half_y  <- (patch_size / 2) * img_res[2]
  img_ext <- terra::ext(image)
  xy      <- terra::crds(centre_points)
  
  saved <- shifted <- skipped <- failed <- 0L
  
  for (i in seq_len(nrow(centre_points))) {
    
    # Each branch ends in a bare string, not return("skipped"): return()
    # inside tryCatch()'s expression exits this whole function immediately,
    # not just the current loop iteration - that bug in the earlier script
    # silently dropped every point after the first skip.
    outcome <- tryCatch({
      
      cx <- xy[i, 1]; cy <- xy[i, 2]
      patch_ext <- terra::ext(cx - half_x, cx + half_x, cy - half_y, cy + half_y)
      
      shift_result <- shift_patch_inward(patch_ext, img_ext)
      patch_ext    <- shift_result$ext
      
      if (shift_result$out_of_bounds) {
        
        "skipped"   # does not fit even after shifting
        
      } else {
        
        img_patch <- terra::crop(image, patch_ext)
        
        if (nrow(img_patch) != patch_size || ncol(img_patch) != patch_size) {
          
          "skipped"   # partial patch, wrong pixel dimensions
          
        } else {
          
          mask_patch <- terra::rasterize(mask_polygons, img_patch, field = 1L, background = 0L)
          
          fname <- sprintf("%s_%s_%05d.tif", file_stub, name_suffix, i)
          terra::writeRaster(img_patch,  file.path(img_dir,  fname),
                             overwrite = TRUE, datatype = "INT1U", gdal = raster_write_options)
          terra::writeRaster(mask_patch, file.path(mask_dir, fname),
                             overwrite = TRUE, datatype = "INT1U", gdal = raster_write_options)
          
          if (shift_result$was_shifted) "shifted" else "saved"
        }
      }
      
    }, error = function(e) {
      warning("Point ", i, " (", file_stub, ") failed: ", conditionMessage(e), call. = FALSE)
      "failed"
    })
    
    if (outcome == "saved")   saved   <- saved   + 1L
    if (outcome == "shifted") { saved <- saved + 1L; shifted <- shifted + 1L }
    if (outcome == "skipped") skipped <- skipped + 1L
    if (outcome == "failed")  failed  <- failed  + 1L
  }
  
  message("    saved ", saved, " | shifted ", shifted,
          " | skipped ", skipped, " | failed ", failed)
  
  list(saved = saved, shifted = shifted, skipped = skipped, failed = failed)
}

# Filters polygons to the target class(es); stops if none match
select_target_polygons <- function(polygons, class_field, target_classes) {
  if (!class_field %in% names(polygons)) {
    stop("Column '", class_field, "' was not found in this polygon file. ",
         "Available columns: ", paste(names(polygons), collapse = ", "), call. = FALSE)
  }
  class_values    <- terra::values(polygons)[[class_field]]
  polygons_target <- polygons[class_values %in% target_classes, ]
  if (nrow(polygons_target) == 0) {
    stop("No polygons found for class(es): ", paste(target_classes, collapse = ", "),
         " in column '", class_field, "'.", call. = FALSE)
  }
  polygons_target
}

#-------------------------------------------------------------------------------
# Check every input file exists before processing anything
#-------------------------------------------------------------------------------
required_rasters <- file.path(
  resampled_image_folder,
  paste0(rep(unique(c(positive_jobs$file_prefix, negative_jobs$file_prefix)),
             each = length(resolutions)),
         "_", resolutions, ".tif")
)
required_vectors <- unique(c(positive_jobs$polygon_path,
                             negative_jobs$points_path, negative_jobs$polygon_path))

check_required_files_exist(c(required_rasters, required_vectors))
message("All ", length(required_rasters), " resampled rasters and ",
        length(required_vectors), " vector files were found.\n")

#-------------------------------------------------------------------------------
# Positive (polygon-centroid) patches
#-------------------------------------------------------------------------------
extraction_log <- data.frame()   # tally per product/resolution, written to CSV at the end

for (resolution in resolutions) {
  message("=== Positive patches at ", resolution, " ===")
  
  for (job_index in seq_len(nrow(positive_jobs))) {
    
    job <- positive_jobs[job_index, ]
    message("  ", job$file_prefix)
    
    image_path <- file.path(resampled_image_folder,
                            paste0(job$file_prefix, "_", resolution, ".tif"))
    image      <- terra::rast(image_path)
    
    polygons        <- terra::vect(job$polygon_path)
    polygons        <- align_vector_crs(polygons, image)
    polygons        <- terra::crop(polygons, terra::ext(image))
    polygons_target <- select_target_polygons(polygons, class_field, target_classes)
    centroids       <- terra::centroids(polygons_target)
    
    img_dir  <- file.path(job$output_folder, resolution, "images")
    mask_dir <- file.path(job$output_folder, resolution, "masks")
    
    tally <- extract_patches_from_points(
      image         = image,
      centre_points = centroids,
      mask_polygons = polygons_target,
      patch_size    = patch_size,
      img_dir       = img_dir,
      mask_dir      = mask_dir,
      file_stub     = paste0(job$file_prefix, "_", resolution),
      name_suffix   = "patch"
    )
    
    extraction_log <- rbind(extraction_log, data.frame(
      patch_type = "positive", product = job$file_prefix, resolution = resolution,
      output_folder = img_dir, saved = tally$saved, shifted = tally$shifted,
      skipped = tally$skipped, failed = tally$failed, stringsAsFactors = FALSE
    ))
  }
}

#-------------------------------------------------------------------------------
# Negative (point-centred) patches
#-------------------------------------------------------------------------------
for (resolution in resolutions) {
  message("\n=== Negative patches at ", resolution, " ===")
  
  for (job_index in seq_len(nrow(negative_jobs))) {
    
    job <- negative_jobs[job_index, ]
    message("  ", job$file_prefix, " (", job$label, ")")
    
    image_path <- file.path(resampled_image_folder,
                            paste0(job$file_prefix, "_", resolution, ".tif"))
    image      <- terra::rast(image_path)
    
    negative_points <- terra::vect(job$points_path)
    negative_points <- align_vector_crs(negative_points, image)
    
    polygons        <- terra::vect(job$polygon_path)
    polygons        <- align_vector_crs(polygons, image)
    polygons        <- terra::crop(polygons, terra::ext(image))
    polygons_target <- select_target_polygons(polygons, class_field, target_classes)
    
    img_dir  <- file.path(job$output_folder, job$label, resolution, "images")
    mask_dir <- file.path(job$output_folder, job$label, resolution, "masks")
    
    tally <- extract_patches_from_points(
      image         = image,
      centre_points = negative_points,
      mask_polygons = polygons_target,
      patch_size    = patch_size,
      img_dir       = img_dir,
      mask_dir      = mask_dir,
      file_stub     = paste0(job$file_prefix, "_", resolution),
      name_suffix   = "point"
    )
    
    extraction_log <- rbind(extraction_log, data.frame(
      patch_type = "negative", product = paste0(job$file_prefix, " (", job$label, ")"),
      resolution = resolution, output_folder = img_dir, saved = tally$saved,
      shifted = tally$shifted, skipped = tally$skipped, failed = tally$failed,
      stringsAsFactors = FALSE
    ))
  }
}

#-------------------------------------------------------------------------------
# Audit trail and sanity checks
#-------------------------------------------------------------------------------
log_path <- file.path(patches_root, "patch_extraction_log.csv")
write.csv(extraction_log, log_path, row.names = FALSE)
message("\nExtraction log written to ", log_path)
print(extraction_log)

# Confirm image/mask counts match for every folder written
message("\n=== Image/mask count check ===")
for (i in seq_len(nrow(extraction_log))) {
  img_dir  <- extraction_log$output_folder[i]
  mask_dir <- sub("[/\\\\]images$", "/masks", img_dir)
  n_images <- length(list.files(img_dir,  pattern = "\\.tif$"))
  n_masks  <- length(list.files(mask_dir, pattern = "\\.tif$"))
  status   <- if (n_images == n_masks) "OK" else "MISMATCH"
  message(sprintf("  [%s] %s | images: %d | masks: %d", status, img_dir, n_images, n_masks))
}

#-------------------------------------------------------------------------------
# plot random image/mask pairs from one folder.
#-------------------------------------------------------------------------------
plot_sample_patches <- function(img_dir, mask_dir, rgb_bands, n_pairs = 4, seed = 42) {
  
  set.seed(seed)
  img_files <- list.files(img_dir, pattern = "\\.tif$", full.names = TRUE)
  if (length(img_files) == 0) stop("No image patches found in ", img_dir, call. = FALSE)
  
  n_pairs      <- min(n_pairs, length(img_files))
  sample_files <- sample(img_files, n_pairs)
  mask_files   <- file.path(mask_dir, basename(sample_files))
  
  op <- par(mfrow = c(n_pairs, 2), mar = c(1.5, 1.5, 2, 1.5))
  for (i in seq_len(n_pairs)) {
    img  <- terra::rast(sample_files[i])
    mask <- terra::rast(mask_files[i])
    terra::plotRGB(img, r = rgb_bands[1], g = rgb_bands[2], b = rgb_bands[3],
                   stretch = "lin", mar = NULL, main = basename(sample_files[i]))
    plot(mask, col = c("grey90", "red"), type = "classes", legend = FALSE,
         axes = FALSE, main = "mask (1 = tree)")
  }
  par(op)
}

# True colour = bands 5,3,2 in any single-date 8-band product; in the
# 16-band Feb+July stack, Feb = bands 1-8, July = bands 9-16 (July true
# colour = 13,11,10).
plot_sample_patches(
  img_dir   = file.path("Training Patches", "Kremetart_feb_july_patches", "0.9m", "images"),
  mask_dir  = file.path("Training Patches", "Kremetart_feb_july_patches", "0.9m", "masks"),
  rgb_bands = c(5, 3, 2)   # February true colour
)