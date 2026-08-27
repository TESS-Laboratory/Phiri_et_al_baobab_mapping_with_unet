#-------------------------------------------------------------------------------
# WorldView-3 pre-processing
#   Part 1  stack Kremetart seasons, rescale to 8-bit, save at 0.3 m
#   Part 2  load the saved 0.3 m images and aggregate to 0.9 m, 2.7 m, 8.1 m
#
# Each band is stretched on its own minimum and maximum within its own image.
#-------------------------------------------------------------------------------
library(terra)
setwd("C:/Users/cp844/University of Exeter/TESSLab - Chafika Phiri Data/Objective 3 Manuscript/")
terraOptions(memfrac = 0.6, progress = 1)
# terraOptions(tempdir = "D:/terra_temp")   # point at a drive with free space

# Paths
source_imagery <- c(
  kremetart_february = "Satellite Data/Data/Mosaicked_Images/Kremetart/Kremetart_feb_8band_pansharpened.tif",
  kremetart_july     = "Satellite Data/Data/Mosaicked_Images/Kremetart/Kremetart_july_8band_pansharpened.tif",
  mapungubwe_july    = "Satellite Data/Data/Mosaicked_Images/Mapungubwe/Mapungubwe_july_8band_pansharpened.tif"
)
output_folder <- "Satellite Data/Data/Processed"
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

# 0.3 m, 8-bit outputs of Part 1; also the inputs Part 2 reads back in.
fine_grain_outputs <- c(
  kremetart_february      = file.path(output_folder, "kremetart_february_0.3m.tif"),
  kremetart_july          = file.path(output_folder, "kremetart_july_0.3m.tif"),
  kremetart_february_july = file.path(output_folder, "kremetart_february_july_0.3m.tif"),
  mapungubwe_july         = file.path(output_folder, "mapungubwe_july_0.3m.tif")
)
stretch_record_path <- file.path(output_folder, "eight_bit_stretch_limits.csv")

# Lossless compression, tiled for fast windowed reads, one byte per pixel.
raster_write_options <- c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
                          "BIGTIFF=IF_SAFER")

# Coarser grains built from the 0.3 m grid in Part 2.
aggregation_factors <- c(3, 9, 27)
aggregation_labels  <- c("0.9m", "2.7m", "8.1m")

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------

# Flags the zero-filled padding outside the imaged swath.
flag_background_pixels <- function(image_stack) {
  zero_band_count <- sum(image_stack == 0)
  background_mask <- zero_band_count == terra::nlyr(image_stack)
  return(background_mask)
}

# Stretches each band to 0-255 on its own limits, measured on the foreground
# only, and records the limits used.
rescale_to_eight_bit <- function(image_stack,
                                 exclude_background = TRUE,
                                 stretch_method     = c("minmax", "percentile"),
                                 lower_probability  = 0.02,
                                 upper_probability  = 0.98,
                                 sample_size        = 1e6) {
  
  stretch_method <- match.arg(stretch_method)
  
  if (isTRUE(exclude_background)) {
    background_mask   <- flag_background_pixels(image_stack)
    measurement_image <- terra::mask(image_stack, background_mask, maskvalues = TRUE)
  } else {
    measurement_image <- image_stack
  }
  
  if (stretch_method == "minmax") {
    lower_limits <- terra::global(measurement_image, "min", na.rm = TRUE)[, 1]
    upper_limits <- terra::global(measurement_image, "max", na.rm = TRUE)[, 1]
  } else {
    # Percentiles are taken from a regular spatial sample rather than every cell.
    pixel_sample <- terra::spatSample(measurement_image, size = sample_size,
                                      method = "regular", na.rm = TRUE)
    lower_limits <- apply(pixel_sample, 2, stats::quantile,
                          probs = lower_probability, na.rm = TRUE)
    upper_limits <- apply(pixel_sample, 2, stats::quantile,
                          probs = upper_probability, na.rm = TRUE)
  }
  
  rescaled_bands <- vector("list", terra::nlyr(image_stack))
  
  for (band_index in seq_len(terra::nlyr(image_stack))) {
    
    band_minimum <- lower_limits[band_index]
    band_maximum <- upper_limits[band_index]
    band_range   <- band_maximum - band_minimum
    
    # A flat or empty band would otherwise fill the whole layer with Inf or NaN.
    if (!is.finite(band_range) || band_range <= 0) {
      stop("Band ", band_index, " has a zero or non-finite range; check the source file.")
    }
    
    scaled_band <- (image_stack[[band_index]] - band_minimum) / band_range * 255
    
    # Clamping catches the padding, which falls below zero, and rounding error.
    rescaled_bands[[band_index]] <- terra::clamp(round(scaled_band),
                                                 lower = 0, upper = 255, values = TRUE)
  }
  
  output_stack <- terra::rast(rescaled_bands)
  names(output_stack) <- names(image_stack)
  
  # The limits travel with the object so the caller can build the audit record.
  attr(output_stack, "stretch_limits") <- data.frame(
    band             = seq_len(terra::nlyr(image_stack)),
    band_name        = names(image_stack),
    lower_limit      = as.numeric(lower_limits),
    upper_limit      = as.numeric(upper_limits),
    stretch_method   = stretch_method,
    stringsAsFactors = FALSE
  )
  
  return(output_stack)
}

# Trims the image so its dimensions divide exactly by the factor, which makes the
# coarse grids nest perfectly inside the fine one.
crop_to_multiple_of <- function(image_stack, aggregation_factor) {
  
  cell_size       <- terra::res(image_stack)
  original_extent <- terra::ext(image_stack)
  
  usable_columns <- (terra::ncol(image_stack) %/% aggregation_factor) * aggregation_factor
  usable_rows    <- (terra::nrow(image_stack) %/% aggregation_factor) * aggregation_factor
  
  trimmed_extent <- terra::ext(
    original_extent$xmin,
    original_extent$xmin + usable_columns * cell_size[1],
    original_extent$ymax - usable_rows    * cell_size[2],
    original_extent$ymax
  )
  
  return(terra::crop(image_stack, trimmed_extent))
}

# Writes the image at every coarser grain, averaging whole blocks of cells as a
# coarser sensor would, always working from the 0.3 m source rather than chaining.
build_multi_scale_series <- function(image_stack, output_prefix,
                                     factors = aggregation_factors,
                                     labels  = aggregation_labels) {
  
  for (scale_index in seq_along(factors)) {
    
    this_factor <- factors[scale_index]
    this_label  <- labels[scale_index]
    
    message("  writing ", this_label, " (factor ", this_factor, ")")
    
    # na.rm keeps a block that is only partly filled instead of returning NA.
    output_image <- terra::aggregate(image_stack, fact = this_factor,
                                     fun = "mean", na.rm = TRUE)
    # Averaging gives decimals; round so the integer format does not truncate.
    output_image <- round(output_image)
    
    terra::writeRaster(output_image,
                       filename  = paste0(output_prefix, "_", this_label, ".tif"),
                       datatype  = "INT1U",
                       gdal      = raster_write_options,
                       overwrite = TRUE)
  }
  invisible(NULL)
}

#===============================================================================
# PART 1: stack, rescale to 8-bit, save at 0.3 m
#===============================================================================

# Check the inputs
missing_files <- source_imagery[!file.exists(source_imagery)]
if (length(missing_files) > 0) {
  stop("These input files were not found:\n  ",
       paste(names(missing_files), missing_files, sep = " -> ", collapse = "\n  "))
}
for (image_label in names(source_imagery)) {
  this_image <- terra::rast(source_imagery[[image_label]])
  message(image_label,
          " | bands: ", terra::nlyr(this_image),
          " | cell size: ", paste(signif(terra::res(this_image), 3), collapse = " x "),
          " | dimensions: ", terra::nrow(this_image), " x ", terra::ncol(this_image),
          " | type: ", terra::datatype(this_image)[1])
  if (terra::nlyr(this_image) != 8) {
    warning(image_label, " has ", terra::nlyr(this_image), " bands, not 8.")
  }
}

# Stack Kremetart February and July into one 16-band raster
message("\nPART 1a: stacking the two Kremetart seasons")
february_image <- terra::rast(source_imagery[["kremetart_february"]])
july_image     <- terra::rast(source_imagery[["kremetart_july"]])

if (!terra::same.crs(february_image, july_image)) {
  stop("February and July are in different coordinate reference systems.\n",
       "  February: ", terra::crs(february_image, describe = TRUE)$name, "\n",
       "  July:     ", terra::crs(july_image, describe = TRUE)$name, "\n",
       "Reproject one to the other with terra::project() before stacking.")
}
# Two mosaics of the same site rarely cover the same rectangle, so use the
# overlap common to both dates.
common_extent <- terra::intersect(terra::ext(february_image), terra::ext(july_image))
if (is.null(common_extent)) {
  stop("The February and July extents do not overlap. Check the inputs.")
}
february_cropped <- terra::crop(february_image, common_extent)
july_cropped     <- terra::crop(july_image,     common_extent)

# Cell boundaries may still be offset by a fraction of a pixel after cropping.
grids_align <- tryCatch({
  terra::compareGeom(february_cropped, july_cropped,
                     crs = TRUE, ext = TRUE, rowcol = TRUE, res = TRUE,
                     stopOnError = TRUE)
  TRUE
}, error = function(e) FALSE)
if (!grids_align) {
  message("  grids do not align exactly; snapping July onto February's grid")
  # Nearest neighbour keeps the original digital numbers; bilinear would invent new ones.
  july_cropped <- terra::resample(july_cropped, february_cropped, method = "near")
}
kremetart_sixteen_band <- c(february_cropped, july_cropped)
names(kremetart_sixteen_band) <- c(
  paste0("february_band", seq_len(terra::nlyr(february_cropped))),
  paste0("july_band",     seq_len(terra::nlyr(july_cropped)))
)
message("  stacked: ", terra::nlyr(kremetart_sixteen_band), " bands, ",
        terra::nrow(kremetart_sixteen_band), " x ",
        terra::ncol(kremetart_sixteen_band), " cells")

# Rescale to 8-bit, trim to a footprint divisible by 27 (so the coarser grains
# in Part 2 nest exactly), and save at 0.3 m
message("\nPART 1b: rescaling to 8-bit and saving at 0.3 m")
images_to_rescale <- list(
  kremetart_february      = february_cropped,
  kremetart_july          = july_cropped,
  kremetart_february_july = kremetart_sixteen_band,
  mapungubwe_july         = terra::rast(source_imagery[["mapungubwe_july"]])
)
stretch_records <- vector("list", length(images_to_rescale))
names(stretch_records) <- names(images_to_rescale)

for (image_label in names(images_to_rescale)) {
  
  message("  ", image_label)
  
  eight_bit_image <- rescale_to_eight_bit(images_to_rescale[[image_label]],
                                          exclude_background = TRUE,
                                          stretch_method     = "minmax")
  eight_bit_image <- crop_to_multiple_of(eight_bit_image, max(aggregation_factors))
  
  terra::writeRaster(eight_bit_image,
                     filename  = fine_grain_outputs[[image_label]],
                     datatype  = "INT1U",
                     gdal      = raster_write_options,
                     overwrite = TRUE)
  
  this_record       <- attr(eight_bit_image, "stretch_limits")
  this_record$image <- image_label
  stretch_records[[image_label]] <- this_record
}
# An audit trail, needed to stretch any new scene by the same procedure later.
stretch_record_table <- do.call(rbind, stretch_records)
write.csv(stretch_record_table, stretch_record_path, row.names = FALSE)
message("  stretch limits written to ", stretch_record_path)
print(stretch_record_table)

#===============================================================================
# PART 2: load the saved 0.3 m images and aggregate to 0.9 m, 2.7 m, 8.1 m
#===============================================================================

message("\nPART 2: aggregating the saved 0.3 m images")
for (image_label in names(fine_grain_outputs)) {
  
  message(image_label, ":")
  
  # Re-opened from disk to confirm the file wrote correctly and to aggregate
  # exactly what a later analysis will read.
  build_multi_scale_series(
    image_stack   = terra::rast(fine_grain_outputs[[image_label]]),
    output_prefix = file.path(output_folder, image_label)
  )
}
# Sixteen files result: four products at four grains each, all sharing a footprint.
#   kremetart_february_0.3m.tif       ... 0.9m, 2.7m, 8.1m
#   kremetart_july_0.3m.tif           ... 0.9m, 2.7m, 8.1m
#   kremetart_february_july_0.3m.tif  ... 0.9m, 2.7m, 8.1m
#   mapungubwe_july_0.3m.tif          ... 0.9m, 2.7m, 8.1m

# Verification
message("\nVERIFICATION")
produced_files <- list.files(output_folder,
                             pattern = "_(0\\.3m|0\\.9m|2\\.7m|8\\.1m)\\.tif$",
                             full.names = TRUE)
for (this_file in produced_files) {
  this_raster <- terra::rast(this_file)
  message(basename(this_file),
          " | bands: ", terra::nlyr(this_raster),
          " | cell size: ", paste(signif(terra::res(this_raster), 3), collapse = " x "),
          " | range: ",
          round(min(terra::global(this_raster, "min", na.rm = TRUE)[, 1]), 1), " to ",
          round(max(terra::global(this_raster, "max", na.rm = TRUE)[, 1]), 1))
}
# Every range above must fall inside 0 to 255; anything outside should not be used.

# Visual check
# WorldView-3 band order: 1 Coastal, 2 Blue, 3 Green, 4 Yellow, 5 Red,
# 6 Red Edge, 7 NIR1, 8 NIR2. True colour is therefore bands 5, 3 and 2.
kremetart_coarse <- terra::rast(file.path(output_folder, "kremetart_february_8.1m.tif"))
# The stretch here affects the display only, not the file.
terra::plotRGB(kremetart_coarse, r = 5, g = 3, b = 2, stretch = "lin",
               main = "Kremetart, February, 8.1 m, true colour")
# False colour infrared, more informative for vegetation:
# terra::plotRGB(kremetart_coarse, r = 7, g = 5, b = 3, stretch = "lin")
# July occupies bands 9-16 of the stack, so July true colour is 13, 11 and 10:
# kremetart_both <- terra::rast(file.path(output_folder, "kremetart_february_july_8.1m.tif"))
# terra::plotRGB(kremetart_both, r = 13, g = 11, b = 10, stretch = "lin")