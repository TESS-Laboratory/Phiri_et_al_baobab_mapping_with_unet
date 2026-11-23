# =============================================================================
# 01_aggregate_imagery.R
# Purpose : Writes each site's pan-sharpened 8-band WorldView-3 image at
#           0.3 m (a lossless copy) and at 0.9, 2.7 and 8.1 m (3 x 3 means,
#           three times) into one imagery folder, and records per-band
#           statistics used later to standardise the network inputs.
# =============================================================================

library(terra)                                       # raster reading, writing and aggregation

# --- Setup 

setwd("~/Chafika/Phiri_et_al_baobab_mapping_with_unet/October_2026/") # my workstation folder

settings <- list(
  images = c(                                        # one 0.3 m pan-sharpened 8-band image per site
    kremetart  = file.path("Data", "Satellite_images", "krem_july_pansharpened.tif"),
    mapungubwe = file.path("Data", "Satellite_images", "maps_july_pansharpened.tif")),
  band_names = c("coastal", "blue", "green", "yellow", # WorldView-3 multispectral band order
                 "red", "red_edge", "nir1", "nir2"),
  resolutions        = c(0.3, 0.9, 2.7, 8.1),        # metres; first = original
  aggregation_factor = 3L,                           # 3 x 3 pixels averaged into one
  nodata_value       = NULL,                         # NULL: use the NoData value stored in the file;
  # or give it, e.g. 0 or 65535
  working_crs        = "EPSG:32735",                 # WGS 84 / UTM 35S (metres)
  tile_size          = 128L,                         # GeoTIFF tile side (pixels)
  sample_cells       = 200000L,                      # cells per site and resolution for band statistics
  gdal_cache_mb      = 2048,                         # GDAL write cache (MB); holds one strip of tiles
  folder_imagery     = file.path("Data", "imagery")
)

# =============================================================================
# Functions
# =============================================================================

# resolution_label(): 0.3 -> "0.3m"; used in file names and tables
resolution_label <- function(resolution) sprintf("%.1fm", resolution)

# -----------------------------------------------------------------------------
# file_nodata(): the NoData value written in a GeoTIFF's header (NA if none).
# terra hides these pixels as NA on reading but does not report the value,
# so it is read from the GDAL description of the file.
# -----------------------------------------------------------------------------
file_nodata <- function(path) {
  line <- grep("NoData Value=", describe(path), value = TRUE)
  if (length(line) == 0) return(NA_real_)
  as.numeric(sub(".*NoData Value=", "", line[1]))
}

# -----------------------------------------------------------------------------
# restore_partial_nodata(): the NoData value can also be a genuine reading in
# one band (for instance a saturated band over bright sand). A pixel is
# treated as empty only when every band is missing; where only some bands
# are missing, those bands get the NoData value back as a real value.
# -----------------------------------------------------------------------------
restore_partial_nodata <- function(values, nodata_value) {
  missing <- is.na(values)
  empty <- missing[, , 1]                            # TRUE where every band is missing
  for (b in seq_len(dim(values)[3])[-1]) empty <- empty & missing[, , b]
  for (b in seq_len(dim(values)[3])) {
    values[, , b][missing[, , b] & !empty] <- nodata_value
  }
  values
}

# -----------------------------------------------------------------------------
# block_mean_3x3(): mean of each 3 x 3 block of a matrix, ignoring NA.
# The matrix is padded with NA to a multiple of 3 (as terra::aggregate does),
# then split into 9 values per block: R stores matrices column by column, so
# array(m, c(3, rows/3, 3, cols/3)) places each block on dimensions 1 and 3.
# -----------------------------------------------------------------------------
block_mean_3x3 <- function(m) {
  nr <- ceiling(nrow(m) / 3) * 3; nc <- ceiling(ncol(m) / 3) * 3
  padded <- matrix(NA_real_, nr, nc)
  padded[seq_len(nrow(m)), seq_len(ncol(m))] <- m
  blocks <- aperm(array(padded, c(3, nr / 3, 3, nc / 3)), c(1, 3, 2, 4)) # 3 x 3 x block rows x block cols
  dim(blocks) <- c(9, (nr / 3) * (nc / 3))           # one column per block
  means <- colMeans(blocks, na.rm = TRUE)            # NaN where all nine are missing
  means[is.nan(means)] <- NA
  matrix(means, nr / 3, nc / 3)
}

# -----------------------------------------------------------------------------
# keep_off_nodata(): a rounded mean that happens to equal the NoData value
# would be read back as empty; it is moved one step into the valid range.
# -----------------------------------------------------------------------------
keep_off_nodata <- function(values, nodata_value) {
  step <- if (nodata_value <= 0) 1 else -1           # NoData at the bottom (0) or top (e.g. 65535)
  values[!is.na(values) & values == nodata_value] <- nodata_value + step
  values
}

# write_options(): tiled, lossless, large-file-safe GeoTIFF
write_options <- function(settings) {
  c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",  # PREDICTOR=2 shrinks integer imagery
    paste0("BLOCKXSIZE=", settings$tile_size), paste0("BLOCKYSIZE=", settings$tile_size),
    "BIGTIFF=IF_SAFER")                              # files over 4 GB need BigTIFF
}

# -----------------------------------------------------------------------------
# write_first_two(): one pass through the original, strip by strip, writing
# the 0.3 m copy and the 0.9 m average together. A strip is 3 tiles tall, so
# it fills exactly three tile rows at 0.3 m and one at 0.9 m; whole tiles are
# written once each, which keeps compressed files compact.
# -----------------------------------------------------------------------------
write_first_two <- function(original, path_03, path_09, nodata_value, datatype, settings) {
  factor     <- settings$aggregation_factor
  strip_rows <- factor * settings$tile_size          # 384 rows at 0.3 m = 128 rows at 0.9 m
  is_integer <- !grepl("^FLT", datatype)             # integer imagery is rounded after averaging
  
  copy <- rast(original)                             # empty raster with the original's grid
  names(copy) <- names(original)
  coarse <- rast(nrows = ceiling(nrow(original) / factor), ncols = ceiling(ncol(original) / factor),
                 nlyrs = nlyr(original), crs = crs(original),
                 xmin = xmin(original), ymax = ymax(original),
                 xmax = xmin(original) + ceiling(ncol(original) / factor) * factor * res(original)[1],
                 ymin = ymax(original) - ceiling(nrow(original) / factor) * factor * res(original)[2])
  names(coarse) <- names(original)
  
  writeStart(copy, path_03, overwrite = TRUE, datatype = datatype,
             NAflag = nodata_value, gdal = write_options(settings))
  writeStart(coarse, path_09, overwrite = TRUE, datatype = datatype,
             NAflag = nodata_value, gdal = write_options(settings))
  readStart(original)
  for (first_row in seq(1, nrow(original), by = strip_rows)) {
    n <- min(strip_rows, nrow(original) - first_row + 1)
    block <- readValues(original, row = first_row, nrows = n, col = 1,
                        ncols = ncol(original), mat = TRUE) # cells x bands, row by row
    writeValues(copy, block, start = first_row, nrows = n) # 0.3 m: the original values, unchanged
    
    values <- array(NA_real_, c(n, ncol(original), nlyr(original)))
    for (b in seq_len(nlyr(original))) values[, , b] <- matrix(block[, b], nrow = n, byrow = TRUE)
    rm(block)
    values <- restore_partial_nodata(values, nodata_value)
    cells <- sapply(seq_len(nlyr(original)), function(b) {
      m <- block_mean_3x3(values[, , b])
      if (is_integer) m <- floor(m + 0.5)           # round halves up, as in write_next()
      as.vector(t(keep_off_nodata(m, nodata_value)))  # row by row
    })
    writeValues(coarse, cells, start = (first_row - 1) / factor + 1, nrows = ceiling(n / factor))
  }
  readStop(original)
  writeStop(copy)
  writeStop(coarse)
}

# -----------------------------------------------------------------------------
# write_next(): one further 3 x 3 averaging with terra (0.9 -> 2.7,
# 2.7 -> 8.1). Starting from the previous step is exact, because the mean of
# equal-sized block means is the overall mean.
# -----------------------------------------------------------------------------
write_next <- function(previous, path, nodata_value, datatype, settings) {
  coarser <- aggregate(previous, fact = settings$aggregation_factor, fun = "mean", na.rm = TRUE)
  if (!grepl("^FLT", datatype)) coarser <- floor(coarser + 0.5) # round halves up
  coarser <- classify(coarser, cbind(nodata_value,   # mean equal to NoData -> one step inside
                                     nodata_value + if (nodata_value <= 0) 1 else -1))
  writeRaster(coarser, path, overwrite = TRUE, datatype = datatype,
              NAflag = nodata_value, gdal = write_options(settings))
}

# =============================================================================
# Main
# =============================================================================
dir.create(settings$folder_imagery, recursive = TRUE, showWarnings = FALSE)
terraOptions(progress = 0)
setGDALconfig("GDAL_CACHEMAX", as.character(settings$gdal_cache_mb))
catalogue <- list(); samples <- list()

for (site in names(settings$images)) {
  original <- rast(settings$images[[site]])
  stopifnot(nlyr(original) == length(settings$band_names))  # must be the 8-band product
  stopifnot(isTRUE(all.equal(res(original)[1], settings$resolutions[1], tolerance = 1e-6))) # must be 0.3 m
  stopifnot(same.crs(original, settings$working_crs))       # must already be in UTM 35S
  names(original) <- settings$band_names
  
  nodata_value <- if (is.null(settings$nodata_value)) file_nodata(settings$images[[site]]) else settings$nodata_value
  if (!is.finite(nodata_value)) stop(site, ": the file stores no NoData value; set settings$nodata_value")
  NAflag(original) <- nodata_value
  datatype <- datatype(original)[1]                  # e.g. "INT2U" for 16-bit imagery
  message(site, ": ", nlyr(original), " bands, ", datatype, ", NoData = ", nodata_value)
  
  paths <- file.path(settings$folder_imagery,        # every resolution, 0.3 m included
                     paste0(site, "_", resolution_label(settings$resolutions), ".tif"))
  
  if (all(file.exists(paths))) {                     # already done: skip (delete files to redo)
    message(site, ": images exist, skipped")
  } else {
    write_first_two(original, paths[1], paths[2], nodata_value, datatype, settings)
    message(site, ": written ", paths[1], " and ", paths[2])
    for (i in 3:length(paths)) {
      write_next(rast(paths[i - 1]), paths[i], nodata_value, datatype, settings)
      message(site, ": written ", paths[i])
    }
  }
  
  for (i in seq_along(paths)) {                      # record every image for later scripts
    image <- rast(paths[i])
    catalogue[[length(catalogue) + 1]] <- data.frame(
      site = site, resolution = settings$resolutions[i], path = paths[i],
      rows = nrow(image), cols = ncol(image), bands = nlyr(image),
      band_names = paste(settings$band_names, collapse = " "),
      xmin = xmin(image), ymax = ymax(image),        # top-left corner (anchors the block grid)
      datatype = datatype, nodata_value = nodata_value, crs = settings$working_crs)
    
    # Regular sample of pixels for band statistics (no labels involved)
    set.seed(2026L + i)
    values <- spatSample(image, min(settings$sample_cells, ncell(image)), method = "regular")
    samples[[length(samples) + 1]] <- cbind(resolution = settings$resolutions[i], values)
  }
}

catalogue <- do.call(rbind, catalogue)
write.csv(catalogue, file.path(settings$folder_imagery, "imagery_catalogue.csv"), row.names = FALSE)

# Band statistics per resolution, both sites pooled: inputs are later
# standardised as (value - mean) / sd so every band carries a similar scale.
# They use image values only, never labels, so no reference data leak in.

samples <- do.call(rbind, samples)
band_statistics <- do.call(rbind, lapply(settings$resolutions, function(r) {
  s <- samples[samples$resolution == r, settings$band_names]
  data.frame(resolution = r, band = settings$band_names, band_index = seq_along(settings$band_names),
             mean = sapply(s, mean, na.rm = TRUE), sd = sapply(s, sd, na.rm = TRUE))
}))
write.csv(band_statistics, file.path(settings$folder_imagery, "band_statistics.csv"), row.names = FALSE)

print(catalogue[, c("site", "resolution", "rows", "cols", "bands", "datatype")])
print(band_statistics[band_statistics$resolution == settings$resolutions[1], ])