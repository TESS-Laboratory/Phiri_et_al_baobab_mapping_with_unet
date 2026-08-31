#-------------------------------------------------------------------------------
# Introduction

# This script runs a trained U-Net model over a full landscape raster in overlapping
# tiles, stitches the overlaps, and writes a probability map plus a binary map.
# The script must be run per image.
#-------------------------------------------------------------------------------

library(torch)
library(terra)
library(torchvision)
library(luz)
library(tidyverse)

#-------------------------------------------------------------------------------
# U-Net architecture
#-------------------------------------------------------------------------------
double_conv_block <- nn_module(
  initialize = function(in_channels, out_channels) {
    self$conv1 <- nn_conv2d(in_channels,  out_channels, kernel_size = 3, padding = 1)
    self$bn1   <- nn_batch_norm2d(out_channels)
    self$conv2 <- nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self$bn2   <- nn_batch_norm2d(out_channels)
  },
  forward = function(x) {
    x <- nnf_relu(self$bn1(self$conv1(x)))
    x <- nnf_relu(self$bn2(self$conv2(x)))
    x
  }
)

three_layer_conv_block <- nn_module(
  initialize = function(in_channels, out_channels) {
    self$conv1 <- nn_conv2d(in_channels,  out_channels, kernel_size = 3, padding = 1)
    self$bn1   <- nn_batch_norm2d(out_channels)
    self$conv2 <- nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self$bn2   <- nn_batch_norm2d(out_channels)
    self$conv3 <- nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self$bn3   <- nn_batch_norm2d(out_channels)
  },
  forward = function(x) {
    x <- nnf_relu(self$bn1(self$conv1(x)))
    x <- nnf_relu(self$bn2(self$conv2(x)))
    x <- nnf_relu(self$bn3(self$conv3(x)))
    x
  }
)

# in_channels has no default here on purpose: n_input_bands (set once, above)
# is the single source of truth, passed in via set_hparams() below
unet_256 <- nn_module(
  initialize = function(in_channels, num_classes = 1) {
    self$down1 <- double_conv_block(in_channels, 64)
    self$pool1 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down2 <- double_conv_block(64,  128)
    self$pool2 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down3 <- double_conv_block(128, 256)
    self$pool3 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down4 <- double_conv_block(256, 512)
    self$pool4 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    
    self$center <- double_conv_block(512, 1024)
    
    self$up4 <- three_layer_conv_blockk(1024 + 512, 512)
    self$up3 <- three_layer_conv_block(512 + 256, 256)
    self$up2 <- three_layer_conv_block(256 + 128,  128)
    self$up1 <- three_layer_conv_block(128 + 64,  64)
    
    self$classify <- nn_conv2d(64, num_classes, kernel_size = 1)
  },
  
  forward = function(x) {
    d1  <- self$down1(x)
    d2  <- self$down2(self$pool1(d1))
    d3  <- self$down3(self$pool2(d2))
    d4  <- self$down4(self$pool3(d3))
    
    ctr <- self$center(self$pool4(d4))
    
    up <- function(tensor, skip)
      nnf_interpolate(tensor, scale_factor = 2, mode = "nearest") |>
      (\(t) torch_cat(list(skip, t), dim = 2))()
    
    u4 <- self$up4(up(ctr, d4))
    u3 <- self$up3(up(u4,  d3))
    u2 <- self$up2(up(u3,  d2))
    u1 <- self$up1(up(u2,  d1))
    
    self$classify(u1)
  }
)

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
# must match the training run this checkpoint came from
site_key      <- "kremetart"   # must match a key in the evaluation script's reference_lookup
site_resolution_tag <- "feb_0.3m"
run_id        <- "300426"
selected_fold <- 3L            # <-- set to your best-performing fold from cv_results_summary.csv
n_input_bands <- 8L            # set to 16L when predicting from the Feb+July stack

training_output_dir <- "Phiri_et_al_baobab_mapping_with_unet/256_pixels_results/Six_Fold/feb/"

# season and resolution, split out of site_resolution_tag so the output name

season_key       <- sub("^(feb_july|feb|july)_.*$", "\\1", site_resolution_tag)
resolution_label <- sub("^(feb_july|feb|july)_([0-9.]+)m$", "\\2", site_resolution_tag)

config <- list(
  landscape_path  = "My_Dataset/kremetart_feb_image_8bit.tif",
  model_path      = file.path(training_output_dir,
                              sprintf("unet_%s_%s_fold%02d.pt", site_resolution_tag, run_id, selected_fold)),
  band_stats_path = file.path(training_output_dir,
                              sprintf("%s_stats_fold%02d.rds", site_resolution_tag, selected_fold)),
  output_dir      = "outputs",
  
  output_prefix   = sprintf("%s_baobab_map_%s_%sm", site_key, season_key, resolution_label),
  
  patch_size   = 256L,   # must match the patch size used for training
  crop_margin  = 32L,    
  overlap_frac = 0.5,    # overlap fraction 
  batch_size   = 8L,
  threshold    = 0.50,
  rgb_bands    = c(5, 3, 2),   # true-colour bands, for the preview image only
  
  save_probability = TRUE,
  save_binary       = TRUE
)

dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)

#-------------------------------------------------------------------------------
# Load model
#-------------------------------------------------------------------------------
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
cat(sprintf("[%s] Running on: %s\n", Sys.time(), device$type))

band_stats <- readRDS(config$band_stats_path)
cat(sprintf("[%s] Band stats loaded (%d bands)\n", Sys.time(), length(band_stats$lo)))
stopifnot("n_input_bands does not match the loaded band_stats" = n_input_bands == length(band_stats$lo))

model <- unet_256(in_channels = n_input_bands, num_classes = 1L)
state <- torch_load(config$model_path)
if (is.null(state$model)) {
  stop("Checkpoint has no '$model' element. Found: ", paste(names(state), collapse = ", "))
}
model$load_state_dict(state$model)
model$to(device = device)
model$eval()
cat(sprintf("[%s] Model loaded: %s\n", Sys.time(), config$model_path))

# sanity check: one dummy tile through the model before touching real data
dummy <- torch_zeros(1L, n_input_bands, config$patch_size, config$patch_size,
                     dtype = torch_float(), device = device)
with_no_grad({ test_out <- model(dummy) })
stopifnot(dim(test_out)[2] == 1)
cat(sprintf("[%s] Model verified on a dummy tile\n", Sys.time()))

#-------------------------------------------------------------------------------
# Inference over one landscape raster, tiled with overlap-blending
#-------------------------------------------------------------------------------
predict_landscape <- function(img_path, model, band_stats, config, device) {
  
  cat(sprintf("[%s] Loading raster: %s\n", Sys.time(), basename(img_path)))
  img       <- terra::rast(img_path)
  img_array <- as.array(img)
  img_array <- aperm(img_array, c(3, 1, 2))   # (row,col,band) -> (band,row,col), channel-first for torch
  
  nchannels <- dim(img_array)[1]
  nrows     <- dim(img_array)[2]
  ncols     <- dim(img_array)[3]
  
  if (nchannels != length(band_stats$lo)) {
    stop("Image has ", nchannels, " bands but band_stats has ",
         length(band_stats$lo), " - mismatch.")
  }
  patch_size <- config$patch_size
  if (nrows < patch_size || ncols < patch_size) {
    stop("Image (", nrows, " x ", ncols, ") is smaller than patch_size (", patch_size, ").")
  }
  
  cat(sprintf("[%s] Image: %d bands x %d rows x %d cols\n", Sys.time(), nchannels, nrows, ncols))
  
  # stretch each band using the SAME limits computed at training time
  cat(sprintf("[%s] Normalising bands\n", Sys.time()))
  for (band_idx in seq_len(nchannels)) {
    lo <- band_stats$lo[band_idx]
    hi <- band_stats$hi[band_idx]
    img_array[band_idx, , ] <- pmax(0, pmin(1, (img_array[band_idx, , ] - lo) / (hi - lo + 1e-6)))
  }
  img_array[is.na(img_array)] <- 0
  
  # tile starts, with a final tile forced flush to each edge for full coverage
  crop_margin <- config$crop_margin
  stride      <- max(1L, as.integer(patch_size * config$overlap_frac))
  
  row_starts <- seq(1L, max(1L, nrows - patch_size + 1L), by = stride)
  col_starts <- seq(1L, max(1L, ncols - patch_size + 1L), by = stride)
  if (tail(row_starts, 1L) + patch_size - 1L < nrows) row_starts <- c(row_starts, nrows - patch_size + 1L)
  if (tail(col_starts, 1L) + patch_size - 1L < ncols) col_starts <- c(col_starts, ncols - patch_size + 1L)
  
  total_tiles <- length(row_starts) * length(col_starts)
  cat(sprintf("[%s] Tiling: %d x %d = %d tiles, stride %d\n",
              Sys.time(), length(row_starts), length(col_starts), total_tiles, stride))
  
  pred_sum   <- array(0,  dim = c(nrows, ncols))   # running sum of probabilities per pixel
  pred_count <- array(0L, dim = c(nrows, ncols))   # how many tiles contributed to each pixel
  
  # writes one batch's predictions into pred_sum/pred_count
  
  flush_batch <- function(tiles, metas) {
    batch    <- torch_stack(tiles, dim = 1)$to(device = device)
    out_prob <- torch_sigmoid(model(batch))
    src      <- as.array(out_prob$cpu())   
    
    for (b in seq_along(metas)) {
      r  <- metas[[b]]$r
      cc <- metas[[b]]$c
      
      local_r_lo <- if (r == 1L) 1L else crop_margin + 1L
      local_r_hi <- if (r + patch_size - 1L >= nrows) patch_size else patch_size - crop_margin
      local_c_lo <- if (cc == 1L) 1L else crop_margin + 1L
      local_c_hi <- if (cc + patch_size - 1L >= ncols) patch_size else patch_size - crop_margin
      
      gr1 <- r  + local_r_lo - 1L
      gr2 <- r  + local_r_hi - 1L
      gc1 <- cc + local_c_lo - 1L
      gc2 <- cc + local_c_hi - 1L
      
      pred_sum[gr1:gr2, gc1:gc2] <<- pred_sum[gr1:gr2, gc1:gc2] +
        src[b, 1, local_r_lo:local_r_hi, local_c_lo:local_c_hi]
      pred_count[gr1:gr2, gc1:gc2] <<- pred_count[gr1:gr2, gc1:gc2] + 1L
    }
  }
  
  cat(sprintf("[%s] Running inference\n", Sys.time()))
  start_time <- Sys.time()
  
  with_no_grad({
    tile_buf <- list(); meta_buf <- list(); tile_num <- 0L
    
    for (r in row_starts) {
      for (cc in col_starts) {
        tile_num <- tile_num + 1L
        
        if (tile_num %% max(1L, total_tiles %/% 10L) == 0L) {
          pct     <- 100 * tile_num / total_tiles
          elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
          eta     <- elapsed * (total_tiles / tile_num - 1)
          cat(sprintf("[%s]   %d/%d tiles (%.0f%%) - ETA: %.0f sec\n",
                      Sys.time(), tile_num, total_tiles, pct, eta))
        }
        
        patch <- img_array[, r:(r + patch_size - 1L), cc:(cc + patch_size - 1L)]
        tile_buf <- c(tile_buf, list(torch_tensor(patch, dtype = torch_float())))
        meta_buf <- c(meta_buf, list(list(r = r, c = cc)))
        
        if (length(tile_buf) == config$batch_size ||
            (r == tail(row_starts, 1L) && cc == tail(col_starts, 1L))) {
          flush_batch(tile_buf, meta_buf)
          tile_buf <- list(); meta_buf <- list()
        }
      }
    }
  })
  
  total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  cat(sprintf("[%s] Inference complete (%.1f min)\n", Sys.time(), total_time))
  
  cat(sprintf("[%s] Averaging overlapping predictions\n", Sys.time()))
  pred_avg <- pred_sum / pmax(pred_count, 1L)
  pred_avg[is.nan(pred_avg)] <- 0
  
  out_rast <- terra::rast(img[[1]])           
  terra::values(out_rast) <- as.vector(t(pred_avg))   
  names(out_rast) <- "baobab_probability"
  
  out_rast
}

#-------------------------------------------------------------------------------
# Run inference and write outputs
#-------------------------------------------------------------------------------

prob_rast <- predict_landscape(
  img_path   = config$landscape_path,
  model      = model,
  band_stats = band_stats,
  config     = config,
  device     = device
)

if (config$save_probability) {
  prob_path <- file.path(config$output_dir, paste0(config$output_prefix, "_probability.tif"))
  terra::writeRaster(prob_rast, prob_path, datatype = "FLT4S", overwrite = TRUE,
                     gdal = c("COMPRESS=LZW", "TILED=YES"))
  cat(sprintf("[%s] Saved probability: %s\n", Sys.time(), prob_path))
}

if (config$save_binary) {
  cat(sprintf("[%s] Thresholding at %.2f\n", Sys.time(), config$threshold))
  binary_rast <- terra::ifel(prob_rast >= config$threshold, 1, 0)
  names(binary_rast) <- "baobab_binary"
  
  binary_path <- file.path(config$output_dir, paste0(config$output_prefix, "_binary.tif"))
  terra::writeRaster(binary_rast, binary_path, datatype = "INT1U", overwrite = TRUE,
                     gdal = c("COMPRESS=LZW", "TILED=YES"))
  cat(sprintf("[%s] Saved binary: %s\n", Sys.time(), binary_path))
}

#-------------------------------------------------------------------------------
# Preview: RGB vs predicted probability, centre 20% of the landscape
#-------------------------------------------------------------------------------

cat(sprintf("\n[%s] Generating preview\n", Sys.time()))
img      <- terra::rast(config$landscape_path)
crop_ext <- terra::ext(prob_rast)
cx_min <- crop_ext$xmin + (crop_ext$xmax - crop_ext$xmin) * 0.4
cx_max <- crop_ext$xmin + (crop_ext$xmax - crop_ext$xmin) * 0.6
cy_min <- crop_ext$ymin + (crop_ext$ymax - crop_ext$ymin) * 0.4
cy_max <- crop_ext$ymin + (crop_ext$ymax - crop_ext$ymin) * 0.6
sub_ext <- terra::ext(cx_min, cx_max, cy_min, cy_max)

preview_path <- file.path(config$output_dir, paste0(config$output_prefix, "_preview.png"))
png(preview_path, width = 1600, height = 800, res = 100)
op <- par(mfrow = c(1, 2), mar = c(2, 2, 3, 2))
terra::plotRGB(terra::crop(img[[config$rgb_bands]], sub_ext), stretch = "lin", main = "RGB (centre 20%)")
terra::plot(terra::crop(prob_rast, sub_ext), col = hcl.colors(20, "Inferno"), main = "Baobab probability")
par(op)
dev.off()
cat(sprintf("[%s] Preview saved: %s\n", Sys.time(), preview_path))

cat("\nDONE.\n")