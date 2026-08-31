#-------------------------------------------------------------------------------
# Introduction

# Six-fold spatial cross-validation for a U-Net baobab segmentation model.
# One model is trained per fold; the best checkpoint (by valid_loss) is kept.

# Configured for: Kremetart, February + July combined (16-band stack), 0.3 m.
# Note: This script run once per image product, update the paths.
#-------------------------------------------------------------------------------
# Load the required packages

library(torch)
library(terra)
library(torchvision)
library(luz)
library(tidyverse)

#-------------------------------------------------------------------------------
# Set-up
#-------------------------------------------------------------------------------
n_folds     <- 6L
base_seed   <- 1000L          
n_input_bands <- 16L          

site_resolution_tag <- "feb_july_0.3m"   
run_id <- "300426"                       

output_dir <- "Phiri_et_al_baobab_mapping_with_unet/256_pixels_results/Six_Fold/feb_july/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# record package versions alongside the outputs, for later reproducibility
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))

#-------------------------------------------------------------------------------
# Dataset
#-------------------------------------------------------------------------------
chip_dataset <- dataset(
  name = "WorldViewChips",
  
  initialize = function(image_files, mask_files,
                        indices = NULL, augment = FALSE,
                        band_stats = NULL) {
    stopifnot(length(image_files) == length(mask_files))
    
    if (!is.null(indices)) {
      image_files <- image_files[indices]
      mask_files  <- mask_files[indices]
    }
    
    self$image_files <- image_files
    self$mask_files  <- mask_files
    self$augment     <- augment
    self$band_stats  <- band_stats
  },
  
  .getitem = function(i) {
    img  <- as.array(terra::rast(self$image_files[[i]]))
    mask <- as.array(terra::rast(self$mask_files[[i]]))
    
    img_t <- torch_tensor(img)$permute(c(3L, 1L, 2L))$float()
    
    # stretch each band to 0-1 using this fold's training-only band statistics
    if (!is.null(self$band_stats)) {
      for (band_idx in seq_len(img_t$shape[1])) {
        lo <- self$band_stats$lo[band_idx]
        hi <- self$band_stats$hi[band_idx]
        img_t[band_idx, , ] <- torch_clamp((img_t[band_idx, , ] - lo) / (hi - lo + 1e-6), 0, 1)
      }
    } else {
      img_t <- torch_clamp(img_t / 255, 0, 1)
    }
    
    mask[is.na(mask)] <- 0
    mask_t <- torch_tensor(mask)$permute(c(3L, 1L, 2L))$float()
    mask_t <- (mask_t > 0.5)$to(dtype = torch_float())
    
    # random flips, rotation and brightness jitter, training data only
    if (self$augment) {
      if (runif(1) > 0.5) { img_t <- transform_hflip(img_t); mask_t <- transform_hflip(mask_t) }
      if (runif(1) > 0.5) { img_t <- transform_vflip(img_t); mask_t <- transform_vflip(mask_t) }
      
      k <- sample(0:3, 1)
      if (k > 0) {
        img_t  <- torch_rot90(img_t,  k = k, dims = c(2, 3))
        mask_t <- torch_rot90(mask_t, k = k, dims = c(2, 3))
      }
      
      if (runif(1) > 0.5) {
        brightness <- runif(1, 0.9, 1.1)
        img_t <- torch_clamp(img_t * brightness, 0, 1)
      }
    }
    
    list(x = img_t, y = mask_t)
  },
  
  .length = function() length(self$image_files)
)

#-------------------------------------------------------------------------------
# File path to both masks and chips
#-------------------------------------------------------------------------------
setwd("C:/Users/cp844/University of Exeter/TESSLab - Chafika Phiri Data/Objective 3 Manuscript/Training Patches")

positive_image_dir <- "Kremetart_feb_july_patches/0.3m/images"
positive_mask_dir  <- "Kremetart_feb_july_patches/0.3m/masks"
negative_image_dir <- "Kremetart_Negative_patches/feb_july/0.3m/images"
negative_mask_dir  <- "Kremetart_Negative_patches/feb_july/0.3m/masks"

# sort each folder on its own, so every image still lines up with its own mask
positive_image_files <- sort(list.files(positive_image_dir, pattern = "\\.tif$", full.names = TRUE))
positive_mask_files  <- sort(list.files(positive_mask_dir,  pattern = "\\.tif$", full.names = TRUE))
negative_image_files <- sort(list.files(negative_image_dir, pattern = "\\.tif$", full.names = TRUE))
negative_mask_files  <- sort(list.files(negative_mask_dir,  pattern = "\\.tif$", full.names = TRUE))

stopifnot(length(positive_image_files) == length(positive_mask_files))
stopifnot(length(negative_image_files) == length(negative_mask_files))

image_files <- c(positive_image_files, negative_image_files)
mask_files  <- c(positive_mask_files,  negative_mask_files)
stopifnot(length(image_files) == length(mask_files))
stopifnot(length(image_files) > 0)   # fail clearly here, not later inside kmeans

cat("Positive chips:", length(positive_image_files), "\n")
cat("Negative chips:", length(negative_image_files), "\n")
cat("Total chips:   ", length(image_files), "\n")

#-------------------------------------------------------------------------------
# Band statistics (computed per fold, training chips only, to avoid leakage)
#-------------------------------------------------------------------------------
compute_band_stats <- function(files, n_sample = 100, bands = 8) {
  files <- sample(files, min(n_sample, length(files)))
  
  all_vals <- vector("list", bands)
  for (f in files) {
    arr <- as.array(terra::rast(f))
    for (b in seq_len(bands)) {
      all_vals[[b]] <- c(all_vals[[b]], as.vector(arr[, , b]))
    }
  }
  
  lo <- sapply(all_vals, quantile, probs = 0.02, na.rm = TRUE)
  hi <- sapply(all_vals, quantile, probs = 0.98, na.rm = TRUE)
  
  list(lo = lo, hi = hi)
}

#-------------------------------------------------------------------------------
# Loss function: binary cross-entropy plus Dice
#-------------------------------------------------------------------------------
bce_dice_loss <- function(pred, target) {
  bce <- nnf_binary_cross_entropy_with_logits(pred, target)
  
  pred <- torch_sigmoid(pred)
  smooth <- 1e-6
  
  intersection <- (pred * target)$sum()
  dice <- (2 * intersection + smooth) / (pred$sum() + target$sum() + smooth)
  
  bce + (1 - dice)
}

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
    
    self$up4 <- three_layer_conv_block(1024 + 512, 512)
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
# Metrics
#-------------------------------------------------------------------------------
# shared helper: updates true/false positive/negative counts for one batch
metric_update_counts <- function(self, preds, target) {
  p <- (torch_sigmoid(preds) > 0.5)$to(dtype = torch_float())
  t <- (target > 0.5)$to(dtype = torch_float())
  self$tp <- self$tp + torch_sum( p *  t)$item()
  self$fp <- self$fp + torch_sum( p * (1 - t))$item()
  self$fn <- self$fn + torch_sum((1 - p) *  t)$item()
  self$tn <- self$tn + torch_sum((1 - p) * (1 - t))$item()
}

luz_metric_dice <- luz_metric(
  abbrev = "Dice",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0 },
  update = function(preds, target) {
    p <- (torch_sigmoid(preds) > 0.5)$to(dtype = torch_float())
    t <- (target > 0.5)$to(dtype = torch_float())
    self$tp <- self$tp + torch_sum( p *  t)$item()
    self$fp <- self$fp + torch_sum( p * (1 - t))$item()
    self$fn <- self$fn + torch_sum((1 - p) *  t)$item()
  },
  compute = function() (2 * self$tp) / (2 * self$tp + self$fp + self$fn + 1e-8)
)

luz_metric_iou <- luz_metric(
  abbrev = "IoU",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0 },
  update = function(preds, target) {
    p <- (torch_sigmoid(preds) > 0.5)$to(dtype = torch_float())
    t <- (target > 0.5)$to(dtype = torch_float())
    self$tp <- self$tp + torch_sum( p *  t)$item()
    self$fp <- self$fp + torch_sum( p * (1 - t))$item()
    self$fn <- self$fn + torch_sum((1 - p) *  t)$item()
  },
  compute = function() self$tp / (self$tp + self$fp + self$fn + 1e-8)
)

luz_metric_precision <- luz_metric(
  abbrev = "Prec",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update  = function(preds, target) metric_update_counts(self, preds, target),
  compute = function() self$tp / (self$tp + self$fp + 1e-8)
)

luz_metric_recall <- luz_metric(
  abbrev = "Rec",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update  = function(preds, target) metric_update_counts(self, preds, target),
  compute = function() self$tp / (self$tp + self$fn + 1e-8)
)

luz_metric_f1 <- luz_metric(
  abbrev = "F1",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update  = function(preds, target) metric_update_counts(self, preds, target),
  compute = function() {
    prec <- self$tp / (self$tp + self$fp + 1e-8)
    rec  <- self$tp / (self$tp + self$fn + 1e-8)
    (2 * prec * rec) / (prec + rec + 1e-8)
  }
)

luz_metric_accuracy <- luz_metric(
  abbrev = "Acc",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update  = function(preds, target) metric_update_counts(self, preds, target),
  compute = function() (self$tp + self$tn) / (self$tp + self$tn + self$fp + self$fn + 1e-8)
)

#-------------------------------------------------------------------------------
# Six-fold spatial cross-validation: cluster chip centroids into folds
#-------------------------------------------------------------------------------
n_chips <- length(image_files)
cat("Total chips:", n_chips, "\n")

get_chip_centre <- function(file_path) {
  raster_extent <- terra::ext(terra::rast(file_path))
  c(x = (raster_extent$xmin + raster_extent$xmax) / 2,
    y = (raster_extent$ymin + raster_extent$ymax) / 2)
}

chip_centres <- t(vapply(image_files, get_chip_centre, numeric(2)))
colnames(chip_centres) <- c("x", "y")

set.seed(base_seed)   # controls the k-means cluster assignment only
spatial_clusters <- kmeans(chip_centres, centers = n_folds, nstart = 25)

fold_table <- data.frame(
  idx  = seq_len(n_chips),
  x    = chip_centres[, "x"],
  y    = chip_centres[, "y"],
  fold = spatial_clusters$cluster
)
cat("Chips per fold:\n"); print(table(fold_table$fold))

#-------------------------------------------------------------------------------
# Fold loop: train, checkpoint the best epoch, evaluate that checkpoint
#-------------------------------------------------------------------------------
cv_results <- vector("list", n_folds)

for (fold in seq_len(n_folds)) {
  
  cat("\n===== Six-fold spatial CV -- fold", fold, "of", n_folds, "=====\n")
  
  # each fold gets its own seed, independent of what earlier folds consumed
  fold_seed <- base_seed + fold
  set.seed(fold_seed)
  torch_manual_seed(fold_seed)
  
  train_idx <- fold_table$idx[fold_table$fold != fold]
  val_idx   <- fold_table$idx[fold_table$fold == fold]
  cat(sprintf("  Training chips: %d  |  Validation chips: %d\n",
              length(train_idx), length(val_idx)))
  
  band_stats <- compute_band_stats(image_files[train_idx], bands = n_input_bands)
  print(band_stats)
  
  band_stats_path <- file.path(output_dir, sprintf("%s_stats_fold%02d.rds", site_resolution_tag, fold))
  saveRDS(band_stats, band_stats_path)
  
  train_ds <- chip_dataset(image_files, mask_files,
                           indices = train_idx, augment = TRUE,
                           band_stats = band_stats)
  val_ds   <- chip_dataset(image_files, mask_files,
                           indices = val_idx, augment = FALSE,
                           band_stats = band_stats)
  
  train_dl <- dataloader(train_ds, batch_size = 8, shuffle = TRUE,  num_workers = 0)
  val_dl   <- dataloader(val_ds,   batch_size = 8, shuffle = FALSE, num_workers = 0)
  
  # visual check: four random validation chips and their masks for this fold
  n_show <- 4
  rgb_bands <- c(5, 3, 2)
  sample_idx <- sample(seq_len(length(val_ds)), n_show)
  
  tensor_to_rgb <- function(t, bands) {
    arr <- as.array(t)
    rgb <- arr[bands, , , drop = FALSE]
    rgb <- aperm(rgb, c(2, 3, 1))
    pmin(pmax(rgb, 0), 1)
  }
  tensor_to_mask <- function(t) as.array(t)[1, , ]
  
  op <- par(mfrow = c(n_show, 2), mar = c(1, 1, 2, 1), oma = c(0, 0, 2, 0))
  for (i in sample_idx) {
    pair <- val_ds$.getitem(i)
    rgb  <- tensor_to_rgb(pair$x, rgb_bands)
    msk  <- tensor_to_mask(pair$y)
    
    plot.new()
    plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1, xaxs = "i", yaxs = "i")
    rasterImage(rgb, 0, 0, 1, 1)
    title(main = sprintf("Image idx %d (bands %s)", i, paste(rgb_bands, collapse = ",")), cex.main = 0.8)
    
    msk_rgb <- array(0, dim = c(dim(msk), 3))
    msk_rgb[, , 1] <- msk
    msk_rgb[, , 2] <- msk
    
    plot.new()
    plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1, xaxs = "i", yaxs = "i")
    rasterImage(msk_rgb, 0, 0, 1, 1)
    title(main = sprintf("Mask idx %d (pos pixels: %d)", i, sum(msk > 0.5)), cex.main = 0.8)
  }
  mtext(sprintf("%d random val_ds chips", n_show), outer = TRUE, cex = 1, font = 2)
  par(op)
  
  ckpt_path <- file.path(output_dir, sprintf("unet_%s_%s_fold%02d.pt", site_resolution_tag, run_id, fold))
  log_path  <- file.path(output_dir, sprintf("unet_%s_%s_fold%02d_log.csv", site_resolution_tag, run_id, fold))
  
  # trains for the 100 epochs
  fitted <- unet_256 |>
    setup(
      loss      = bce_dice_loss,
      optimizer = optim_adam,
      metrics   = list(
        luz_metric_dice(),
        luz_metric_iou(),
        luz_metric_precision(),
        luz_metric_recall(),
        luz_metric_f1(),
        luz_metric_accuracy()
      )
    ) |>
    set_hparams(in_channels = n_input_bands, num_classes = 1L) |>
    set_opt_hparams(lr = 3e-4) |>
    fit(
      data       = train_dl,
      valid_data = val_dl,
      epochs     = 100,
      callbacks  = list(
        luz_callback_model_checkpoint(
          path           = ckpt_path,
          monitor        = "valid_loss",
          mode           = "min",
          save_best_only = TRUE
        ),
        luz_callback_csv_logger(log_path)
      )
    )
  
  plot(fitted)
  
  # fitted reflects the LAST epoch trained; reload the saved best checkpoint before evaluating
  best_model <- tryCatch(
    luz_load_checkpoint(fitted, ckpt_path),
    error = function(e) {
      warning("Fold ", fold, ": could not reload best checkpoint (", conditionMessage(e),
              "); evaluating the last epoch instead.")
      fitted
    }
  )
  
  evaluation <- best_model |> evaluate(data = val_dl)
  fold_metrics <- get_metrics(evaluation)
  fold_metrics$fold <- fold
  
  cv_results[[fold]] <- list(
    fold      = fold,
    seed      = fold_seed,
    ckpt_path = ckpt_path,
    metrics   = fold_metrics
  )
  
  cat("  Fold", fold, "validation metrics (best checkpoint):\n")
  print(fold_metrics)
}

#-------------------------------------------------------------------------------
# Save the cross-fold summary
#-------------------------------------------------------------------------------
saveRDS(cv_results, file.path(output_dir, "cv_results_summary.rds"))

all_fold_metrics <- do.call(rbind, lapply(cv_results, function(r) r$metrics))
write.csv(all_fold_metrics, file.path(output_dir, "cv_results_summary.csv"), row.names = FALSE)

cat("\nAll folds complete. Summary saved to", output_dir, "\n")