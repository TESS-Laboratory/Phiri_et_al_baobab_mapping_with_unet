#-------------------------------------------------------------------------------
# Introduction
#-------------------------------------------------------------------------------

# This script implements a six-fold cross-validation framework for training a 
# U-Net deep learning model to map African baobab trees using WorldView-3 
# high-resolution image chips. The workflow includes data loading, image 
# augmentation, loss function definition, U-Net architecture development, 
# model training, and evaluation.

# load the required packages

library(torch)
library(terra)
library(torchvision)
library(luz)
library(tidyverse)

# number of folds.

n_folds <- 6L

# 'output_dir' is the single folder where every fold writes its model
# checkpoint, its training log and its band-statistics file.

output_dir <- "Phiri_et_al_baobab_mapping_with_unet/256_pixels_results/Six_Fold/feb/"

# Create the folder if it does not already exist.
#   showWarnings = FALSE  -> stay quiet if the folder is already there
#   recursive    = TRUE   -> also create any missing parent folders

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------- Dataset

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
    
    if (!is.null(self$band_stats)) {
      for (c in seq_len(img_t$shape[1])) {
        lo <- self$band_stats$lo[c]
        hi <- self$band_stats$hi[c]
        img_t[c, , ] <- torch_clamp((img_t[c, , ] - lo) / (hi - lo + 1e-6), 0, 1)
      }
    } else {
      img_t <- torch_clamp(img_t / 255, 0, 1)
    }
    
    mask[is.na(mask)] <- 0
    mask_t <- torch_tensor(mask)$permute(c(3L, 1L, 2L))$float()
    mask_t <- (mask_t > 0.5)$to(dtype = torch_float())
    
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


# ----------------- file lists

# set the working directory

setwd("C:/Users/cp844/University of Exeter/TESSLab - Chafika Phiri Data/Objective 3 Manuscript/Training Patches")

image_dir <- "image patches/Kremetart_feb_polygons/new_feb_256_8bands/images"
mask_dir  <- "image patches/Kremetart_feb_polygons/new_feb_256_8bands/masks"

image_files <- sort(list.files(image_dir, pattern = "\\.tif$", full.names = TRUE))
mask_files  <- sort(list.files(mask_dir,  pattern = "\\.tif$", full.names = TRUE))

stopifnot(length(image_files) == length(mask_files))


# drop excess empty chips

#mask_summary <- data.frame(file = mask_files, pos_pixels = NA_integer_)

#for (i in seq_along(mask_files)) {
#  m <- as.array(terra::rast(mask_files[i]))
#  m <- m[,,1] > 0.5
#  mask_summary$pos_pixels[i] <- sum(m, na.rm = TRUE)
#}

#has_baobab <- mask_summary$pos_pixels > 0
#pos_files  <- which(has_baobab)
#neg_files  <- which(!has_baobab)

#n_pos        <- length(pos_files)
#n_neg_target <- round(n_pos * 0.40)

#set.seed(42)
#neg_keep <- sample(neg_files, n_neg_target)
#keep_idx <- sort(c(pos_files, neg_keep))

#image_files <- image_files[keep_idx]
#mask_files  <- mask_files[keep_idx]

#cat("Final dataset:", length(image_files), "chips\n")
#cat("  Positive:", n_pos, "\n")
#cat("  Negative:", n_neg_target,
#    sprintf("(%.1f%%)\n", 100 * n_neg_target / length(image_files)))


# Band statistics computation 

# Note: With cross-validation we instead call it
#       inside the fold loop, so each fold gets band statistics computed only
#       from that fold's own training chips (this avoids information leaking
#       from the validation chips into normalisation).

compute_band_stats <- function(files, n_sample = 100, bands = 8) {
  files <- sample(files, min(n_sample, length(files)))
  
  all_vals <- vector("list", bands)
  for (f in files) {
    arr <- as.array(terra::rast(f))
    for (b in seq_len(bands)) {
      all_vals[[b]] <- c(all_vals[[b]], as.vector(arr[,,b]))
    }
  }
  
  lo <- sapply(all_vals, quantile, probs = 0.02, na.rm = TRUE)
  hi <- sapply(all_vals, quantile, probs = 0.98, na.rm = TRUE)
  
  list(lo = lo, hi = hi)
}

# set-up the loss function combination of binary cross-entropy and dice loss


bce_dice_loss <- function(pred, target) {
  bce <- nnf_binary_cross_entropy_with_logits(pred, target)
  
  pred <- torch_sigmoid(pred)
  smooth <- 1e-6
  
  intersection <- (pred * target)$sum()
  dice <- (2 * intersection + smooth) /
    (pred$sum() + target$sum() + smooth)
  
  bce + (1 - dice)
}


# U-Net MODEL


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

triple_conv_block <- nn_module(
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

unet_256 <- nn_module(
  initialize = function(in_channels = 8, num_classes = 1) {
    self$down1 <- double_conv_block(in_channels, 32)
    self$pool1 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down2 <- double_conv_block(32,  64)
    self$pool2 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down3 <- double_conv_block(64, 128)
    self$pool3 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down4 <- double_conv_block(128, 256)
    self$pool4 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    
    self$center <- double_conv_block(256, 512)
    
    self$up4 <- triple_conv_block(512 + 256, 256)
    self$up3 <- triple_conv_block(256 + 128, 128)
    self$up2 <- triple_conv_block(128 +  64,  64)
    self$up1 <- triple_conv_block( 64 +  32,  32)
    
    self$classify <- nn_conv2d(32, num_classes, kernel_size = 1)
  },
  
  forward = function(x) {
    d1  <- self$down1(x)
    d2  <- self$down2(self$pool1(d1))
    d3  <- self$down3(self$pool2(d2))
    d4  <- self$down4(self$pool3(d3))
    
    ctr <- self$center(self$pool4(d4))
    
    up <- function(tensor, skip)
      nnf_interpolate(tensor, scale_factor = 2,
                      mode = "bilinear", align_corners = FALSE) |>
      (\(t) torch_cat(list(skip, t), dim = 2))()
    
    u4 <- self$up4(up(ctr, d4))
    u3 <- self$up3(up(u4,  d3))
    u2 <- self$up2(up(u3,  d2))
    u1 <- self$up1(up(u2,  d1))
    
    self$classify(u1)
  }
)

# Set-out metrics to montor during training

# updates the metric counts for true positives, false positives, true negatives, and false negatives

metric_update_counts <- function(self, preds, target) {
  p <- (torch_sigmoid(preds) > 0.5)$to(dtype = torch_float())
  t <- (target > 0.5)$to(dtype = torch_float())
  self$tp <- self$tp + torch_sum( p *  t)$item()
  self$fp <- self$fp + torch_sum( p * (1 - t))$item()
  self$fn <- self$fn + torch_sum((1 - p) *  t)$item()
  self$tn <- self$tn + torch_sum((1 - p) * (1 - t))$item()
}

# dice

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

# logs intersection over union (iou)

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

# logs precision

luz_metric_precision <- luz_metric(
  abbrev = "Prec",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update  = function(preds, target) metric_update_counts(self, preds, target),
  compute = function() self$tp / (self$tp + self$fp + 1e-8)
)

# logs recall

luz_metric_recall <- luz_metric(
  abbrev = "Rec",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update  = function(preds, target) metric_update_counts(self, preds, target),
  compute = function() self$tp / (self$tp + self$fn + 1e-8)
)

# logs f1-score

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

# logs accuracy

luz_metric_accuracy <- luz_metric(
  abbrev = "Acc",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update  = function(preds, target) metric_update_counts(self, preds, target),
  compute = function() (self$tp + self$tn) / (self$tp + self$tn + self$fp + self$fn + 1e-8)
)

# Six-fold Spatila Cross Validation (SCV)       

# We group every chip into one of six spatial clusters,
# so each chip belongs to exactly one fold. Folds are formed by geography:
# chips that sit near each other on the ground are into the same fold.
# This gives a fairer test, because the validation chips are spatially
# separated from the training chips.

# 'n_chips' is simply the total number of chips we have.

n_chips <- length(image_files)

# prints total number

cat("Total chips:", n_chips, "\n")

# returns the geographic centre point (x, y) of the chip.

get_chip_centre <- function(file_path) {
  
  raster_extent <- terra::ext(terra::rast(file_path))
 
  c(x = (raster_extent$xmin + raster_extent$xmax) / 2,
    
    y = (raster_extent$ymin + raster_extent$ymax) / 2)
}

# Apply 'get_chip_centre' to every image file.

chip_centres <- t(vapply(image_files, get_chip_centre, numeric(2)))

colnames(chip_centres) <- c("x", "y")  

# Set a seed 

set.seed(100)

# k-means clustering groups the chip centres into 'n_folds' spatial clusters.

spatial_clusters <- kmeans(chip_centres, centers = n_folds, nstart = 25)

fold_table <- data.frame(
  idx  = seq_len(n_chips),          # the chip's position in image_files / mask_files
  x    = chip_centres[, "x"],       # its centre x coordinate
  y    = chip_centres[, "y"],       # its centre y coordinate
  fold = spatial_clusters$cluster   # the fold number (1 to 6) it was assigned to
)

# prints how many chips ended up in each fold (they will not be perfectly equal).

cat("Chips per fold:\n"); print(table(fold_table$fold))

# Six fold validation loop           

# "cv_results" is an empty list with one slot per fold. Each slot will hold
# that fold's trained model and its validation metrics once the fold finishes.

cv_results <- vector("list", n_folds)

# Loop once per fold. On each pass, one fold becomes the validation set and
# the other five folds together become the training set.

for (fold in seq_len(n_folds)) {
  
  cat("\n===== Six-fold spatial CV  --  fold", fold, "of", n_folds, "=====\n")
  
  # Split for this fold 
  # Training chips
  
  train_idx <- fold_table$idx[fold_table$fold != fold]
  
  # Validation chips
  
  val_idx   <- fold_table$idx[fold_table$fold == fold]
  cat(sprintf("  Training chips: %d  |  Validation chips: %d\n",
              length(train_idx), length(val_idx)))
  
  # Band statistics for this fold 
  
  band_stats <- compute_band_stats(image_files[train_idx])
  print(band_stats)
  
  # Save this fold's band statistics, with the fold number in the file name
  # so folds do not overwrite each other.
  
  saveRDS(band_stats,
          file.path(output_dir,
                    sprintf("feb_0.3m_stats_fold%02d.rds", fold)))
  
  # Datasets & loaders for this fold 
  
  train_ds <- chip_dataset(image_files, mask_files,
                           indices = train_idx, augment = TRUE,
                           band_stats = band_stats)
  
  val_ds   <- chip_dataset(image_files, mask_files,
                           indices = val_idx, augment = FALSE,
                           band_stats = band_stats)
  
  train_dl <- dataloader(train_ds, batch_size = 8, shuffle = TRUE,  num_workers = 0)
  val_dl   <- dataloader(val_ds,   batch_size = 8, shuffle = FALSE, num_workers = 0)
  
  # Visualise four Image and Mask Pairs
  
  n_show <- 4
  rgb_bands <- c(5, 3, 2)
  
  set.seed(123)
  sample_idx <- sample(seq_len(length(val_ds)), n_show)
  
  # helper function
  
  tensor_to_rgb <- function(t, bands) {
    arr <- as.array(t)
    rgb <- arr[bands, , , drop = FALSE]
    rgb <- aperm(rgb, c(2, 3, 1))
    pmin(pmax(rgb, 0), 1)
  }
  
  tensor_to_mask <- function(t) {
    as.array(t)[1, , ]
  }
  
  # Plot grid 
  
  op <- par(mfrow = c(n_show, 2),
            mar   = c(1, 1, 2, 1),
            oma   = c(0, 0, 2, 0))
  
  for (i in sample_idx) {
    pair <- val_ds$.getitem(i)
    rgb  <- tensor_to_rgb(pair$x, rgb_bands)
    msk  <- tensor_to_mask(pair$y)
    
    plot.new()
    plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1, xaxs = "i", yaxs = "i")
    rasterImage(rgb, 0, 0, 1, 1)
    title(main = sprintf("Image idx %d  (bands %s)",
                         i, paste(rgb_bands, collapse = ",")),
          cex.main = 0.8)
    
    msk_rgb <- array(0, dim = c(dim(msk), 3))
    msk_rgb[, , 1] <- msk
    msk_rgb[, , 2] <- msk
    
    plot.new()
    plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1, xaxs = "i", yaxs = "i")
    rasterImage(msk_rgb, 0, 0, 1, 1)
    title(main = sprintf("Mask idx %d  (pos pixels: %d)",
                         i, sum(msk > 0.5)),
          cex.main = 0.8)
  }
  
  mtext(sprintf("%d random val_ds chips", n_show),
        outer = TRUE, cex = 1, font = 2)
  
  par(op)
  
  # Per-fold output paths 
  # The fold number is inserted into each file name so every fold keeps its
  # own model checkpoint and its own training log.
  
  ckpt_path <- file.path(output_dir,
                         sprintf("unet_feb_0.3m_300426_fold%02d.pt", fold))
  log_path  <- file.path(output_dir,
                         sprintf("unet_feb_0.3m_300426_fold%02d_log.csv", fold))
  
  # Train the model for each fold 
 
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
    set_hparams(in_channels = 8L, num_classes = 1L) |>
    set_opt_hparams(lr = 3e-4) |>
    fit(
      data       = train_dl,
      valid_data = val_dl,
      epochs     = 100,
      callbacks  = list(
        luz_callback_lr_scheduler(
          lr_scheduler = torch::lr_step,
          step_size    = 15,
          gamma        = 0.5
        ),
        luz_callback_early_stopping(
          monitor  = "valid_loss",
          patience = 25,
          mode     = "min"
        ),
        luz_callback_model_checkpoint(
          path           = ckpt_path,
          monitor        = "valid_loss",
          mode           = "min",
          save_best_only = TRUE
        ),
        luz_callback_csv_logger(log_path)
      )
    )
  
  # plot the training/validation curves for this fold.
  
  plot(fitted)
  
  # Evaluate this fold and store the result ------------------------------------
  # evaluate() runs the trained model over this fold's validation set and
  # returns its final metric scores.
  evaluation <- fitted |> evaluate(data = val_dl)
  
  # Save the fold number, the trained model, and the validation metrics
  # into this fold's slot of 'cv_results'.
  
  cv_results[[fold]] <- list(
    fold    = fold,
    fitted  = fitted,
    metrics = get_metrics(evaluation)
  )
  
  cat("  Fold", fold, "validation metrics:\n")
  print(cv_results[[fold]]$metrics)
}

