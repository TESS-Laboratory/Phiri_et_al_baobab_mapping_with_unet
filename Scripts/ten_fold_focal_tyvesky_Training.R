#-------------------------------------------------------------------------------
# The purpose of this script is to train a Unet model on worldview 3 satellite images 
# with eight bands and using a ten-fold spatial cross validation approach
#-------------------------------------------------------------------------------

# packages

library(torch)
library(terra)
library(torchvision)
library(luz)
library(sf)

# File paths 

image_dir  <- "~/Chafika/image patches/New_patches_128pixels/large/images"
mask_dir   <- "~/Chafika/image patches/New_patches_128pixels/large/masks"
output_dir <- "~/Chafika/Phiri_et_al_baobab_mapping_with_unet/128_pixels_results/Ten_Fold/large/spatial_cv"

dir.create(output_dir, showWarnings = FALSE) # will create output folder

n_folds <- 10L # number of folds

# Load file paths 

img_files  <- sort(list.files(image_dir, pattern = "\\.(tif|tiff)$", full.names = TRUE, ignore.case = TRUE))
mask_files <- sort(list.files(mask_dir,  pattern = "\\.(tif|tiff)$", full.names = TRUE, ignore.case = TRUE))
stopifnot("Image and mask counts do not match" = length(img_files) == length(mask_files))
n_chips <- length(img_files)
cat("Total chips:", n_chips, "\n")

# K-means spatial fold assignment

get_centroid <- function(path) {
  e <- terra::ext(terra::rast(path))
  c(x = (e$xmin + e$xmax) / 2, y = (e$ymin + e$ymax) / 2)
}
coords <- t(vapply(img_files, get_centroid, numeric(2)))
colnames(coords) <- c("x", "y")   # explicitly set column names after transposing

set.seed(100)
km <- kmeans(coords, centers = n_folds, nstart = 25)

centroids <- data.frame(
  idx  = seq_len(n_chips),
  x    = coords[, "x"],
  y    = coords[, "y"],
  fold = km$cluster
)
cat("Chips per fold:\n"); print(table(centroids$fold))

# Dataset 

chip_dataset <- dataset(
  name = "WorldViewChips",
  initialize = function(image_files, mask_files, augment = FALSE) {
    self$image_files <- image_files
    self$mask_files  <- mask_files
    self$augment     <- augment
  },
  .getitem = function(i) {
    img  <- as.array(terra::rast(self$image_files[[i]]))
    mask <- as.array(terra::rast(self$mask_files[[i]]))
    img_t  <- torch_tensor(img)$permute(c(3L, 1L, 2L))$float()
    mask[is.na(mask)] <- 0
    mask_t <- torch_tensor(mask)$permute(c(3L, 1L, 2L))$float()
    mask_t <- (mask_t > 0.5)$to(dtype = torch_float())
    img_t  <- torch_clamp(img_t / 255, min = 0, max = 1)
    if (self$augment) {
      if (runif(1) > 0.5) { img_t <- transform_hflip(img_t); mask_t <- transform_hflip(mask_t) }
      if (runif(1) > 0.5) { img_t <- transform_vflip(img_t); mask_t <- transform_vflip(mask_t) }
    }
    list(x = img_t, y = mask_t)
  },
  .length = function() length(self$image_files)
)

# Loss functions

# Dice Coefficient

dice_coef <- function(y_pred, y_true, smooth = 1.0) {
  y_true_f     <- torch_flatten(y_true)
  y_pred_f     <- torch_flatten(y_pred)
  intersection <- torch_sum(y_true_f * y_pred_f)
  (2 * intersection + smooth) / (torch_sum(y_true_f) + torch_sum(y_pred_f) + smooth)
}


# BCE + Dice Loss 
# torch convention: (y_pred, y_true) 

#bce_dice_loss <- function(y_pred, y_true) {
#  nnf_binary_cross_entropy(y_pred, y_true) + (1 - dice_coef(y_pred, y_true))
#}


# Focal Tversky Loss 
# Combines the Tversky loss with a focal modulating exponent gamma.
#
# Tversky index = TP / (TP + alpha*FP + beta*FN)
#   alpha = 0.45 — weight on false positives
#   beta  = 0.55 — weight on false negatives (prioritises recall / fewer missed crowns)
#
# Focal modulation raises the loss to the power of gamma:
#   gamma = 0.75 — down-weights easy pixels; focuses learning on hard
#                  examples such as crown edges and shadowed canopy.
#   gamma = 1.0  — reduces to standard Tversky loss.

focal_tversky_loss <- function(y_pred, y_true,
                               alpha  = 0.45,
                               beta   = 0.55,
                               gamma  = 0.75,
                               smooth = 1.0) {
  y_true_f    <- torch_flatten(y_true)
  y_pred_f    <- torch_flatten(y_pred)
  tp          <- torch_sum(y_true_f * y_pred_f)
  fp          <- torch_sum((1 - y_true_f) * y_pred_f)   # predicted crown, true background
  fn          <- torch_sum(y_true_f * (1 - y_pred_f))   # predicted background, true crown
  tversky_idx <- (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
  torch_pow(1 - tversky_idx, gamma)
}

# Metrics functions (dice, iou, precision, recall and accuracy)

metric_update_counts <- function(self, preds, target) {
  p <- (preds  > 0.5)$to(dtype = torch_float())
  t <- (target > 0.5)$to(dtype = torch_float())
  self$tp <- self$tp + torch_sum( p *  t)$item()
  self$fp <- self$fp + torch_sum( p * (1 - t))$item()
  self$fn <- self$fn + torch_sum((1 - p) *  t)$item()
  self$tn <- self$tn + torch_sum((1 - p) * (1 - t))$item()
}

luz_metric_dice <- luz_metric(
  abbrev     = "Dice",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0 },
  update     = function(preds, target) {
    p <- (preds  > 0.5)$to(dtype = torch_float())
    t <- (target > 0.5)$to(dtype = torch_float())
    self$tp <- self$tp + torch_sum(p * t)$item()
    self$fp <- self$fp + torch_sum(p * (1 - t))$item()
    self$fn <- self$fn + torch_sum((1 - p) * t)$item()
  },
  compute    = function() (2 * self$tp) / (2 * self$tp + self$fp + self$fn + 1e-8)
)
luz_metric_precision <- luz_metric(
  abbrev     = "Prec",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update     = function(preds, target) metric_update_counts(self, preds, target),
  compute    = function() self$tp / (self$tp + self$fp + 1e-8)
)
luz_metric_recall <- luz_metric(
  abbrev     = "Rec",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update     = function(preds, target) metric_update_counts(self, preds, target),
  compute    = function() self$tp / (self$tp + self$fn + 1e-8)
)
luz_metric_f1 <- luz_metric(
  abbrev     = "F1",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update     = function(preds, target) metric_update_counts(self, preds, target),
  compute    = function() {
    prec <- self$tp / (self$tp + self$fp + 1e-8)
    rec  <- self$tp / (self$tp + self$fn + 1e-8)
    (2 * prec * rec) / (prec + rec + 1e-8)
  }
)
luz_metric_accuracy <- luz_metric(
  abbrev     = "Acc",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0; self$tn <- 0 },
  update     = function(preds, target) metric_update_counts(self, preds, target),
  compute    = function() (self$tp + self$tn) / (self$tp + self$tn + self$fp + self$fn + 1e-8)
)
luz_metric_iou <- luz_metric(
  abbrev     = "IoU",
  initialize = function() { self$tp <- 0; self$fp <- 0; self$fn <- 0 },
  update     = function(preds, target) {
    p <- (preds  > 0.5)$to(dtype = torch_float())
    t <- (target > 0.5)$to(dtype = torch_float())
    self$tp <- self$tp + torch_sum(p * t)$item()
    self$fp <- self$fp + torch_sum(p * (1 - t))$item()
    self$fn <- self$fn + torch_sum((1 - p) * t)$item()
  },
  compute    = function() self$tp / (self$tp + self$fp + self$fn + 1e-8)
)


# Helper blocks for the U-Net model building

# padding = 1 with kernel_size = 3 preserves spatial dimensions 
# Tensor format is NCHW throughout (Batch, Channels, Height, Width)

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


#  U-Net model building

unet_128 <- nn_module(
  
  initialize = function(in_channels = 8, num_classes = 1) {
    
    # Encoder
    self$down1 <- double_conv_block(in_channels, 64)
    self$pool1 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down2 <- double_conv_block(64,  128)
    self$pool2 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down3 <- double_conv_block(128, 256)
    self$pool3 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    self$down4 <- double_conv_block(256, 512)
    self$pool4 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    
    # Bottleneck
    self$center <- double_conv_block(512, 1024)
    
    # Decoder
    self$up4 <- triple_conv_block(1024 + 512, 512)
    self$up3 <- triple_conv_block( 512 + 256, 256)
    self$up2 <- triple_conv_block( 256 + 128, 128)
    self$up1 <- triple_conv_block( 128 +  64,  64)
    
    # Classifier
    
    self$classify <- nn_conv2d(64, num_classes, kernel_size = 1)
  },
  
  forward = function(x) {
    
    # Encoder
    
    d1  <- self$down1(x)
    d2  <- self$down2(self$pool1(d1))
    d3  <- self$down3(self$pool2(d2))
    d4  <- self$down4(self$pool3(d3))
    
    # Bottleneck
    
    ctr <- self$center(self$pool4(d4))
    
    # Decoder — upsample then concatenate skip connections on the channel dim (dim = 2)
    
    up <- function(tensor, skip) nnf_interpolate(tensor, scale_factor = 2, mode = "nearest") |>
      (\(t) torch_cat(list(skip, t), dim = 2))()
    
    u4 <- self$up4(up(ctr, d4))
    u3 <- self$up3(up(u4,  d3))
    u2 <- self$up2(up(u3,  d2))
    u1 <- self$up1(up(u2,  d1))
    
    # Output
    
    torch_sigmoid(self$classify(u1))
  }
)

# Sanity check with dumy figures to confirm if the model is working

#model       <- unet_128(in_channels = 8, num_classes = 1)
#dummy_input <- torch_randn(2, 8, 128, 128)
#output      <- model(dummy_input)
# Expected: 2 x 1 x 128 x 128
#cat("Output shape:", paste(dim(output), collapse = " x "), "\n")


# Ten-fold spatial CV training and validation loop

cv_results <- vector("list", n_folds)

for (fold in seq_len(n_folds)) {
  
  cat(sprintf("\n══ Fold %d / %d ══\n", fold, n_folds))
  
  train_idx <- centroids$idx[centroids$fold != fold]
  val_idx   <- centroids$idx[centroids$fold == fold]
  cat(sprintf("  Train: %d chips  |  Val: %d chips\n", length(train_idx), length(val_idx)))
  
  train_ds <- chip_dataset(img_files[train_idx], mask_files[train_idx], augment = TRUE)
  val_ds   <- chip_dataset(img_files[val_idx],   mask_files[val_idx],   augment = FALSE)
  
  train_dl <- dataloader(train_ds, batch_size = 8, shuffle = TRUE,  num_workers = 0)
  val_dl   <- dataloader(val_ds,   batch_size = 8, shuffle = FALSE, num_workers = 0)
  
# fit the model 
  
  fitted <- unet_128 |>
    setup(
      loss      = focal_tversky_loss,
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
    set_hparams(in_channels = 8, num_classes = 1) |>
    set_opt_hparams(lr = 1e-4) |>
    fit(
      data       = train_dl,
      valid_data = val_dl,
      epochs     = 100,
      callbacks  = list(
        luz_callback_early_stopping(monitor = "valid_loss", patience = 20),
        luz_callback_model_checkpoint(
          path           = file.path(output_dir, sprintf("fold_%02d/", fold)),
          monitor        = "valid_loss",
          save_best_only = TRUE
        )
      )
    )
  
  # Save per-fold metrics CSV
  
  m            <- get_metrics(fitted)
  last_epoch   <- max(m$epoch)
  fold_metrics <- m[m$epoch == last_epoch & m$set == "valid", c("metric", "value")]
  fold_metrics$fold <- fold
  write.csv(fold_metrics,
            file.path(output_dir, sprintf("fold_%02d_metrics.csv", fold)),
            row.names = FALSE)
  cat(sprintf("  Fold %d — checkpoint and metrics CSV saved.\n", fold))
  
  cv_results[[fold]] <- fitted
}

# Summary across all folds

extract_val_metrics <- function(fitted) {
  m <- get_metrics(fitted)
  last_epoch <- max(m$epoch)
  m[m$epoch == last_epoch & m$set == "valid", c("metric", "value")]
}

summary_df <- do.call(rbind, lapply(seq_len(n_folds), function(fold) {
  m <- extract_val_metrics(cv_results[[fold]])
  m$fold <- fold
  m
}))

summary_wide <- reshape(summary_df, idvar = "fold", timevar = "metric", direction = "wide")
names(summary_wide) <- gsub("value\\.", "", names(summary_wide))

cat("\n...Per-fold validation metrics....\n")
print(summary_wide, row.names = FALSE)

metric_cols <- setdiff(names(summary_wide), "fold")
stats_df <- data.frame(
  metric = metric_cols,
  mean   = sapply(metric_cols, function(m) mean(summary_wide[[m]], na.rm = TRUE)),
  sd     = sapply(metric_cols, function(m) sd(summary_wide[[m]],   na.rm = TRUE))
)
cat("\n Mean ± SD across folds....\n")
print(stats_df, row.names = FALSE)

write.csv(summary_wide, file.path(output_dir, "spatial_cv_results.csv"), row.names = FALSE)
cat("\nSummary saved to:", file.path(output_dir, "spatial_cv_results.csv"), "\n")