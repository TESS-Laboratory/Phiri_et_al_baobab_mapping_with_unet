# ==============================================================================
# Introduction
# This Script Trains and Validates simple UNet model on WorldView-3 image chips
# to map baobab trees
# ==============================================================================

# Load the required packages --------------------------------------------------- 

library(torch)        
library(terra)        
library(luz)          
library(torchvision)  
library(rsample)      
library(tidyverse)    
library(patchwork)    
library(grid)         
library(gridExtra)    
library(tools)        
library(ggplot2)
library(viridis)
library(ggpubr)

# File paths containing images and mask ----------------------------------------

image_dir <- "~/Chafika/image patches/New_patches_128pixels/large/images"
mask_dir <- "~/Chafika/image patches/New_patches_128pixels/large/masks"

# Get full file paths for all TIFF images in the image directory
image_paths <- list.files(image_dir, full.names = TRUE, ignore.case = TRUE, pattern = "\\.(tif|tiff)$")

# Get full file paths for all TIFF masks in the mask directory
mask_paths  <- list.files(mask_dir,  full.names = TRUE, ignore.case = TRUE, pattern = "\\.(tif|tiff)$")

# Sort image paths alphabetically to ensure consistent ordering
image_paths <- sort(image_paths)

# Sort mask paths alphabetically so they align with their corresponding images
mask_paths  <- sort(mask_paths)

# Create a dataframe pairing each image with its corresponding mask
paired_image_masks_dataframe <- data.frame(image = image_paths, mask = mask_paths, stringsAsFactors = FALSE)


# Create 5-fold cross-validation splits

# set set
set.seed(100)


# Spatial Cross validation Set-up ----------------------------------------------


get_centroid <- function(path) {
  r <- terra::rast(path)          # Load raster to read its spatial extent
  e <- terra::ext(r)              # Extract bounding box (xmin, xmax, ymin, ymax)
  c(x = (e$xmin + e$xmax) / 2,   # Midpoint on x-axis
    y = (e$ymin + e$ymax) / 2)   # Midpoint on y-axis
}

# Apply get_centroid to every image; result is a 2-column matrix (x, y)
coords <- t(vapply(image_paths, get_centroid, numeric(2)))

# K-means clusters patches into 6 spatially contiguous groups

set.seed(100)
km <- kmeans(coords, centers = 6, nstart = 25)

# Attach the cluster ID (1-6) as the fold column
paired_image_masks_dataframe$fold <- km$cluster



# Creating the dataset class for images and masks ------------------------------


baobab_dataset <- dataset(
  name = "baobab_dataset",
  
  initialize = function(img_files = NULL,
                        mask_files = NULL,
                        img_dir = NULL,
                        augmentation_params = NULL,
                        random_sampling = FALSE) {
    
    if (!is.null(img_files) && !is.null(mask_files)) {
      img_base  <- file_path_sans_ext(basename(img_files))
      mask_base <- file_path_sans_ext(basename(mask_files))
      matched_base <- intersect(img_base, mask_base)
      
      if (length(matched_base) == 0) stop("No matching image-mask pairs found.")
      
      matched_imgs  <- vapply(matched_base, function(b) img_files [which(img_base  == b)][1], character(1))
      matched_masks <- vapply(matched_base, function(b) mask_files[which(mask_base == b)][1], character(1))
      
      self$images <- tibble(img = matched_imgs, mask = matched_masks)
      
    } else if (!is.null(img_dir)) {
      all_files <- list.files(img_dir, full.names = TRUE,
                              pattern = "\\.(tif|tiff)$",
                              recursive = TRUE, ignore.case = TRUE)
      self$images <- tibble(
        img  = grep(all_files, pattern = "mask", invert = TRUE, value = TRUE),
        mask = grep(all_files, pattern = "mask",                value = TRUE)
      )
    } else {
      stop("Must provide either img_files and mask_files, or img_dir")
    }
    
    self$load_image_tensor <- function(path) {
      arr <- as.array(terra::rast(path))   
      dimnames(arr) <- NULL
      torch_tensor(aperm(arr, c(3, 1, 2)),  
                   dtype = torch_float())$div(255)  
    }
    
    self$load_mask_tensor <- function(path) {
      arr <- as.array(terra::rast(path))
      dimnames(arr) <- NULL
      if (length(dim(arr)) == 3 && dim(arr)[3] == 1) arr <- arr[,,1]  # Drop singleton channel
      arr[is.nan(arr)] <- 0
      arr[arr == 2]    <- 0
      arr[arr == 255]  <- 0
      arr[arr > 1 & arr < 255] <- 0
      arr <- arr + 1   
      torch_tensor(arr, dtype = torch_long())
    }
  },
  
  .getitem = function(i) {
    list(x = self$load_image_tensor(self$images$img[i]),
         y = self$load_mask_tensor(self$images$mask[i]))
  },
  
  .length = function() nrow(self$images)
)

# Define U-Net Model -----------------------------------------------------------

unet <- nn_module(
  "UNet",
  
  initialize = function(input_channels = 8, num_classes = 2) {
    
    # Downsampling/encoder
    self$enc1_1 <- nn_conv2d(input_channels, 64, kernel_size = 3, padding = 1)
    self$enc1_2 <- nn_conv2d(64, 64, kernel_size = 3, padding = 1)
    self$pool1 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    
    self$enc2_1 <- nn_conv2d(64, 128, kernel_size = 3, padding = 1)
    self$enc2_2 <- nn_conv2d(128, 128, kernel_size = 3, padding = 1)
    self$pool2 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    
    self$enc3_1 <- nn_conv2d(128, 256, kernel_size = 3, padding = 1)
    self$enc3_2 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
    self$pool3 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    
    self$enc4_1 <- nn_conv2d(256, 512, kernel_size = 3, padding = 1)
    self$enc4_2 <- nn_conv2d(512, 512, kernel_size = 3, padding = 1)
    self$pool4 <- nn_max_pool2d(kernel_size = 2, stride = 2)
    
    # Bottleneck
    self$bottleneck1 <- nn_conv2d(512, 1024, kernel_size = 3, padding = 1)
    self$bottleneck2 <- nn_conv2d(1024, 1024, kernel_size = 3, padding = 1)
    
    # Upsampling/decoder
    #self$up1 <- nn_upsample(scale_factor = 2, mode = "bilinear")
    #self$up1    <- nn_conv_transpose2d(1024, 1024, kernel_size = 2, stride = 2)
    self$up1 <- nn_upsample(scale_factor = 2, mode = "nearest")
    self$dec1_1 <- nn_conv2d(1536, 512, kernel_size = 3, padding = 1)  # 1024 + 512
    self$dec1_2 <- nn_conv2d(512, 512, kernel_size = 3, padding = 1)
    
    #self$up2 <- nn_upsample(scale_factor = 2, mode = "bilinear")
    #self$up2    <- nn_conv_transpose2d(512, 512, kernel_size = 2, stride = 2)
    self$up2 <- nn_upsample(scale_factor = 2, mode = "nearest")
    self$dec2_1 <- nn_conv2d(768, 256, kernel_size = 3, padding = 1)   # 512 + 256
    self$dec2_2 <- nn_conv2d(256, 256, kernel_size = 3, padding = 1)
    
    #self$up3 <- nn_upsample(scale_factor = 2, mode = "bilinear")
    #self$up3    <- nn_conv_transpose2d(256, 256, kernel_size = 2, stride = 2)
    self$up3 <- nn_upsample(scale_factor = 2, mode = "nearest")
    self$dec3_1 <- nn_conv2d(384, 128, kernel_size = 3, padding = 1)   # 256 + 128
    self$dec3_2 <- nn_conv2d(128, 128, kernel_size = 3, padding = 1)
    
    #self$up4 <- nn_upsample(scale_factor = 2, mode = "bilinear")
    #self$up4    <- nn_conv_transpose2d(128, 128, kernel_size = 2, stride = 2)
    self$up4 <- nn_upsample(scale_factor = 2, mode = "nearest")
    self$dec4_1 <- nn_conv2d(192, 64, kernel_size = 3, padding = 1)    # 128 + 64
    self$dec4_2 <- nn_conv2d(64, 64, kernel_size = 3, padding = 1)
    
    # Output
    self$output <- nn_conv2d(64, num_classes, kernel_size = 1)
    
    self$relu <- nn_relu()
  },
  
  forward = function(x) {
    enc1 <- self$relu(self$enc1_1(x))
    enc1 <- self$relu(self$enc1_2(enc1))
    
    enc2 <- self$pool1(enc1)
    enc2 <- self$relu(self$enc2_1(enc2))
    enc2 <- self$relu(self$enc2_2(enc2))
    
    enc3 <- self$pool2(enc2)
    enc3 <- self$relu(self$enc3_1(enc3))
    enc3 <- self$relu(self$enc3_2(enc3))
    
    enc4 <- self$pool3(enc3)
    enc4 <- self$relu(self$enc4_1(enc4))
    enc4 <- self$relu(self$enc4_2(enc4))
    
    bottleneck <- self$pool4(enc4)
    bottleneck <- self$relu(self$bottleneck1(bottleneck))
    bottleneck <- self$relu(self$bottleneck2(bottleneck))
    
    dec1 <- self$up1(bottleneck)
    dec1 <- torch_cat(list(enc4, dec1), dim = 2)
    dec1 <- self$relu(self$dec1_1(dec1))
    dec1 <- self$relu(self$dec1_2(dec1))
    
    dec2 <- self$up2(dec1)
    dec2 <- torch_cat(list(enc3, dec2), dim = 2)
    dec2 <- self$relu(self$dec2_1(dec2))
    dec2 <- self$relu(self$dec2_2(dec2))
    
    dec3 <- self$up3(dec2)
    dec3 <- torch_cat(list(enc2, dec3), dim = 2)
    dec3 <- self$relu(self$dec3_1(dec3))
    dec3 <- self$relu(self$dec3_2(dec3))
    
    dec4 <- self$up4(dec3)
    dec4 <- torch_cat(list(enc1, dec4), dim = 2)
    dec4 <- self$relu(self$dec4_1(dec4))
    dec4 <- self$relu(self$dec4_2(dec4))
    
    output <- self$output(dec4)
    
    return(output)
  }
)


# Example use

#model <- unet(input_channels = 8, num_classes = 1)

# Test with dummy input (batch size 1, 8×512×512 image)
#x <- torch_randn(1, 8, 512, 512)
#y <- model(x)

#print(y$shape)

# IoU Metric (Intersection over Union) for Binary Segmentation
luz_metric_iou_binary <- luz_metric(
  abbrev = "IoU",
  initialize = function() {
    self$intersection <- 0
    self$union <- 0
  },
  update = function(preds, target) {
    preds_class <- torch_argmax(preds, dim = 2)
    self$intersection <- self$intersection + ((preds_class == 2) & (target == 2))$sum()$item()
    self$union       <- self$union        + ((preds_class == 2) | (target == 2))$sum()$item()
  },
  compute = function() {
    ifelse(self$union > 0, self$intersection / self$union, 0)
  }
)

# Precision: Of all pixels predicted as class 2, what fraction was correct?
luz_metric_precision_binary <- luz_metric(
  abbrev = "Precision",
  initialize = function() {
    self$true_positives  <- 0
    self$false_positives <- 0
  },
  update = function(preds, target) {
    preds_class <- torch_argmax(preds, dim = 2)
    self$true_positives  <- self$true_positives  + ((preds_class == 2) & (target == 2))$sum()$item()
    self$false_positives <- self$false_positives + ((preds_class == 2) & (target != 2))$sum()$item()
  },
  compute = function() {
    denom <- self$true_positives + self$false_positives
    ifelse(denom > 0, self$true_positives / denom, 0)
  }
)

# Recall: Of all actual class 2 pixels, what fraction was found?
luz_metric_recall_binary <- luz_metric(
  abbrev = "Recall",
  initialize = function() {
    self$true_positives  <- 0
    self$false_negatives <- 0
  },
  update = function(preds, target) {
    preds_class <- torch_argmax(preds, dim = 2)
    self$true_positives  <- self$true_positives  + ((preds_class == 2) & (target == 2))$sum()$item()
    self$false_negatives <- self$false_negatives + ((preds_class != 2) & (target == 2))$sum()$item()
  },
  compute = function() {
    denom <- self$true_positives + self$false_negatives
    ifelse(denom > 0, self$true_positives / denom, 0)
  }
)

# F1-Score: Harmonic mean of Precision and Recall
luz_metric_f1_binary <- luz_metric(
  abbrev = "F1",
  initialize = function() {
    self$true_positives  <- 0
    self$false_positives <- 0
    self$false_negatives <- 0
  },
  update = function(preds, target) {
    preds_class <- torch_argmax(preds, dim = 2)
    self$true_positives  <- self$true_positives  + ((preds_class == 2) & (target == 2))$sum()$item()
    self$false_positives <- self$false_positives + ((preds_class == 2) & (target != 2))$sum()$item()
    self$false_negatives <- self$false_negatives + ((preds_class != 2) & (target == 2))$sum()$item()
  },
  compute = function() {
    precision <- ifelse(
      self$true_positives + self$false_positives > 0,
      self$true_positives / (self$true_positives + self$false_positives), 0)
    recall <- ifelse(
      self$true_positives + self$false_negatives > 0,
      self$true_positives / (self$true_positives + self$false_negatives), 0)
    ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  }
)

# Dice Loss – focuses on foreground (class 2) spatial overlap
dice_loss_binary <- function(pred, target) {
  pred_sm     <- nnf_softmax(pred, dim = 2)
  pred_class2 <- pred_sm[, 2, , ]
  target_class2 <- (target == 2)$to(dtype = pred_class2$dtype)
  
  intersection <- (pred_class2 * target_class2)$sum()
  # 1e-6 added for numerical stability (avoids division by zero)
  dice <- (2.0 * intersection + 1e-6) / (pred_class2$sum() + target_class2$sum() + 1e-6)
  1.0 - dice
}

# Weighted BCE Loss – high weight on class 2 to counter class imbalance
#class_weights_binary <- torch_tensor(c(0.1, 2.0), dtype = torch_float())
#if (cuda_is_available()) class_weights_binary <- class_weights_binary$cuda()
#bce_loss_weighted <- nn_cross_entropy_loss(weight = class_weights_binary)

# Combined BCE + Dice Loss
#combined_loss_binary <- function(pred, target, lambda_dice = 1.0, lambda_bce = 1.0) {
#  lambda_bce * bce_loss_weighted(pred, target) + lambda_dice * dice_loss_binary(pred, target)
#}

# Focal Loss – down-weights easy examples, focuses training on hard pixels
focal_loss_binary <- function(logits, target, gamma = 2.0, alpha = 0.55) {
  probs      <- nnf_softmax(logits, dim = 2)
  prob_class2 <- probs[, 2, , ]
  is_class2  <- (target == 2)$to(dtype = probs$dtype)
  
  pt           <- is_class2 * prob_class2 + (1 - is_class2) * (1 - prob_class2)
  focal_weight <- (1 - pt)$pow(gamma)
  alpha_weight <- is_class2 * alpha + (1 - is_class2) * (1 - alpha)
  
  # 1e-6 prevents log(0)
  loss <- focal_weight * alpha_weight * -torch_log(pt + 1e-6)
  loss$mean()
}

# Combined Focal + Dice Loss
combined_focal_dice_binary <- function(pred, target,
                                       lambda_dice  = 1.0,
                                       lambda_focal = 1.5,
                                       gamma = 2.0,
                                       alpha = 0.55) {
  lambda_focal * focal_loss_binary(pred, target, gamma, alpha) +
    lambda_dice  * dice_loss_binary(pred, target)
}

# ==============================================================================
# SPATIAL 6-FOLD TRAINING LOOP
# ==============================================================================

fold_results <- vector("list", 6)   # Stores fitted model and metrics for each fold

for (fold_id in 1:6) {
  
  cat("\n===== Spatial Fold", fold_id, "/ 6 =====\n")
  
  # Rows NOT in the current fold become the training set
  train_df <- paired_image_masks_dataframe[paired_image_masks_dataframe$fold != fold_id, ]
  
  # Rows IN the current fold become the validation set
  val_df   <- paired_image_masks_dataframe[paired_image_masks_dataframe$fold == fold_id, ]
  
  # Build dataset objects from the split file paths
  train_ds <- baobab_dataset(train_df$image, train_df$mask)
  val_ds   <- baobab_dataset(val_df$image,   val_df$mask)
  
  # Wrap datasets in dataloaders for batched iteration
  train_dl <- dataloader(train_ds, batch_size = 4, shuffle = TRUE,  drop_last = TRUE)
  val_dl   <- dataloader(val_ds,   batch_size = 4, shuffle = FALSE, drop_last = FALSE)
  
  # Build checkpoint and log paths that are fold-specific
  ckpt_path <- sprintf(
    "~/Chafika/Phiri_et_al_baobab_mapping_with_unet/models/128large_crowns_Unet2_fold%d_best.pt",
    fold_id)
  log_path  <- sprintf(
    "~/Chafika/Phiri_et_al_baobab_mapping_with_unet/CSV files/128large_crowns_training2_fold%d_log.csv",
    fold_id)
  
  fitted <- setup(
    unet,
    loss = function(pred, target) combined_focal_dice_binary(
      pred, target,
      lambda_dice  = 1.0,
      lambda_focal = 1.5,
      gamma        = 2.0,
      alpha        = 0.55
    ),
    optimizer = function(params) optim_adamw(
      params,
      lr           = 0.0001,
      betas        = c(0.9, 0.999),
      eps          = 1e-08,
      weight_decay = 0.01,
      amsgrad      = FALSE
    ),
    metrics = list(
      luz_metric_iou_binary(),
      luz_metric_precision_binary(),
      luz_metric_recall_binary(),
      luz_metric_f1_binary()
    )
  ) %>%
    set_hparams(input_channels = 8, num_classes = 2) %>%
    fit(
      train_dl,
      epochs  = 20,
      valid   = val_dl,
      verbose = TRUE,
      callbacks = list(
        luz_callback_early_stopping(
          patience  = 10,
          monitor   = "valid_loss",
          mode      = "min",
          min_delta = 0
        ),
        luz_callback_model_checkpoint(
          path           = ckpt_path,
          monitor        = "valid_loss",
          mode           = "min",
          save_best_only = TRUE,
          min_delta      = 0
        ),
        luz_callback_metrics(),
        luz_callback_csv_logger(log_path)
      )
    )
  
  # Evaluate on the held-out spatial fold and store results
  evaluation         <- fitted %>% evaluate(data = val_dl)
  fold_results[[fold_id]] <- list(
    fold    = fold_id,
    fitted  = fitted,
    metrics = get_metrics(evaluation)
  )
  
  print(fold_results[[fold_id]]$metrics)
}


# Check metrics accross all folds ----------------------------------------------

# Combine per-fold metric tables into one dataframe and compute mean per metric
all_metrics <- lapply(seq_along(fold_results), function(i) {
  df       <- fold_results[[i]]$metrics   # data.frame with columns: metric, value, set
  df$fold  <- i                           # Tag which fold the row belongs to
  df
})

summary_metrics <- bind_rows(all_metrics) %>%
  group_by(metric) %>%
  summarise(mean_value = mean(value, na.rm = TRUE),  # Average metric over 6 folds
            sd_value   = sd(value,   na.rm = TRUE),  # Spread across folds
            .groups    = "drop")

print(summary_metrics)
