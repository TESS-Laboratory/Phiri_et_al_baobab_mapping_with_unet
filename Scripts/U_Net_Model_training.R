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

image_dir <- "~/Chafika/image patches/New_patches_128pixels/small/images"
mask_dir <- "~/Chafika/image patches/New_patches_128pixels/small/masks"

# Get full file paths for all TIFF images in the image directory
image_paths <- list.files(image_dir, full.names = TRUE, ignore.case = TRUE, pattern = "\\.(tif|tiff)$")

# Get full file paths for all TIFF masks in the mask directory
mask_paths  <- list.files(mask_dir,  full.names = TRUE, ignore.case = TRUE, pattern = "\\.(tif|tiff)$")

# Sort image paths alphabetically to ensure consistent ordering
image_paths <- sort(image_paths)
                    
# Sort mask paths alphabetically so they align with their corresponding images
mask_paths  <- sort(mask_paths)

# Create a dataframe pairing each image with its corresponding mask
# stringsAsFactors = FALSE keeps paths as character strings, not factors
paired_image_masks_dataframe <- data.frame(image = image_paths, mask = mask_paths, stringsAsFactors = FALSE)


# Create 5-fold cross-validation splits

# set set
set.seed(100)

# Creates 5 stratified folds preserving class distribution
folds <- vfold_cv(paired_image_masks_dataframe, v = 5)


# Creating the dataset class for images and masks ------------------------------


baobab_dataset <- dataset(
  name = "baobab_dataset",
  
  initialize = function(img_files = NULL,
                        mask_files = NULL,
                        img_dir = NULL,
                        augmentation_params = NULL,
                        random_sampling = FALSE) {
    
    if (!is.null(img_files) && !is.null(mask_files)) {
      img_base <- file_path_sans_ext(basename(img_files))
      mask_base <- file_path_sans_ext(basename(mask_files))
      matched_base <- intersect(img_base, mask_base)
      
      if (length(matched_base) == 0) stop("No matching image-mask pairs found.")
      
      matched_imgs <- vapply(matched_base, function(base) img_files[which(img_base == base)][1], character(1))
      matched_masks <- vapply(matched_base, function(base) mask_files[which(mask_base == base)][1], character(1))
      
      self$images <- tibble(img = matched_imgs, mask = matched_masks)
      
    } else if (!is.null(img_dir)) {
      self$images <- tibble(
        img = grep(
          list.files(img_dir, full.names = TRUE, pattern = "\\.(tif|tiff)$", recursive = TRUE, ignore.case = TRUE),
          pattern = 'mask', invert = TRUE, value = TRUE
        ),
        mask = grep(
          list.files(img_dir, full.names = TRUE, pattern = "\\.(tif|tiff)$", recursive = TRUE, ignore.case = TRUE),
          pattern = 'mask', value = TRUE
        )
      )
    } else {
      stop("Must provide either img_files and mask_files, or img_dir")
    }
    
    self$load_image_tensor <- function(path) {
      arr <- as.array(terra::rast(path))
      dimnames(arr) <- NULL
      
      img_tensor <- torch_tensor(aperm(arr, c(3, 1, 2)),
                                 dtype = torch_float())$div(255)
      img_tensor
    }
    
    self$load_mask_tensor <- function(path) {
      arr <- as.array(terra::rast(path))
      dimnames(arr) <- NULL
      
      if (length(dim(arr)) == 3 && dim(arr)[3] == 1)
        arr <- arr[,,1]
      
      arr[is.nan(arr)] <- 0
      arr[arr == 2] <- 0
      arr[arr == 255] <- 0
      arr[arr > 1 & arr < 255] <- 0
      arr <- arr + 1
      
      torch_tensor(arr, dtype = torch_long())
    }
  },
    
#    self$load_mask_tensor <- function(path) {
 #     arr <- as.array(terra::rast(path))
  #    dimnames(arr) <- NULL
   #   
    #  # If 3D array with single channel, reduce to 2D
     # if (length(dim(arr)) == 3 && dim(arr)[3] == 1)
      #  arr <- arr[,,1]
      
      # Clean up invalid values
#      arr[is.nan(arr)] <- 0
 #     arr[arr == 2] <- 0
  #    arr[arr == 255] <- 0
   #   arr[arr > 1 & arr < 255] <- 0
      
      # CHANGE: Convert to binary {0, 1} instead of {1, 2}
      # Don't add 1 - keep it as 0 (background) and 1 (baobab)
      # arr <- arr + 1  ← REMOVE THIS LINE
      
      # Return as float32 for BCE loss (not long)
    #  torch_tensor(arr, dtype = torch_float32())
    #}
  #},
  
  .getitem = function(i) {
    img <- self$load_image_tensor(self$images$img[i])
    mask <- self$load_mask_tensor(self$images$mask[i])
    list(x = img, y = mask)
  },
  
  .length = function() {
    nrow(self$images)
  }
)

# Function to create dataloaders -----------------------------------------------

for(i in seq_along(folds$splits)) {
  split <- folds$splits[[i]]
  
  train_df <- analysis(split)
  test_df  <- assessment(split)
  
  train_ds <- baobab_dataset(train_df$image, train_df$mask)
  test_ds  <- baobab_dataset(test_df$image, test_df$mask)
  
  train_dl <- dataloader(train_ds, batch_size = 4, shuffle = TRUE, drop_last = TRUE)
  test_dl  <- dataloader(test_ds, batch_size = 4, shuffle = FALSE, drop_last = FALSE)
  
}


# Testing dataloader to check batch shapes and mask values ---------------------

batch <- dataloader_make_iter(train_dl) |> dataloader_next()

print(batch$x$shape)
print(batch$y$shape)
print(table(as_array(batch$y)))
print(batch$y$dtype)
#print(batch$y[10])
#batch 

# ------------------------------------------------------------------------------
# visualise the batch sample
# ------------------------------------------------------------------------------

#tensor_to_rgb <- function(img_tensor) {
  
  # Convert torch tensor to array and rearrange dimensions to H x W x C
  
 # img_array <- as_array(img_tensor)
  #img_rgb <- aperm(img_array, c(2, 3, 1))
  
  # Normalize values 
  
  #img_rgb <- (img_rgb - min(img_rgb)) / (max(img_rgb) - min(img_rgb))
  
  # Convert to raster object
  
  #as.raster(img_rgb)
#}

tensor_to_rgb <- function(img_tensor) {
  
  # Convert torch tensor to array
  # This converts from shape [8, 512, 512] to [8, 512, 512] array
  img_array <- as_array(img_tensor)
  
  # Rearrange dimensions from C x H x W to H x W x C
  # This changes from [8, 512, 512] to [512, 512, 8]
  img_rgb <- aperm(img_array, c(2, 3, 1))
  
  # **FIX: Select only the first 3 channels for RGB display**
  # This creates a [512, 512, 3] array from [512, 512, 8]
  img_rgb <- img_rgb[, , c(5, 3, 2)]
  
  # Normalize values between 0 and 1
  # This scales all pixel values to the range [0, 1] for proper display
  img_rgb <- (img_rgb - min(img_rgb)) / (max(img_rgb) - min(img_rgb))
  
  # Convert to raster object
  # Now it works because we have exactly 3 channels
  as.raster(img_rgb)
}

# Function to convert mask tensor (H,W) to a colored raster for visualization

#mask_to_raster <- function(mask_tensor) {
 # mask_array <- as_array(mask_tensor)
  
  # Map mask classes to colors (example: 1 = grey, 2 = forestgreen, 3 = red)
  
  #colors <- c("grey", "forestgreen", "red")
  
  #mask_colors <- matrix(colors[mask_array], nrow = nrow(mask_array), ncol = ncol(mask_array))
  #as.raster(mask_colors)
#}

mask_to_raster <- function(mask_tensor) {
  # Convert torch tensor to array
  # Shape: [512, 512] with values 0.0 and 1.0
  mask_array <- as_array(mask_tensor)
  
  # Map mask classes to colors:
  # Index 1 (value 0) = grey (background)
  # Index 2 (value 1) = forestgreen (baobab trees)
  colors <- c("forestgreen", "grey")
  
  # Add 1 to convert from 0-based mask values to 1-based R indexing
  # This converts: 0 -> 1 (grey), 1 -> 2 (forestgreen)
  mask_colors <- matrix(colors[mask_array + 1], 
                        nrow = nrow(mask_array), 
                        ncol = ncol(mask_array))
  
  # Convert to raster object for visualization
  as.raster(mask_colors)
}

# Plot images and masks from the batch

plot_batch <- function(batch, n = 4) {
  images <- batch$x[1:n]
  masks <- batch$y[1:n]
  
  plots <- list()
  for (i in 1:n) {
    img_raster <- tensor_to_rgb(images[i])
    mask_raster <- mask_to_raster(masks[i])
    
    p_img <- rasterGrob(img_raster, interpolate=TRUE)
    p_mask <- rasterGrob(mask_raster, interpolate=FALSE)
    
    plots[[2*i - 1]] <- grid::grid.grabExpr(grid::grid.draw(p_img))
    plots[[2*i]] <- grid::grid.grabExpr(grid::grid.draw(p_mask))
  }
  
  gridExtra::grid.arrange(grobs = plots, ncol = 2)
}

# Call function with the batch from the dataloader

plot_batch(batch, 4)

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

# Fit the model-----------------------------------------------------------------

fitted <- setup(
  unet,
  #loss = function(x, y) combined_loss_binary(
  # x, y, 
  #lambda_dice = 1.0, 
  #lambda_bce = 1.0
  #),
  loss = function(pred, target) combined_focal_dice_binary(
    pred, target, 
    lambda_dice = 1.0, 
    lambda_focal = 1.5,
    gamma = 2.0,
    alpha = 0.55  # High value to focus on class 2 (foreground)
    
  ),
  optimizer = function(params) optim_adamw(params, lr = 0.0001, betas = c(0.9, 0.999), eps = 1e-08, weight_decay = 0.01, amsgrad = FALSE),
  metrics = list(luz_metric_iou_binary(),
                 luz_metric_precision_binary(),
                 luz_metric_recall_binary(),
                 luz_metric_f1_binary()
  )
) %>% 
  set_hparams(input_channels = 8, num_classes = 2) %>%
  #set_opt_hparams(lr = 5e-4, weight_decay = 1e-4) %>% 
  fit(train_dl, epochs = 100, valid = test_dl, verbose = TRUE,
      callbacks = list(
        luz_callback_early_stopping(
          patience = 20, 
          monitor = "valid_loss",  
          mode = "min",
          min_delta = 0
        ),
        luz_callback_model_checkpoint(
          path = "~/Chafika/Phiri_et_al_baobab_mapping_with_unet/models/128small_crowns_Unet2_best_model.pt",
          monitor = "valid_loss",  
          mode = "min",
          save_best_only = TRUE,
          min_delta = 0
        ),
        luz_callback_metrics(),
        luz_callback_csv_logger("~/Chafika/Phiri_et_al_baobab_mapping_with_unet/CSV files/128small_crowns_training2_log.csv"))
  )

plot(fitted)

# model evaluation

evaluation <- fitted %>% evaluate(data = test_dl)
metrics <- get_metrics(evaluation)
print(metrics)

luz_save(fitted, path = "~/Chafika/Phiri_et_al_baobab_mapping_with_unet/models/128small_crowns_unet2_all.rds")


# ========================
# Plotting Function
# ========================
n_show <- 3  

# Get predictions for the whole validation loader
preds <- predict(fitted, test_dl)
pred_classes <- torch_argmax(preds, dim = 2)                                         

# Extract a batch of images + masks from the loader
batch <- dataloader_make_iter(test_dl) %>% dataloader_next()
images <- batch[[1]]
masks  <- batch[[2]]

# Setup layout: n_show rows, 3 columns
par(mfrow = c(n_show, 3), 
    mar = c(0, 0, 0, 0), 
    oma = c(0, 0, 3, 0))

# Define consistent colors (3 classes)
cols <- c("lightgrey", "forestgreen", "red")

# Define breaks so all classes (1,2,3) are always mapped
breaks <- c(0.5, 1.5, 2.5, 3.5)

for (i in 1:n_show) {
  # --- Extract image, mask, pred ---
  img   <- as_array(images[i, , , ])       # [3, H, W]
  mask  <- as_array(masks[i, , ])
  pred  <- as_array(pred_classes[i, , ])
  
  # Reorder image to H x W x 3
  img_rgb <- aperm(img, c(2, 3, 1))
  
  # Put into terra rasters
  r_img  <- rast(img_rgb)
  r_mask <- rast(mask)
  r_pred <- rast(pred)
  
  #  Resample masks/preds to match image
  r_mask <- terra::resample(r_mask, r_img, method = "near")
  r_pred <- terra::resample(r_pred, r_img, method = "near")
  
  # Rescale to 0–255 
  if (max(values(r_img), na.rm=TRUE) <= 1) {
    values(r_img) <- values(r_img) * 255
  }
  
  # --- Plot with consistent colors across all classes ---
  plotRGB(r_img, r=5, g=3, b=2, stretch="lin", axes = FALSE)
  plot(r_mask, legend = FALSE, axes = FALSE, col = cols, breaks = breaks)
  plot(r_pred, legend = FALSE, axes = FALSE, col = cols, breaks = breaks)
}

# Add one top title row
mtext("Image", side = 3, line = 1, outer = TRUE, at = 0.17, font = 2)
mtext("Mask", side = 3, line = 1, outer = TRUE, at = 0.5, font = 2)
mtext("Prediction", side = 3, line = 1, outer = TRUE, at = 0.83, font = 2)



# Load the trained model

fitted_reload <- luz_load("~/Chafika/Phiri_et_al_baobab_mapping_with_unet/models/128small_crowns_unet2_all.rds")

#---------------===================================================

#---------------===================================================

# Number of samples to visualize
n_show <- 3  

# Get all test data for random sampling
test_data <- test_dl$dataset
total_samples <- length(test_data)

# Randomly select indices to visualize
set.seed(as.integer(Sys.time()))
random_indices <- sample(1:total_samples, n_show, replace = FALSE)

# create quality plot with proper aspect ratio

plot_list <- list()

for (i in 1:n_show) {
  idx <- random_indices[i]
  
  # Get the specific sample
  sample <- test_data[idx]
  img_tensor <- sample[[1]]$unsqueeze(1)
  mask_tensor <- sample[[2]]$unsqueeze(1)
  
  # Get prediction
  pred <- predict(fitted_reload, img_tensor)
  pred_class <- torch_argmax(pred, dim = 2)$squeeze()
  
  # Convert to arrays
  img <- as_array(img_tensor$squeeze())
  mask <- as_array(mask_tensor$squeeze())
  pred <- as_array(pred_class)
  
  # Reorder image to H x W x 3 and normalize
  img_rgb <- aperm(img, c(2, 3, 1))
  img_rgb <- (img_rgb - min(img_rgb)) / (max(img_rgb) - min(img_rgb))
  
  # Get dimensions for proper aspect ratio
  height <- nrow(img_rgb)
  width <- ncol(img_rgb)
  aspect_ratio <- height / width
  
  # Create data frames for plotting
  img_df <- data.frame(
    x = rep(1:width, each = height),
    y = rep(height:1, width),
    r = as.vector(img_rgb[,,5]),
    g = as.vector(img_rgb[,,3]),
    b = as.vector(img_rgb[,,2])
  )
  
  mask_df <- data.frame(
    x = rep(1:width, each = height),
    y = rep(height:1, width),
    value = as.vector(mask)
  )
  
  pred_df <- data.frame(
    x = rep(1:width, each = height),
    y = rep(height:1, width),
    value = as.vector(pred)
  )
  
  # Create individual plots with proper aspect ratio
  p1 <- ggplot(img_df, aes(x = x, y = y)) +
    geom_raster(aes(fill = rgb(r, g, b))) +
    scale_fill_identity() +
    labs(title = ifelse(i == 1, "Input Image", "")) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
      axis.title = element_blank(),
      axis.text = element_blank(), #element_text(size = 8),
      panel.grid = element_blank(),
      plot.margin = margin(2, 2, 2, 2),
      aspect.ratio = aspect_ratio  # Maintain original aspect ratio
    ) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))
  
  p2 <- ggplot(mask_df, aes(x = x, y = y, fill = factor(value))) +
    geom_raster() +
    scale_fill_viridis_d(option = "plasma", name = "Class") +
    labs(title = ifelse(i == 1, "Reference", "")) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
      axis.title = element_blank(),
      axis.text = element_blank(), #element_text(size = 8),
      panel.grid = element_blank(),
      legend.position = "none",
      plot.margin = margin(2, 2, 2, 2),
      aspect.ratio = aspect_ratio  # Maintain original aspect ratio
    ) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))
  
  p3 <- ggplot(pred_df, aes(x = x, y = y, fill = factor(value))) +
    geom_raster() +
    scale_fill_viridis_d(option = "plasma", name = "Class") +
    labs(title = ifelse(i == 1, "Predicted Mask", "")) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
      axis.title = element_blank(),
      axis.text = element_blank(), #element_text(size = 8),
      panel.grid = element_blank(),
      legend.position = "none",
      plot.margin = margin(2, 2, 2, 2),
      aspect.ratio = aspect_ratio  # Maintain original aspect ratio
    ) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))
  
  # Combine plots for this sample
  sample_plot <- p1 + p2 + p3 + 
    plot_layout(nrow = 1) +
    plot_annotation(
      title = paste("Sample", i),
      theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12))
    )
  
  plot_list[[i]] <- sample_plot
}

# Create legend separately
legend_plot <- ggplot(mask_df, aes(x = x, y = y, fill = factor(value))) +
  geom_tile() + #geom_point(size = 5) +  # Invisible points for legend
  scale_fill_viridis_d(option = "plasma", name = "Segmentation Classes",
                       labels = c("Background", "Baobab Crown")) +
  theme_minimal() +
  theme(axis.text = element_blank(),  
        legend.position = "bottom",
        legend.title = element_text(face = "bold", size = 10),
        legend.text = element_text(size = 9),
        legend.key.size = unit(0.5, "cm"))

legend <- get_legend(legend_plot)

# Combine all samples with proper spacing and aspect ratio
final_plot <- wrap_plots(plot_list, ncol = 1) +
  plot_layout(heights = rep(1, n_show)) +
  plot_annotation(
    title = "U-Net Semantic Segmentation Results",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold", 
                                margin = margin(b = 10))
    )
  )

# Add legend at the bottom with proper spacing
final_plot <- final_plot / legend + 
  plot_layout(heights = c(rep(3, n_show), 0.5))

# Add axis labels to the entire figure
final_plot <- final_plot +
  plot_annotation(
    #caption = "Width (pixels)                                                                						 Height (pixels)",
    theme = theme(
      plot.caption = element_text(hjust = 0.5, size = 10, 
                                  margin = margin(t = 10)),
      plot.margin = margin(10, 10, 10, 10)
    )
  )

# Display the plot
print(final_plot)

# Save 
ggsave("small_crowns_segmentation_results_128pixels.tiff", final_plot, 
       width = 6,  
       height = 2.5 * n_show + 1, 
       dpi = 600, compression = "lzw", bg = "white")

# Also save as PDF
ggsave("small_crowns_segmentation_results_128pixels.pdf", final_plot, 
       width = 6, height = 2.5 * n_show + 1,
       device = cairo_pdf, bg = "white")

# ==============================================================================
# Define the theme
# ==============================================================================

theme_fancy <- function() {
  theme_bw() +
    theme(
      text = element_text(family = "Helvetica"),
      axis.text = element_text(size = 10, color = "black"),
      axis.title = element_text(size = 10, color = "black"),
      axis.line.x = element_line(linewidth = 0.5, color = "black"),
      axis.line.y = element_line(linewidth = 0.5, color = "black"),
      axis.ticks = element_line(linewidth = 0.4, color = "black"),
      panel.border = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), units = , "cm"),
      plot.title = element_text(
        size = 10,
        vjust = 1,
        hjust = 0.5,
        color = "black"
      ),
      legend.text = element_text(size = 9, color = "black"),
      legend.title = element_text(size = 9, color = "black"),
      legend.position.inside = c(0.9, 0.9),
      legend.key.size = unit(0.9, "line"),
      legend.background = element_rect(
        color = "black",
        fill = "transparent",
        linewidth = 2,
        linetype = "blank"
      )
    )
}

# Load and combine metrics from different model sizes

#small  <- read_csv("~/Chafika/Phiri_et_al_baobab_mapping_with_unet/CSV files/128small_crowns_training2_log.csv")  %>% 
 # mutate(size = "small")
#medium <- read_csv("~/Chafika/Phiri_et_al_baobab_mapping_with_unet/CSV files/128medium_crowns_training1_log.csv") %>% 
 # mutate(size = "medium")
large  <- read_csv("~/Chafika/image patches/128large_crowns_training_log.csv")  %>% 
  mutate(size = "large")

all_metrics <- bind_rows(large)#, medium, small)


# Plot metrics over epochs for different model sizes

# re-order sizes

loss <- ggplot(
  all_metrics,
  aes(
    x = epoch,
    y = loss,
    color = size,
    linetype = set
  )
) +
  geom_line(linewidth = 1) +
  scale_linetype_manual(
    values = c(train = "dotted", valid = "solid")
  ) +
  labs(
    title = "Training and Validation loss over Epochs",
    x = "Epoch",
    y = "loss",
    color = "Crown size",
    linetype = "Dataset"
  ) +
  theme_fancy()

loss

ggsave(
  loss,
  filename = "~/Chafika/Phiri_et_al_baobab_mapping_with_unet/Plots/loss_Metrics_128.png",
  width = 29.7,
  height = 21.0,
  units = "cm",
  dpi = 600
)

# multiple metrics in one plot

long_metrics <- all_metrics %>%
  pivot_longer(
    cols = c(loss, recall, iou, precision, f1),
    names_to = "metric",
    values_to = "value"
  )

plot_metrics <- ggplot(
  long_metrics,
  aes(
    x = epoch,
    y = value,
    color = size,
    linetype = set
  )
) +
  geom_line(linewidth = 1) +
  facet_wrap(~ metric, scales = "free_y") +
  scale_linetype_manual(
    values = c(train = "dotted", valid = "solid")
  ) +
  labs(
    title = "Training and Validation Metrics Large Crown Sizes",
    color = "Crown size",
    linetype = "Dataset"
  ) +
  theme_fancy()

plot_metrics

ggsave(
  plot_metrics,
  filename = "~/Chafika/Phiri_et_al_baobab_mapping_with_unet/Plots/256Large_Training_metrics_plot.png",
  width = 29.7,
  height = 21.0,
  units = "cm",
  dpi = 600
)
