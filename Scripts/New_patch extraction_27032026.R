# ------------------------------------------------------------------------------
# Introduction
# This script is intended to be used for extracting centroid based image and mask
# chips for the U-Net model training and validation. 
# Each chip is centred on the centroid of one tree polygon.
# Every chip is guaranteed to contain at least one complete baobab crown.
# It also stratifies the chips into three equal buckets based on the crown
# sizes: Small, Medium and Large
# ------------------------------------------------------------------------------

# Load necessary library for raster and vector operations

library(tidyverse)
library(terra)
library(sf)
library(knitr)
library(tools)
library(geodl)
library(fs)
library(purrr)

# Load data february -----------------------------------------------------------

# load and read image

Kremetart_image <- rast("~/Chafika/My_Dataset/mapungubwe_july_image_8bit.tif")

# load and read polygons

Kremetart_tree_polygons <- vect("~/Chafika/My_Dataset/Dry_Mapungubwe_tree_polygons.gpkg")

# Reproject polygons -----------------------------------------------------------

# Check if coordinate reference systems (CRS) match between polygons and image

# If not, reproject polygons to match the image CRS

if (!crs(Kremetart_tree_polygons) == crs(Kremetart_image)) {
  Kremetart_tree_polygons <- project(
    Kremetart_tree_polygons,
    crs(Kremetart_image)  # Target CRS from the image
  )
}

# Crop polygons to image extent ------------------------------------------------

Kremetart_tree_polygons <- crop(
  Kremetart_tree_polygons,
  ext(Kremetart_image)
)

# Output directories -----------------------------------------------------------

# Define where to save the extracted patches

img_dir  <- "~/Chafika/image patches/Mapungubwe_july_polygons/128_8bands/images"
mask_dir <- "~/Chafika/image patches/Mapungubwe_july_polygons/128_8bands/masks"

# Create the directories if they don't exist

dir.create(img_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(mask_dir, recursive = TRUE, showWarnings = FALSE)

# Function to extract patches based on polygon centroids -----------------------

extract_polygon_centroid_patches <- function(image, polygons,
                                             patch_size = 128,
                                             img_dir, mask_dir,
                                             season_prefix = "july",
                                             target_classes = 1) {
  
# Get image dimensions in pixels ---------------------------------------------
  
  ncols <- ncol(image)  # Width (number of columns)
  nrows <- nrow(image)  # Height (number of rows)
  
  # Get spatial properties needed for coordinate conversion
  
  full_extent <- ext(image)      # Geographic bounding box (xmin, xmax, ymin, ymax)
  res_x <- res(image)[1]         # Pixel width in map units (e.g., meters)
  res_y <- res(image)[2]         # Pixel height in map units (e.g., meters)
  xmin_full <- full_extent[1]    # Leftmost X coordinate
  ymax_full <- full_extent[4]    # Topmost Y coordinate
  
  # Filter polygons to only target classes
  
  polygons_filtered <- polygons[polygons$Class %in% target_classes, ]
  
  # Get total count of polygons to process
  
  n_polygons <- nrow(polygons_filtered)
  cat(sprintf("\nProcessing %d tree crown polygons...\n", n_polygons))
  
  # Create raster template for rasterising individual crowns
  
  mask_template <- rast(image)[[1]]
  
  # Calculate centroids of all polygons at once
  
  polygon_centroids <- centroids(polygons_filtered)
  
  # Extract centroid coordinates as a matrix
  
  centroid_coords <- crds(polygon_centroids)
  
  # Initialize counters
  
  patch_id <- 1       # Sequential patch number
  skipped_count <- 0  # Count of crowns too close to edge
  
  # Calculate half-patch size once (used for all patches)
  
  half_patch <- floor(patch_size / 2)
  
  # Loop through each polygon
  
  for (i in seq_len(n_polygons)) {
    # Get current polygon and its properties
    current_polygon <- polygons_filtered[i, ]        # Extract single polygon by index
    class_value <- current_polygon$Class             # Get class value (1 or 2)
    # Get centroid coordinates for this polygon
    centroid_x <- centroid_coords[i, 1]  # X coordinate (easting)
    centroid_y <- centroid_coords[i, 2]  # Y coordinate (northing)
    # Convert geographic coordinates to pixel indices
    centroid_col <- round((centroid_x - xmin_full) / res_x) + 1  
    centroid_row <- round((ymax_full - centroid_y) / res_y) + 1  
    # Calculate patch boundaries in pixel coordinates
    row_start <- centroid_row - half_patch      
    row_end   <- centroid_row + half_patch - 1  
    col_start <- centroid_col - half_patch      
    col_end   <- centroid_col + half_patch - 1  
    # Check if patch extends beyond image boundaries
    if (row_start < 1 || row_end > nrows || col_start < 1 || col_end > ncols) {
      skipped_count <- skipped_count + 1
      next 
    }
    # Convert pixel indices back to geographic coordinates for cropping
    
    xmin_patch <- xmin_full + (col_start - 1) * res_x  
    xmax_patch <- xmin_full + col_end * res_x          
    ymax_patch <- ymax_full - (row_start - 1) * res_y  
    ymin_patch <- ymax_full - row_end * res_y          
    # Create extent object for the patch
    patch_extent <- ext(xmin_patch, xmax_patch, ymin_patch, ymax_patch)
    # Extract image patch
    #img_patch <- crop(image, patch_extent)
    # Create mask patch by rasterizing only the current polygon
    #mask_patch <- mask_template
    # Rasterize the current polygon onto the template
    #mask_patch <- rasterize(
    # current_polygon,
    #mask_patch,
    #field = "Class",
    #background = NA,
    #touches = TRUE
    #)
    
    # Extract image patch
    img_patch <- crop(image, patch_extent)
    
    mask_patch <- rasterize(
      current_polygon,
      img_patch[[1]],
      field      = "Class",
      background = 0L,     # explicitly labels non-crown pixels (NA) as 0
      touches    = TRUE
    )
    # Crop mask to same extent as image patch
    mask_patch <- crop(mask_patch, patch_extent)
    # Create informative filenames
    img_file  <- file.path(img_dir,  
                           sprintf("%s_class%d_poly%d_%05d.tif", 
                                   season_prefix, class_value, i, patch_id))
    mask_file <- file.path(mask_dir, 
                           sprintf("%s_class%d_poly%d_%05d.tif", 
                                   season_prefix, class_value, i, patch_id))
    # Save patches to disk as GeoTIFF files
    writeRaster(img_patch,  img_file,  overwrite = TRUE, filetype = "GTiff")
    writeRaster(mask_patch, mask_file, overwrite = TRUE, filetype = "GTiff")
    # Print progress every 50 patches
    if (patch_id %% 100 == 0) {
      cat(sprintf("  Saved %d patches...\n", patch_id))
    }
    # Increment patch counter
    patch_id <- patch_id + 1
  }
  # Calculate final statistics
  total_saved <- patch_id - 1  
  # Print summary
  cat(sprintf("\n=== Summary ===\n"))
  cat(sprintf("Total polygons processed: %d\n", n_polygons))
  cat(sprintf("Patches saved: %d\n", total_saved))
  cat(sprintf("Skipped (too close to edge): %d\n", skipped_count))
  # Return count of saved patches
  return(total_saved)
}

# Execute patch extraction -----------------------------------------------------

cat("\n=== Starting patch extraction from polygons ===\n")
# Call the function with parameters
patches_extracted <- extract_polygon_centroid_patches(
  image = Kremetart_image,              # The satellite image
  polygons = Kremetart_tree_polygons,   # Tree crown polygons with Class attribute
  patch_size = 128,                     # 128 x 128 pixel patches
  img_dir = img_dir,                    # Where to save image patches
  mask_dir = mask_dir,                  # Where to save mask patches
  season_prefix = "maps_july",           # Prefix for filenames
  target_classes = 1              # Which classes to process
)
cat(sprintf("\n=== Completed: %d patches extracted ===\n", patches_extracted))

# Organize Class 1 patches into separate folders--------------------------------

# Define the current directories where all patches are stored
current_img_dir  <- "~/Chafika/image patches/Mapungubwe_july_polygons/128_8bands/images"

current_mask_dir <- "~/Chafika/image patches/Mapungubwe_july_polygons/128_8bands/masks"

# Define new directories specifically for class 1 patches
class1_img_dir  <- "~/Chafika/image patches/New_patches_128pixels/class1_images"

class1_mask_dir <- "~/Chafika/image patches/New_patches_128pixels/class1_masks"
# This will be the new folder to store only class 1 mask patches

# Create the new directories if they don't exist
dir.create(class1_img_dir,  recursive = TRUE, showWarnings = FALSE)
# dir.create() makes a new folder on your computer
# recursive = TRUE means it will create parent folders if needed
# showWarnings = FALSE prevents error messages if the folder already exists

dir.create(class1_mask_dir, recursive = TRUE, showWarnings = FALSE)
# Same as above, but for the mask directory

# Get list of class 1 files

# List all .tif files in the images directory
all_img_files <- list.files(current_img_dir, pattern = "\\.tif$", full.names = TRUE)
# list.files() finds all files in a directory
# pattern = "\\.tif$" is a regular expression that matches files ending in .tif
# full.names = TRUE returns complete file paths instead of just filenames

# Filter to get only class 1 image files
class1_img_files <- all_img_files[grepl("_class1_", all_img_files)]
# grepl() searches for the text "_class1_" in each filename
# Returns TRUE/FALSE for each file, and we use this to subset the vector
# This selects only files with "_class1_" in their name

# List all .tif files in the masks directory
all_mask_files <- list.files(current_mask_dir, pattern = "\\.tif$", full.names = TRUE)
# Same as for images, but for the masks directory

# Filter to get only class 1 mask files
class1_mask_files <- all_mask_files[grepl("_class1_", all_mask_files)]
# Same filtering logic as for images

# Display what will be moved

cat(sprintf("\nFound %d class 1 image patches\n", length(class1_img_files)))
# length() counts the number of files in the vector
# cat() prints text to the console
# sprintf() formats the text with the number inserted

cat(sprintf("Found %d class 1 mask patches\n", length(class1_mask_files)))
# Same as above, but for mask files

# Show a few examples
cat("\nExample filenames:\n")
# Prints a header for the examples

if (length(class1_img_files) > 0) {
  # if() checks whether we found any files
  # length(class1_img_files) > 0 means "if there's at least one file"
  
  cat("Images:\n")
  print(head(basename(class1_img_files), 3))
  # head() takes the first few items from a vector (default is 6, we specify 3)
  # basename() extracts just the filename without the full path
  # print() displays the result
}

if (length(class1_mask_files) > 0) {
  cat("Masks:\n")
  print(head(basename(class1_mask_files), 3))
}

# ------------------------------------------------------------------------------
# Move the files
# ------------------------------------------------------------------------------

# Loop through each class 1 image file
for (img_file in class1_img_files) {
  # for loop iterates through each file path in the vector
  # img_file is a variable that holds the current file path in each iteration
  
  # Get just the filename (without the path)
  filename <- basename(img_file)
  # basename() extracts the filename from the full path
  # Example: "/path/to/maps_july_class1_poly1_00001.tif" becomes "maps_july_class1_poly1_00001.tif"
  
  # Create the destination path
  destination <- file.path(class1_img_dir, filename)
  # file.path() combines directory path and filename with correct separator
  # This creates the complete path where we want to move the file
  
  # Move the file
  # file.rename(img_file, destination)
  # file.rename() moves (or renames) a file from one location to another
  # First argument is the current location, second is the new location
  # This removes the file from the original location
}

#cat(sprintf("Moved %d image patches\n", length(class1_img_files)))
# Reports how many files were moved

#cat("\nMoving class 1 mask patches...\n")

# Loop through each class 1 mask file
#for (mask_file in class1_mask_files) {
# Same structure as the image loop above

# Get just the filename
# filename <- basename(mask_file)

# Create the destination path
# destination <- file.path(class1_mask_dir, filename)

# Move the file
#  file.rename(mask_file, destination)
# }

cat(sprintf("Moved %d mask patches\n", length(class1_mask_files)))

cat("\nMoving class 1 image patches (skip masked)...\n")

moved_count <- 0
skipped_count <- 0

for (img_file in class1_img_files) {
  
  filename <- basename(img_file)
  
  # ---- find matching mask ----
  mask_file <- file.path(current_mask_dir, filename)
  
  if (!file.exists(mask_file)) {
    cat("Mask not found for:", filename, "\n")
    next
  }
  
  # ---- read image patch ----
  img_rast <- rast(img_file)
  
  vals <- values(img_rast, mat = FALSE)
  
  # ---- check for masked pixels ----
  if (any(is.na(vals))) {
    
    skipped_count <- skipped_count + 1
    next
    
  }
  
  # ---- move image ----
  img_dest <- file.path(class1_img_dir, filename)
  file.rename(img_file, img_dest)
  
  # ---- move mask ----
  mask_dest <- file.path(class1_mask_dir, filename)
  file.rename(mask_file, mask_dest)
  
  moved_count <- moved_count + 1
}

cat("Moved:", moved_count, "\n")
cat("Skipped (masked):", skipped_count, "\n")

# -------------------------------
# Verify the move
# -------------------------------

cat("\n=== Verification ===\n")

# Count files in new directories
new_img_count <- length(list.files(class1_img_dir, pattern = "\\.tif$"))
# Counts how many .tif files are now in the new class 1 images folder

new_mask_count <- length(list.files(class1_mask_dir, pattern = "\\.tif$"))
# Counts how many .tif files are now in the new class 1 masks folder

cat(sprintf("Class 1 images in new folder: %d\n", new_img_count))
cat(sprintf("Class 1 masks in new folder: %d\n", new_mask_count))

# Count remaining files in original directories
remaining_img_count <- length(list.files(current_img_dir, pattern = "\\.tif$"))
# Counts how many .tif files remain in the original images folder (should be class 2 only)

remaining_mask_count <- length(list.files(current_mask_dir, pattern = "\\.tif$"))
# Counts how many .tif files remain in the original masks folder (should be class 2 only)

cat(sprintf("Remaining images in original folder: %d (class 2)\n", remaining_img_count))
cat(sprintf("Remaining masks in original folder: %d (class 2)\n", remaining_mask_count))

cat("\n=== Completed: Class 1 patches organized ===\n")

###############################################################################

# ==============================================================================
# Baobab Crown Size classification 
# Three size classes were defined based on crown area, 
# # We define area thresholds in square meters (35 m²) for crown sizes by
# creating equal-sized groups
# Large crowns: Top 33%
# Medium crowns: Middle 33%
# Small crowns: Bottom 33%
# ==============================================================================

# Our input directories where original images and masks are stored

image_dir  <- "~/Chafika/image patches/New_patches_128pixels/class1_images"
mask_dir   <- "~/Chafika/image patches/New_patches_128pixels/class1_masks"

# We define the output directory where classified files / folders will be stored

output_dir <- "~/Chafika/image patches/New_patches_128pixels"


# We define area thresholds in square meters for crown sizes
# NOTE: These will be calculated dynamically to create equal-sized groups
# Large crowns: Top 33%
# Medium crowns: Middle 33%
# Small crowns: Bottom 33%

pixel_size <- 0.3  # pixel resolution (meters per pixel)

# ==============================================================================
# Output folders
# ==============================================================================

# Each size category gets separate folders for images and masks

dirs <- c("small/images", "small/masks",
          "medium/images", "medium/masks",
          "large/images", "large/masks")

# Create each directory if it doesn't exist

for (d in dirs) {
  dir.create(file.path(output_dir, d),
             recursive = TRUE,      
             showWarnings = FALSE)  # Don't warn if directory already exists
}

# ==============================================================================
# File matching
# ==============================================================================

# Get full file paths for all images and masks

images <- list.files(image_dir, full.names = TRUE, pattern = "\\.tif$|\\.tiff$")
masks  <- list.files(mask_dir,  full.names = TRUE, pattern = "\\.tif$|\\.tiff$")

# Helper function to extract filename without extension
# Example: "tree_001.tif" becomes "tree_001"

stem <- function(x) tools::file_path_sans_ext(basename(x))

# Create a dataframe matching each image with its corresponding mask

df <- data.frame(
  image = images,  # Full path to image file
  mask  = masks[match(stem(images), stem(masks))],  # Match mask by filename
  name  = stem(images),  # Filename without extension
  stringsAsFactors = FALSE  # Keep strings as text, not factors
)

# Remove rows where no matching mask was found

df <- df[!is.na(df$mask), ]

cat("Found", nrow(df), "matching image-mask pairs\n\n")

# ==============================================================================
# Measure Baobab
# ==============================================================================
# Initialize columns for storing measurements

df$total_crown_pixels <- NA_integer_  # Total pixels with label == 1
df$largest_crown_pixels <- NA_integer_  # Pixels in the largest individual crown
df$num_crowns <- NA_integer_  # Number of separate crowns detected
df$crown_area_m2 <- NA_real_  # Area of largest crown in square meters

# Loop through each image-mask pair

cat("Processing masks to measure crown areas...\n")
cat("Note: Classification based on LARGEST crown when multiple crowns present\n\n")

for (i in seq_len(nrow(df))) {
  
  # Load the mask image as a raster
  
  m <- rast(df$mask[i])
  
  # Convert raster to matrix for easier manipulation
  
  m_vals <- values(m)
  
  # Count total pixels where mask value equals 1 (all crown pixels)
  
  df$total_crown_pixels[i] <- sum(m_vals == 1, na.rm = TRUE)
  
  # If there are crown pixels, find connected components (individual crowns)
  
  if (df$total_crown_pixels[i] > 0) {
    
    # Create a binary mask (1 for crown, 0 for background)
    
    binary_mask <- m
    values(binary_mask) <- ifelse(m_vals == 1, 1, 0)
    
    # We use terra's patches() function to identify connected components
    # This labels each separate crown with a unique ID
    
    crown_patches <- patches(binary_mask, directions = 8, zeroAsNA = TRUE)
    
    # Get the patch labels
    
    patch_vals <- values(crown_patches)
    
    # Count number of unique crowns (excluding NA)
    
    unique_crowns <- unique(patch_vals[!is.na(patch_vals)])
    df$num_crowns[i] <- length(unique_crowns)
    
    # Calculate size of each crown
    
    crown_sizes <- table(patch_vals)
    
    # Find the largest crown (maximum pixel count)
    
    df$largest_crown_pixels[i] <- as.integer(max(crown_sizes))
    
    # Convert largest crown pixels to area in square meters
    # Formula: (number of pixels) × (pixel_size)²
    
    df$crown_area_m2[i] <- df$largest_crown_pixels[i] * (pixel_size^2)
    
  } else {
    
    # No crown pixels found
    
    df$largest_crown_pixels[i] <- 0
    df$num_crowns[i] <- 0
    df$crown_area_m2[i] <- 0
  }
  
  # Progress indicator
  
  if (i %% 50 == 0) {
    cat("  Processed", i, "of", nrow(df), "masks\n")
  }
}

# Remove rows where no crown pixels were found (empty masks)

df <- df[df$total_crown_pixels > 0, ]

cat("\nAfter removing empty masks:", nrow(df), "images remain\n")

# Print statistics about multiple crowns

multi_crown <- df[df$num_crowns > 1, ]
cat("Patches with multiple crowns:", nrow(multi_crown), 
    "(", round(100 * nrow(multi_crown) / nrow(df), 1), "%)\n")
cat("Average crowns per patch:", round(mean(df$num_crowns), 2), "\n")
cat("Max crowns in a single patch:", max(df$num_crowns), "\n\n")

# ==============================================================================
# Classify patches into equal-sizes groups based on crown area
# ==============================================================================

# Calculate the 33.33rd and 66.67th percentiles of crown areas
# This divides the data into three equal groups

quantiles <- quantile(df$crown_area_m2, probs = c(1/3, 2/3))
area_small_max <- quantiles[1]   # 33.33rd percentile
area_medium_max <- quantiles[2]  # 66.67th percentile

# Classify each patch based on these calculated thresholds
# This ensures approximately equal numbers in each category

df$class <- ifelse(df$crown_area_m2 <= area_small_max, "small",
                   ifelse(df$crown_area_m2 <= area_medium_max, "medium", "large"))

# Print classification summary to console

cat("\n=== CLASSIFICATION SUMMARY (EQUAL-SIZED GROUPS) ===\n")
cat("Classification based on tertiles (33.33rd and 66.67th percentiles)\n\n")
cat("Small:  ≤ ", round(area_small_max, 2), " m² (bottom 33%)\n", sep = "")
cat("Medium: ", round(area_small_max, 2), "-", round(area_medium_max, 2), " m² (middle 33%)\n", sep = "")
cat("Large:  > ", round(area_medium_max, 2), " m² (top 33%)\n\n", sep = "")
cat("Counts:\n")
print(table(df$class))  # Show how many patches in each category
cat("\nPercentages:\n")
print(round(prop.table(table(df$class)) * 100, 1))  # Show percentages
cat("\n")

# ==============================================================================
# Copy each image and mask to the appropriate size category folder
# ==============================================================================

cat("Copying files to classified folders...\n")
for (i in seq_len(nrow(df))) {
  cls <- df$class[i]  # Get the classification (small/medium/large)
  
  # Copy image file to classified folder
  
  file.copy(df$image[i],
            file.path(output_dir, cls, "images", basename(df$image[i])),
            overwrite = TRUE)
  
  # Copy mask file to classified folder
  
  file.copy(df$mask[i],
            file.path(output_dir, cls, "masks", basename(df$mask[i])),
            overwrite = TRUE)
  
  # Progress indicator
  
  if (i %% 50 == 0) {
    cat("  Copied", i, "of", nrow(df), "files\n")
  }
}

# ==============================================================================
# Save a summary table with filename, measurements, and classification
# ==============================================================================

write.csv(df[, c("name", "num_crowns", "total_crown_pixels", 
                 "largest_crown_pixels", "crown_area_m2", "class")],
          file.path(output_dir, "crown_size_summary128.csv"),
          row.names = FALSE)  # Don't include row numbers in CSV

cat("Summary CSV saved\n")

# ==============================================================================
# Define the custom theme
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

# ==============================================================================
# Create a histogram showing the distribution of crown areas 
# with equal-sized groups
# ==============================================================================

p <- ggplot(df, aes(x = crown_area_m2)) +  # Plot crown area in m²
  geom_histogram(bins = 40, fill = "steelblue", color = "black") +  # 40 bins
  geom_vline(xintercept = area_small_max, linetype = "dashed", color = "red", linewidth = 1) +  # 33rd percentile
  geom_vline(xintercept = area_medium_max, linetype = "dashed", color = "red", linewidth = 1) +  # 67th percentile
  annotate("text", x = area_small_max, y = Inf, label = paste0(round(area_small_max, 1), " m²"), 
           vjust = 2, hjust = -0.1, color = "red") +
  annotate("text", x = area_medium_max, y = Inf, label = paste0(round(area_medium_max, 1), " m²"), 
           vjust = 2, hjust = -0.1, color = "red") +
  labs(title = "Distribution of Crown Sizes (Equal Groups)",
       x = "Crown Area (m²)",  # Label shows area in square meters
       y = "Count") +
  theme_fancy()  

# Save the histogram as a PNG file
ggsave(file.path(output_dir, "crown_size_histogram128.png"),
       p, width = 8, height = 5, dpi = 600)

cat("Histogram saved\n")

# ==============================================================================
# Create a bar plot showing distribution of number of crowns per patch
# ==============================================================================
p2 <- ggplot(df, aes(x = factor(num_crowns))) +
  geom_bar(fill = "darkgreen", color = "black") +
  labs(title = "Distribution of Crown Count per Patch",
       x = "Number of Crowns in Patch",
       y = "Number of Patches") +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  theme_fancy()

# Save the crown count distribution plot
ggsave(file.path(output_dir, "crown_count_distribution128.png"),
       p2, width = 8, height = 5, dpi = 600)

cat("Crown count distribution plot saved\n")

# ==============================================================================
# Create a density plot showing the distribution of crown areas 
# with equal-sized groups marked
# ==============================================================================

p <- ggplot(df, aes(x = crown_area_m2)) +  
  geom_density(fill = "steelblue", color = "black", alpha = 0.3) +  
  geom_vline(xintercept = area_small_max, linetype = "dashed", color = "red", linewidth = 1) +  
  geom_vline(xintercept = area_medium_max, linetype = "dashed", color = "red", linewidth = 1) +  
  annotate("text", x = area_small_max, y = Inf, label = paste0(round(area_small_max, 1), " m²"), 
           vjust = 2, hjust = -0.1, color = "red") +
  annotate("text", x = area_medium_max, y = Inf, label = paste0(round(area_medium_max, 1), " m²"), 
           vjust = 2, hjust = -0.1, color = "red") +
  labs(title = "Distribution of Crown Sizes (Density - Equal Groups)",
       x = "Crown Area (m²)",  
       y = "Density") +
  theme_fancy()  

ggsave(file.path(output_dir, "crown_size_density128.png"),
       p, width = 8, height = 5, dpi = 600)
cat("Density plot saved\n")