# ==============================================================================
# Baobab crown mapping: polygon-level evaluation
# ------------------------------------------------------------------------------

# Evaluates predicted crowns against manually digitised trees. The script 
# calculates the intersection of each predicted crown with each reference crown,
# and matches them one-to-one.

# We applied an IoU = 0.1, because the main aim is population counting. A tree 
# can therefore count as detected despite imperfect crown delineation. 

# Outputs:
#   - Full evaluation table
#   - Summary table
#   - Summary figure
#   - Evaluated predicted polygons
#   - Evaluated reference polygons
# ==============================================================================


# ---- Packages ----------------------------------------------------------------

if (!requireNamespace("terra", quietly = TRUE)) {
  stop(
    "The 'terra' package is required. ",
    "Install it with install.packages('terra')"
  )
}


# ---- Small helpers ----------------------------------------------------------

polygon_area_m2 <- function(polygons) {
  
  if (nrow(polygons) == 0) {
    return(numeric(0))
  }
  
  terra::expanse(
    polygons,
    unit = "m",
    transform = isTRUE(terra::is.lonlat(polygons))
  )
}

safe_ratio <- function(numerator, denominator) {
  
  if (length(numerator) != 1 || length(denominator) != 1) {
    return(NA_real_)
  }
  
  if (!is.finite(numerator) || !is.finite(denominator) || denominator == 0) {
    return(NA_real_)
  }
  
  numerator / denominator
}


assign_sequential_ids <- function(polygons, id_name) {
  
  if (nrow(polygons) == 0) {
    return(polygons)
  }
  
  attribute_table <- terra::values(polygons)
  
  
  if (!is.data.frame(attribute_table)) {
    attribute_table <- data.frame(row.names = seq_len(nrow(polygons)))
  }
  
  attribute_table[[id_name]] <- seq_len(nrow(polygons))
  
  terra::values(polygons) <- attribute_table
  
  polygons
}


keep_polygons_inside <- function(polygons, boundary) {
  
  if (nrow(polygons) == 0) {
    return(polygons)
  }
  
  
  centroid_points <- terra::centroids(polygons, inside = TRUE)
  
  intersection_matrix <- terra::relate(
    centroid_points,
    boundary,
    relation = "intersects"
  )
  
  inside_boundary <- rowSums(
    matrix(intersection_matrix, nrow = nrow(centroid_points))
  ) > 0
  
  polygons[inside_boundary, ]
}


# ---- Geometry Validation-----------------------------------------------------

repair_geometries <- function(polygons, label = "polygons", verbose = TRUE) {
  
  if (nrow(polygons) == 0) {
    return(polygons)
  }
  
  valid <- terra::is.valid(polygons, messages = FALSE)
  
  invalid_count <- sum(!valid, na.rm = TRUE)
  
  if (invalid_count > 0) {
    
    if (verbose) {
      cat(sprintf("  Repairing %d invalid %s\n", invalid_count, label))
    }
    
    polygons <- terra::makeValid(polygons)
  }
  
  polygons
}


remove_empty_geometries <- function(polygons) {
  
  if (nrow(polygons) == 0) {
    return(polygons)
  }
  
  polygons[!terra::is.empty(polygons), ]
}


# ---- Layer preparation ------------------------------------------------------

prepare_layers <- function(reference_path,
                           raster_path,
                           foreground_value = 1,
                           boundary_path = NA_character_,
                           min_predicted_area_m2 = 0,
                           verbose = TRUE) {
  
  if (!file.exists(raster_path)) {
    stop("Raster not found: ", raster_path)
  }
  
  if (!file.exists(reference_path)) {
    stop("Reference polygons not found: ", reference_path)
  }
  
  
  # --- Predicted crowns ---
  
  prediction_raster <- terra::rast(raster_path)
  
  # Use the first band only; some exports carry an alpha or mask band.
  prediction_raster <- prediction_raster[[1]]
  
  # Comparison produces a logical raster: TRUE where the class is crown.
  foreground_raster <- prediction_raster == foreground_value
  
  patch_raster <- terra::patches(
    foreground_raster,
    directions = 8,
    zeroAsNA = TRUE
  )
  
  # dissolve = TRUE merges all cells sharing a patch number into one polygon.
  predicted <- terra::as.polygons(patch_raster, dissolve = TRUE)
  
  predicted <- remove_empty_geometries(predicted)
  
  # Drop the background polygon, which carries an NA patch number.
  if (nrow(predicted) > 0) {
    patch_values <- terra::values(predicted)[[1]]
    predicted <- predicted[!is.na(patch_values), ]
  }
  
  # Repair before any area or overlap computation.
  predicted <- repair_geometries(
    predicted,
    label = "predicted polygons",
    verbose = verbose
  )
  
  predicted <- remove_empty_geometries(predicted)
  
  if (min_predicted_area_m2 > 0 && nrow(predicted) > 0) {
    
    predicted_areas <- polygon_area_m2(predicted)
    
    kept <- is.finite(predicted_areas) &
      predicted_areas >= min_predicted_area_m2
    
    if (verbose && any(!kept)) {
      cat(sprintf(
        "  Removing %d predicted polygons below %.2f m2\n",
        sum(!kept),
        min_predicted_area_m2
      ))
    }
    
    predicted <- predicted[kept, ]
  }
  
  
  # --- Reference crowns ---
  
  reference <- terra::vect(reference_path)
  
  
  if (!terra::same.crs(reference, predicted)) {
    
    if (verbose) {
      cat("  Reprojecting reference polygons to raster CRS\n")
    }
    
    reference <- terra::project(reference, terra::crs(predicted))
  }
  
  reference <- repair_geometries(
    reference,
    label = "reference polygons",
    verbose = verbose
  )
  
  reference <- remove_empty_geometries(reference)
  
  
  # --- Optional evaluation boundary ---
  
  boundary_supplied <- length(boundary_path) == 1 &&
    !is.na(boundary_path) &&
    nzchar(boundary_path)
  
  if (boundary_supplied) {
    
    if (!file.exists(boundary_path)) {
      stop("Boundary file not found: ", boundary_path)
    }
    
    boundary <- terra::vect(boundary_path)
    
    if (!terra::same.crs(boundary, predicted)) {
      boundary <- terra::project(boundary, terra::crs(predicted))
    }
    
    boundary <- repair_geometries(
      boundary,
      label = "boundary",
      verbose = verbose
    )
    
    boundary <- terra::aggregate(boundary)
    
    predicted <- keep_polygons_inside(predicted, boundary)
    reference <- keep_polygons_inside(reference, boundary)
    
    predicted <- remove_empty_geometries(predicted)
    reference <- remove_empty_geometries(reference)
  }
  
  predicted <- assign_sequential_ids(predicted, "pred_id")
  reference <- assign_sequential_ids(reference, "ref_id")
  
  if (verbose) {
    cat(sprintf(
      "  %d predicted polygons | %d reference trees\n",
      nrow(predicted),
      nrow(reference)
    ))
  }
  
  list(predicted = predicted, reference = reference)
}


# ---- Candidate pairs --------------------------------------------------------

find_candidate_pairs <- function(predicted, reference) {
  
  empty_pairs <- data.frame(
    pred_row = integer(0),
    ref_row = integer(0)
  )
  
  if (nrow(predicted) == 0 || nrow(reference) == 0) {
    return(empty_pairs)
  }
  
  pair_index <- tryCatch(
    
    terra::relate(
      predicted,
      reference,
      relation = "intersects",
      pairs = TRUE
    ),
    
    error = function(e) {
      
      full_matrix <- terra::relate(
        predicted,
        reference,
        relation = "intersects"
      )
      
      which(
        matrix(full_matrix, nrow = nrow(predicted)),
        arr.ind = TRUE
      )
    }
  )
  
  if (is.null(pair_index) || length(pair_index) == 0) {
    return(empty_pairs)
  }
  
  # Force a plain two-column integer matrix, whichever route produced it.
  pair_index <- matrix(as.integer(pair_index), ncol = 2)
  
  if (nrow(pair_index) == 0) {
    return(empty_pairs)
  }
  
  data.frame(
    pred_row = pair_index[, 1],
    ref_row  = pair_index[, 2]
  )
}


# ---- Pairwise overlap -------------------------------------------------------

compute_pair_overlap_m2 <- function(predicted, reference, pred_row, ref_row) {
  
  tryCatch(
    {
      intersection <- terra::intersect(
        predicted[pred_row, ],
        reference[ref_row, ]
      )
      
      if (is.null(intersection) || nrow(intersection) == 0) {
        
        0
        
      } else {
        
        # Polygons that share only an edge intersect as a line, whose area is
        # zero; sum() handles multi-part intersections.
        total_area <- sum(polygon_area_m2(intersection), na.rm = TRUE)
        
        if (is.finite(total_area)) total_area else 0
      }
    },
    error = function(e) NA_real_
  )
}


compute_candidate_pairs <- function(predicted,
                                    reference,
                                    verbose = TRUE,
                                    progress_every = 1000) {
  
  empty_pairs <- data.frame(
    pred_row     = integer(0),
    ref_row      = integer(0),
    pred_id      = integer(0),
    ref_id       = integer(0),
    pred_area_m2 = numeric(0),
    ref_area_m2  = numeric(0),
    overlap_m2   = numeric(0),
    iou          = numeric(0)
  )
  
  candidates <- find_candidate_pairs(predicted, reference)
  
  if (nrow(candidates) == 0) {
    return(empty_pairs)
  }
  
  predicted_areas <- polygon_area_m2(predicted)
  reference_areas <- polygon_area_m2(reference)
  
  n_pairs <- nrow(candidates)
  
  overlap_m2 <- rep(NA_real_, n_pairs)
  
  for (i in seq_len(n_pairs)) {
    
    overlap_m2[i] <- compute_pair_overlap_m2(
      predicted = predicted,
      reference = reference,
      pred_row  = candidates$pred_row[i],
      ref_row   = candidates$ref_row[i]
    )
    
    if (verbose && progress_every > 0 && i %% progress_every == 0) {
      cat(sprintf("    %d / %d intersections\n", i, n_pairs))
    }
  }
  
  # NA marks a pair terra could not process; report and discard it.
  failed <- is.na(overlap_m2)
  
  if (any(failed)) {
    
    if (verbose) {
      cat(sprintf(
        "  %d polygon intersections could not be calculated\n",
        sum(failed)
      ))
    }
    
    candidates <- candidates[!failed, , drop = FALSE]
    overlap_m2 <- overlap_m2[!failed]
  }
  
  if (nrow(candidates) == 0) {
    return(empty_pairs)
  }
  
  pred_area_m2 <- predicted_areas[candidates$pred_row]
  ref_area_m2  <- reference_areas[candidates$ref_row]
  
  # IoU = shared area / combined area. The union is the sum of both areas less
  # the overlap, which would otherwise be counted twice.
  union_m2 <- pred_area_m2 + ref_area_m2 - overlap_m2
  
  iou <- ifelse(
    is.finite(union_m2) & union_m2 > 0,
    overlap_m2 / union_m2,
    0
  )
  
  data.frame(
    pred_row     = candidates$pred_row,
    ref_row      = candidates$ref_row,
    pred_id      = predicted$pred_id[candidates$pred_row],
    ref_id       = reference$ref_id[candidates$ref_row],
    pred_area_m2 = pred_area_m2,
    ref_area_m2  = ref_area_m2,
    overlap_m2   = overlap_m2,
    iou          = iou,
    stringsAsFactors = FALSE
  )
}


# ---- One-to-one matching ----------------------------------------------------

match_one_to_one <- function(candidate_pairs, min_iou = 0.1) {
  
  if (nrow(candidate_pairs) == 0) {
    return(candidate_pairs)
  }
  
  eligible <- candidate_pairs[candidate_pairs$iou >= min_iou, , drop = FALSE]
  
  if (nrow(eligible) == 0) {
    return(eligible)
  }
  
  eligible <- eligible[order(-eligible$iou), , drop = FALSE]
  
  # Logical vectors act as ledgers of which features have been claimed.
  pred_claimed <- logical(max(candidate_pairs$pred_row))
  ref_claimed  <- logical(max(candidate_pairs$ref_row))
  
  accept <- logical(nrow(eligible))
  
  for (i in seq_len(nrow(eligible))) {
    
    p <- eligible$pred_row[i]
    r <- eligible$ref_row[i]
    
    if (!pred_claimed[p] && !ref_claimed[r]) {
      
      accept[i] <- TRUE
      
      pred_claimed[p] <- TRUE
      ref_claimed[r]  <- TRUE
    }
  }
  
  eligible[accept, , drop = FALSE]
}


# ---- Segmentation rates -----------------------------------------------------

# Over-segmentation  : a reference crown covered by more than one prediction.
# Under-segmentation : a prediction covering more than one reference crown.

calculate_segmentation_rates <- function(candidate_pairs,
                                         n_predicted,
                                         n_reference,
                                         min_overlap_fraction = 0.1) {
  
  if (nrow(candidate_pairs) == 0) {
    return(list(oversegmentation_rate = 0, undersegmentation_rate = 0))
  }
  
  smaller_area <- pmin(
    candidate_pairs$pred_area_m2,
    candidate_pairs$ref_area_m2
  )
  
  overlap_fraction <- candidate_pairs$overlap_m2 / smaller_area
  
  substantive <- candidate_pairs[
    is.finite(overlap_fraction) & overlap_fraction >= min_overlap_fraction,
    ,
    drop = FALSE
  ]
  
  if (nrow(substantive) == 0) {
    return(list(oversegmentation_rate = 0, undersegmentation_rate = 0))
  }
  
  # table() counts how many partners each feature has.
  ref_overlap_count  <- table(substantive$ref_id)
  pred_overlap_count <- table(substantive$pred_id)
  
  list(
    oversegmentation_rate = safe_ratio(
      sum(ref_overlap_count > 1),
      n_reference
    ),
    undersegmentation_rate = safe_ratio(
      sum(pred_overlap_count > 1),
      n_predicted
    )
  )
}


# ---- Site evaluation --------------------------------------------------------

evaluate_site <- function(reference_path,
                          raster_path,
                          min_iou = 0.1,
                          foreground_value = 1,
                          boundary_path = NA_character_,
                          min_predicted_area_m2 = 0,
                          segmentation_min_overlap_fraction = 0.1,
                          label_fields = list(),
                          output_dir = NULL,
                          output_prefix = "run",
                          verbose = TRUE) {
  
  layers <- prepare_layers(
    reference_path        = reference_path,
    raster_path           = raster_path,
    foreground_value      = foreground_value,
    boundary_path         = boundary_path,
    min_predicted_area_m2 = min_predicted_area_m2,
    verbose               = verbose
  )
  
  predicted <- layers$predicted
  reference <- layers$reference
  
  n_predicted <- nrow(predicted)
  n_reference <- nrow(reference)
  
  pairs <- compute_candidate_pairs(
    predicted = predicted,
    reference = reference,
    verbose   = verbose
  )
  
  matches <- match_one_to_one(pairs, min_iou = min_iou)
  
  
  # --- Counts ---
  
  tp <- nrow(matches)                 # reference trees correctly detected
  fp <- n_predicted - tp              # predictions matching no reference tree
  fn <- n_reference - tp              # reference trees missed
  
  
  # --- Detection metrics ---
  
  precision <- safe_ratio(tp, tp + fp)   # predictions that are correct
  recall    <- safe_ratio(tp, tp + fn)   # real trees that were found
  
  # F1 is the harmonic mean of precision and recall.
  
  f1 <- if (is.na(precision) || is.na(recall) || precision + recall == 0) {
    NA_real_
  } else {
    2 * precision * recall / (precision + recall)
  }
  
  # Mean IoU over matched pairs only: delineation quality of detected trees.
  
  mean_iou <- if (tp > 0) mean(matches$iou, na.rm = TRUE) else NA_real_
  
  # Count bias: relative over- or under-count of the population.
  
  count_bias <- safe_ratio(n_predicted - n_reference, n_reference)
  
  segmentation_rates <- calculate_segmentation_rates(
    candidate_pairs      = pairs,
    n_predicted          = n_predicted,
    n_reference          = n_reference,
    min_overlap_fraction = segmentation_min_overlap_fraction
  )
  
  
  # --- Write labelled polygons ---
  
  if (!is.null(output_dir)) {
    
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    
    if (n_predicted > 0) {
      
      predicted$status <- ifelse(
        predicted$pred_id %in% matches$pred_id,
        "TP",
        "FP"
      )
      
      # match() returns the position of each prediction within the match table,
      # or NA where it was not matched
      
      predicted$iou <- if (tp > 0) {
        matches$iou[match(predicted$pred_id, matches$pred_id)]
      } else {
        NA_real_
      }
      
      terra::writeVector(
        predicted,
        file.path(
          output_dir,
          paste0(output_prefix, "_predicted_evaluated.gpkg")
        ),
        overwrite = TRUE
      )
    }
    
    if (n_reference > 0) {
      
      reference$status <- ifelse(
        reference$ref_id %in% matches$ref_id,
        "TP",
        "FN"
      )
      
      reference$iou <- if (tp > 0) {
        matches$iou[match(reference$ref_id, matches$ref_id)]
      } else {
        NA_real_
      }
      
      terra::writeVector(
        reference,
        file.path(
          output_dir,
          paste0(output_prefix, "_reference_evaluated.gpkg")
        ),
        overwrite = TRUE
      )
    }
  }
  
  # Internal consistency check: every polygon must be accounted for exactly once.
  
  stopifnot(
    tp + fp == n_predicted,
    tp + fn == n_reference
  )
  
  result <- data.frame(
    reference_trees    = n_reference,
    predicted_polygons = n_predicted,
    iou_threshold      = min_iou,
    tp                 = tp,
    fp                 = fp,
    fn                 = fn,
    precision          = precision,
    recall             = recall,
    f1                 = f1,
    iou                = mean_iou,
    count_bias_pct     = 100 * count_bias,
    oversegmentation_rate_pct  = 100 * segmentation_rates$oversegmentation_rate,
    undersegmentation_rate_pct = 100 * segmentation_rates$undersegmentation_rate,
    stringsAsFactors = FALSE
  )
  
  # Prepend the site, season and resolution labels.
  
  if (length(label_fields) > 0) {
    result <- cbind(
      as.data.frame(label_fields, stringsAsFactors = FALSE),
      result
    )
  }
  
  result
}


# ---- Batch runner -----------------------------------------------------------

make_safe_label <- function(text) {
  
  cleaned <- gsub("[^A-Za-z0-9]+", "_", text)
  
  gsub("^_|_$", "", cleaned)
}


# run_evaluation_batch(): evaluates every row of the manifest.

run_evaluation_batch <- function(manifest,
                                 min_iou = 0.1,
                                 output_dir = NULL,
                                 min_predicted_area_m2 = 0,
                                 verbose = TRUE) {
  
  results_list <- vector("list", nrow(manifest))
  
  failed_runs <- list()
  
  for (i in seq_len(nrow(manifest))) {
    
    row <- manifest[i, , drop = FALSE]
    
    run_label <- make_safe_label(
      sprintf("%s_%s_%sm", row$site, row$season, row$resolution)
    )
    
    if (verbose) {
      cat(sprintf("[%d/%d] %s\n", i, nrow(manifest), run_label))
    }
    
    # Normalise the boundary entry: absent, NA or empty all mean "no boundary".
    
    boundary_value <- NA_character_
    
    if ("boundary_path" %in% names(row)) {
      
      candidate_boundary <- as.character(row$boundary_path)
      
      if (length(candidate_boundary) == 1 &&
          !is.na(candidate_boundary) &&
          nzchar(candidate_boundary)) {
        
        boundary_value <- candidate_boundary
      }
    }
    
    outcome <- tryCatch(
      
      evaluate_site(
        reference_path        = as.character(row$reference_path),
        raster_path           = as.character(row$raster_path),
        min_iou               = min_iou,
        foreground_value      = 1,
        boundary_path         = boundary_value,
        min_predicted_area_m2 = min_predicted_area_m2,
        label_fields = list(
          site       = as.character(row$site),
          season     = as.character(row$season),
          resolution = as.numeric(row$resolution)
        ),
        output_dir    = output_dir,
        output_prefix = run_label,
        verbose       = verbose
      ),
      
      # The handler returns a marker rather than assigning to an outer variable,
      
      error = function(e) {
        list(failed = TRUE, message = conditionMessage(e))
      }
    )
    
    if (is.list(outcome) && isTRUE(outcome$failed)) {
      
      failed_runs[[length(failed_runs) + 1]] <- list(
        run     = run_label,
        message = outcome$message
      )
      
      cat(sprintf("  FAILED: %s\n", outcome$message))
      
      next
    }
    
    results_list[[i]] <- outcome
  }
  
  # Filter() with Negate(is.null) keeps only the runs that produced a result.
  
  valid_results <- Filter(Negate(is.null), results_list)
  
  if (length(valid_results) == 0) {
    stop("No evaluation runs completed successfully.")
  }
  
  results_table <- do.call(rbind, valid_results)
  
  results_table <- results_table[
    order(
      results_table[["site"]],
      results_table[["season"]],
      results_table[["resolution"]]
    ),
    ,
    drop = FALSE
  ]
  
  rownames(results_table) <- NULL
  
  if (length(failed_runs) > 0) {
    
    failed_text <- vapply(
      failed_runs,
      function(x) paste0(x$run, ": ", x$message),
      character(1)
    )
    
    warning(
      "The following runs failed:\n",
      paste(failed_text, collapse = "\n"),
      call. = FALSE
    )
    
    attr(results_table, "failed_runs") <- failed_text
  }
  
  results_table
}


# ---- Result formatting ------------------------------------------------------

display_names <- c(
  site                       = "Site",
  season                     = "Season",
  resolution                 = "Resolution",
  reference_trees            = "Reference trees",
  predicted_polygons         = "Predicted trees",
  iou_threshold              = "IoU threshold",
  tp                         = "TP",
  fp                         = "FP",
  fn                         = "FN",
  precision                  = "Precision",
  recall                     = "Recall",
  f1                         = "F1",
  iou                        = "IoU",
  count_bias_pct             = "Count bias (%)",
  oversegmentation_rate_pct  = "Over-segmentation (%)",
  undersegmentation_rate_pct = "Under-segmentation (%)"
)


summary_columns <- c(
  "site", "season", "resolution",
  "reference_trees", "predicted_polygons",
  "precision", "recall", "f1", "iou",
  "count_bias_pct",
  "oversegmentation_rate_pct",
  "undersegmentation_rate_pct"
)


# format_results_table(): selects columns, rounds numbers and applies the

format_results_table <- function(results_table, columns = NULL, digits = 3) {
  
  if (!is.null(columns)) {
    
    missing_columns <- setdiff(columns, names(results_table))
    
    if (length(missing_columns) > 0) {
      stop(
        "Requested column(s) not present: ",
        paste(missing_columns, collapse = ", ")
      )
    }
    
    results_table <- results_table[, columns, drop = FALSE]
  }
  
  # vapply() with logical(1) guarantees a logical vector, one entry per column.
  
  numeric_columns <- vapply(results_table, is.numeric, logical(1))
  
  results_table[numeric_columns] <- lapply(
    results_table[numeric_columns],
    round,
    digits
  )
  
  headings <- display_names[names(results_table)]
  
  # Any column without a mapping keeps its original name.
  
  headings[is.na(headings)] <- names(results_table)[is.na(headings)]
  
  names(results_table) <- unname(headings)
  
  data.frame(
    results_table,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}


# ---- Summary figure ---------------------------------------------------------

save_summary_plot <- function(summary_table,
                              file_path,
                              width = 14,
                              height = NULL,
                              dpi = 300) {
  
  if (!requireNamespace("gridExtra", quietly = TRUE)) {
    warning("The 'gridExtra' package is needed for the summary figure.")
    return(invisible(NULL))
  }
  
  if (is.null(height)) {
    height <- max(3, 0.28 * (nrow(summary_table) + 2))
  }
  
  # Format each column as text so that decimals align in the rendered table.
  
  as_text <- function(column) {
    
    if (is.numeric(column) && any(column %% 1 != 0, na.rm = TRUE)) {
      return(sprintf("%.3f", column))
    }
    
    format(column, trim = TRUE)
  }
  
  text_table <- as.data.frame(
    lapply(summary_table, as_text),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  table_theme <- gridExtra::ttheme_minimal(
    base_size = 9,
    core    = list(fg_params = list(hjust = 0.5, x = 0.5)),
    colhead = list(fg_params = list(fontface = "bold", hjust = 0.5, x = 0.5))
  )
  
  table_grob <- gridExtra::tableGrob(
    text_table,
    rows  = NULL,
    theme = table_theme
  )
  
  grDevices::png(
    filename = file_path,
    width    = width,
    height   = height,
    units    = "in",
    res      = dpi,
    bg       = "white"
  )
  
  on.exit(grDevices::dev.off(), add = TRUE)
  
  grid::grid.newpage()
  grid::grid.draw(table_grob)
  
  invisible(file_path)
}


# ---- Paths ------------------------------------------------------------------

base_dir <- "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet"

raster_dir <- file.path(base_dir, "Predicted_Maps/Evaluation_Site_Maps")

crowns_dir <- file.path(base_dir, "Tree_Crowns_Polygons")

ellipses_dir <- file.path(crowns_dir, "Clipped")

output_dir <- file.path(base_dir, "Evaluation_Results")


# ---- Reference polygons -----------------------------------------------------

site_keys <- c("game_reserve", "irrigated", "kremetart", "mapungubwe")

# setNames() attaches the site keys as names, so the vector acts as a lookup.
reference_lookup <- setNames(
  file.path(
    ellipses_dir,
    paste0(site_keys, "_baobab_polygons_clipped.gpkg")
  ),
  site_keys
)


# ---- Evaluation boundaries --------------------------------------------------

boundary_lookup <- c(
  game_reserve = NA_character_,
  irrigated    = NA_character_,
  kremetart    = file.path(ellipses_dir, "Kremetart_Evaluation_Site.gpkg"),
  mapungubwe   = file.path(ellipses_dir, "Mapungubwe_Evaluation_Site.gpkg")
)


# ---- Labels -----------------------------------------------------------------

site_labels <- c(
  game_reserve = "Game Reserve",
  irrigated    = "Irrigated",
  kremetart    = "Kremetart",
  mapungubwe   = "Mapungubwe"
)

season_labels <- c(
  feb      = "February",
  july     = "July",
  feb_july = "Combined"
)


# ---- Manifest ---------------------------------------------------------------

build_manifest_from_directory <- function(raster_dir,
                                          reference_lookup,
                                          boundary_lookup = NULL,
                                          site_labels = NULL,
                                          season_labels = NULL) {
  
  if (!dir.exists(raster_dir)) {
    stop("Raster directory not found: ", raster_dir)
  }
  
  raster_files <- list.files(
    raster_dir,
    pattern     = "_binary\\.tif$",
    ignore.case = TRUE,
    full.names  = FALSE
  )
  
  if (length(raster_files) == 0) {
    stop("No binary rasters found in: ", raster_dir)
  }
  
  # Capture groups: site, season, resolution.
  
  pattern <- paste0(
    "^(.+?)_baobab_map_",
    "(feb_july|feb|july)_",
    "([0-9.]+)m_binary\\.tif$"
  )
  
  parsed <- regmatches(
    raster_files,
    regexec(pattern, raster_files, ignore.case = TRUE)
  )
  
  recognised <- lengths(parsed) == 4
  
  if (any(!recognised)) {
    
    warning(
      "Skipping files that do not match the naming convention:\n  ",
      paste(raster_files[!recognised], collapse = "\n  "),
      call. = FALSE
    )
    
    parsed       <- parsed[recognised]
    raster_files <- raster_files[recognised]
  }
  
  if (length(parsed) == 0) {
    stop("No rasters matched the expected naming convention.")
  }
  
  site_key   <- tolower(vapply(parsed, `[`, character(1), 2))
  season_key <- tolower(vapply(parsed, `[`, character(1), 3))
  resolution <- as.numeric(vapply(parsed, `[`, character(1), 4))
  
  missing_reference <- setdiff(unique(site_key), names(reference_lookup))
  
  if (length(missing_reference) > 0) {
    stop(
      "No reference polygons listed for site(s): ",
      paste(missing_reference, collapse = ", ")
    )
  }
  
  relabel <- function(keys, labels) {
    
    if (is.null(labels)) {
      return(keys)
    }
    
    out <- unname(labels[keys])
    
    ifelse(is.na(out), keys, out)
  }
  
  manifest <- data.frame(
    site           = relabel(site_key, site_labels),
    season         = relabel(season_key, season_labels),
    resolution     = resolution,
    reference_path = unname(reference_lookup[site_key]),
    boundary_path  = if (is.null(boundary_lookup)) {
      NA_character_
    } else {
      unname(boundary_lookup[site_key])
    },
    raster_path    = file.path(raster_dir, raster_files),
    stringsAsFactors = FALSE
  )
  
  missing_reference_files <- unique(
    manifest$reference_path[!file.exists(manifest$reference_path)]
  )
  
  if (length(missing_reference_files) > 0) {
    stop(
      "Reference file(s) not found:\n  ",
      paste(missing_reference_files, collapse = "\n  ")
    )
  }
  
  boundary_present <- !is.na(manifest$boundary_path)
  
  missing_boundary_files <- unique(
    manifest$boundary_path[
      boundary_present & !file.exists(manifest$boundary_path)
    ]
  )
  
  if (length(missing_boundary_files) > 0) {
    stop(
      "Boundary file(s) not found:\n  ",
      paste(missing_boundary_files, collapse = "\n  ")
    )
  }
  
  sorted_manifest <- manifest[
    order(
      manifest[["site"]],
      manifest[["season"]],
      manifest[["resolution"]]
    ),
    ,
    drop = FALSE
  ]
  
  rownames(sorted_manifest) <- NULL
  
  sorted_manifest
}


evaluation_manifest <- build_manifest_from_directory(
  raster_dir       = raster_dir,
  reference_lookup = reference_lookup,
  boundary_lookup  = boundary_lookup,
  site_labels      = site_labels,
  season_labels    = season_labels
)

print(evaluation_manifest)

cat(sprintf("\n%d runs found in the manifest.\n\n", nrow(evaluation_manifest)))


# ---- Run evaluation ---------------------------------------------------------

# min_iou = we have applied 0.1 to prioritis detection over exact crown shape.

evaluation_results <- run_evaluation_batch(
  manifest              = evaluation_manifest,
  min_iou               = 0.1,
  output_dir            = output_dir,
  min_predicted_area_m2 = 0,
  verbose               = TRUE
)


# ---- Format results ---------------------------------------------------------

full_table <- format_results_table(evaluation_results)

summary_table <- format_results_table(
  evaluation_results,
  columns = summary_columns
)

print(summary_table)


# ---- Save results -----------------------------------------------------------

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(
  full_table,
  file.path(output_dir, "evaluation_results.csv"),
  row.names = FALSE
)

write.csv(
  summary_table,
  file.path(output_dir, "evaluation_summary.csv"),
  row.names = FALSE
)

save_summary_plot(
  summary_table,
  file.path(output_dir, "evaluation_summary_plot.png")
)

cat(sprintf(
  "\nCompleted %d of %d evaluation runs.\nResults written to: %s\n",
  nrow(evaluation_results),
  nrow(evaluation_manifest),
  output_dir
))

failed_runs <- attr(evaluation_results, "failed_runs")

if (!is.null(failed_runs)) {
  cat("\nFailed runs:\n  ", paste(failed_runs, collapse = "\n  "), "\n", sep = "")
}

#===============================================================================
# This section contains helper functions for calculating polygon areas, that 
# will be used to calculate the areas of the reference and predicted polygons 
# in square meters.The areas are later used to generate crown classes.
#-------------------------------------------------------------------------------

# Load the folder with polygons to calculate the areas

evaluated_dir <- file.path(
  "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet",
  "Evaluation_Results"
)

# helper function --------------------------------------------------------------

# calculate_polygon_areas_m2()

calculate_polygon_areas_m2 <- function(polygons) {
  
  if (nrow(polygons) == 0) {
    return(numeric(0))
  }
  
  terra::expanse(
    polygons,
    unit      = "m",
    transform = isTRUE(terra::is.lonlat(polygons))
  )
}

# divide_safely()

divide_safely <- function(numerator, denominator) {
  
  if (!is.finite(numerator) || !is.finite(denominator) || denominator == 0) {
    return(NA_real_)
  }
  
  numerator / denominator
}


clean_label <- function(text) {
  
  cleaned <- gsub("[^A-Za-z0-9]+", "_", text)
  
  gsub("^_|_$", "", cleaned)
}


# ---- Locating the evaluated files --------------------------------------------

list_evaluated_files <- function(evaluated_dir) {
  
  if (!dir.exists(evaluated_dir)) {
    stop("Evaluation folder not found: ", evaluated_dir)
  }
  
  file_names <- list.files(
    evaluated_dir,
    pattern     = "_evaluated\\.gpkg$",
    ignore.case = TRUE,
    full.names  = FALSE
  )
  
  if (length(file_names) == 0) {
    stop("No evaluated GeoPackages found in: ", evaluated_dir)
  }
  
  # Capture groups: site, season, resolution, layer type.

  pattern <- paste0(
    "^(.+)_",                       # site, 
    "([A-Za-z]+)_",                 # season, 
    "([0-9]+(?:_[0-9]+)?)m_",       # resolution,
    "(predicted|reference)",        # which layer
    "_evaluated\\.gpkg$"
  )
  
  # regexec() locates the match and its capture groups
  
  parsed <- regmatches(
    file_names,
    regexec(pattern, file_names, ignore.case = TRUE)
  )
  
  recognised <- lengths(parsed) == 5
  
  if (any(!recognised)) {
    
    warning(
      "Skipping files that do not match the expected naming pattern:\n  ",
      paste(file_names[!recognised], collapse = "\n  "),
      call. = FALSE
    )
    
    parsed     <- parsed[recognised]
    file_names <- file_names[recognised]
  }
  
  if (length(parsed) == 0) {
    stop("No evaluated GeoPackages matched the expected naming pattern.")
  }
  
  # `[` used as a function pulls the nth element out of each match.
  site_label   <- vapply(parsed, `[`, character(1), 2)
  season_label <- vapply(parsed, `[`, character(1), 3)
  resolution_t <- vapply(parsed, `[`, character(1), 4)
  layer_type   <- tolower(vapply(parsed, `[`, character(1), 5))
  
  data.frame(
    site       = gsub("_", " ", site_label),          # restore the space
    season     = season_label,
    
    resolution = as.numeric(sub("_", ".", resolution_t, fixed = TRUE)),
    source     = layer_type,                          # predicted or reference
    file_path  = file.path(evaluated_dir, file_names),
    stringsAsFactors = FALSE
  )
}


# ---- Loading one layer -------------------------------------------------------

load_polygon_areas <- function(file_path, source_label) {
  
  polygons <- terra::vect(file_path)
  
  if (nrow(polygons) == 0) {
    return(NULL)
  }
  
  attribute_table <- as.data.frame(terra::values(polygons))
  
  # The identifier column differs between the two layers.
  id_field <- if (source_label == "predicted") "pred_id" else "ref_id"
  
  polygon_id <- if (id_field %in% names(attribute_table)) {
    as.integer(attribute_table[[id_field]])
  } else {
    seq_len(nrow(polygons))       # fall back to row order
  }
  
  status <- if ("status" %in% names(attribute_table)) {
    as.character(attribute_table[["status"]])
  } else {
    rep(NA_character_, nrow(polygons))
  }
  
  iou <- if ("iou" %in% names(attribute_table)) {
    as.numeric(attribute_table[["iou"]])
  } else {
    rep(NA_real_, nrow(polygons))
  }
  
  data.frame(
    source     = source_label,
    polygon_id = polygon_id,
    status     = status,
    iou        = iou,
    area_m2    = calculate_polygon_areas_m2(polygons),
    stringsAsFactors = FALSE
  )
}


# ---- Building the full area table --------------------------------------------

build_area_table <- function(file_index, verbose = TRUE) {
  
  collected <- vector("list", nrow(file_index))
  
  for (i in seq_len(nrow(file_index))) {
    
    entry <- file_index[i, , drop = FALSE]
    
    layer_table <- tryCatch(
      load_polygon_areas(entry$file_path, entry$source),
      error = function(e) {
        warning(
          "Could not read ", basename(entry$file_path), ": ",
          conditionMessage(e),
          call. = FALSE
        )
        NULL
      }
    )
    
    if (is.null(layer_table)) {
      next
    }
    
    collected[[i]] <- cbind(
      site       = entry$site,
      season     = entry$season,
      resolution = entry$resolution,
      layer_table,
      stringsAsFactors = FALSE
    )
    
    if (verbose) {
      cat(sprintf(
        "  %s: %d polygons\n",
        basename(entry$file_path),
        nrow(layer_table)
      ))
    }
  }
  
  collected <- Filter(Negate(is.null), collected)
  
  if (length(collected) == 0) {
    stop("None of the evaluated GeoPackages could be read.")
  }
  
  area_table <- do.call(rbind, collected)
  
  rownames(area_table) <- NULL
  
  area_table
}


# ---- Recalculating the detection metrics -------------------------------------

summarise_area_table <- function(area_table) {
  
  runs <- unique(area_table[, c("site", "season", "resolution")])
  
  runs <- runs[order(runs$site, runs$season, runs$resolution), , drop = FALSE]
  
  summaries <- vector("list", nrow(runs))
  
  mean_or_na <- function(values) {
    if (length(values) == 0) NA_real_ else mean(values, na.rm = TRUE)
  }
  
  for (i in seq_len(nrow(runs))) {
    
    in_run <- area_table$site == runs$site[i] &
      area_table$season == runs$season[i] &
      area_table$resolution == runs$resolution[i]
    
    run_rows <- area_table[in_run, , drop = FALSE]
    
    is_predicted <- run_rows$source == "predicted"
    is_reference <- run_rows$source == "reference"
    
    # Counting true positives on the predicted side only prevents the matched
    # pairs from being counted twice.
    
    tp <- sum(is_predicted & run_rows$status == "TP", na.rm = TRUE)
    fp <- sum(is_predicted & run_rows$status == "FP", na.rm = TRUE)
    fn <- sum(is_reference & run_rows$status == "FN", na.rm = TRUE)
    
    precision <- divide_safely(tp, tp + fp)
    recall    <- divide_safely(tp, tp + fn)
    
    # F1 is the harmonic mean of precision and recall.
    f1 <- if (is.na(precision) || is.na(recall) || precision + recall == 0) {
      NA_real_
    } else {
      2 * precision * recall / (precision + recall)
    }
    
    summaries[[i]] <- data.frame(
      site       = runs$site[i],
      season     = runs$season[i],
      resolution = runs$resolution[i],
      tp         = tp,
      fp         = fp,
      fn         = fn,
      precision  = precision,
      recall     = recall,
      f1         = f1,
      total_predicted_area_m2 = sum(run_rows$area_m2[is_predicted], na.rm = TRUE),
      total_reference_area_m2 = sum(run_rows$area_m2[is_reference], na.rm = TRUE),
      mean_tp_predicted_area_m2 = mean_or_na(
        run_rows$area_m2[is_predicted & run_rows$status == "TP"]
      ),
      mean_tp_reference_area_m2 = mean_or_na(
        run_rows$area_m2[is_reference & run_rows$status == "TP"]
      ),
      mean_fp_area_m2 = mean_or_na(
        run_rows$area_m2[is_predicted & run_rows$status == "FP"]
      ),
      mean_fn_area_m2 = mean_or_na(
        run_rows$area_m2[is_reference & run_rows$status == "FN"]
      ),
      stringsAsFactors = FALSE
    )
  }
  
  summary_table <- do.call(rbind, summaries)
  
  rownames(summary_table) <- NULL
  
  summary_table
}

# ---- Writing one CSV per site -----------------------------------------------

write_area_tables_by_site <- function(area_table,
                                      output_dir,
                                      digits = 3,
                                      verbose = TRUE) {
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # split() returns a named list, one element per site.
  by_site <- split(area_table, area_table$site)
  
  written <- character(0)
  
  for (site_name in names(by_site)) {
    
    site_rows <- by_site[[site_name]]
    
    site_rows <- site_rows[
      order(
        site_rows$season,
        site_rows$resolution,
        site_rows$source,
        site_rows$polygon_id
      ),
      ,
      drop = FALSE
    ]
    
    site_rows$area_m2 <- round(site_rows$area_m2, digits)
    site_rows$iou     <- round(site_rows$iou, digits)
    
    rownames(site_rows) <- NULL
    
    file_path <- file.path(
      output_dir,
      paste0("polygon_areas_", clean_label(site_name), ".csv")
    )
    
    write.csv(site_rows, file_path, row.names = FALSE)
    
    written <- c(written, file_path)
    
    if (verbose) {
      cat(sprintf("  %s: %d rows written\n", site_name, nrow(site_rows)))
    }
  }
  
  invisible(written)
}

# ---- Run ---------------------------------------------------------------------

evaluated_files <- list_evaluated_files(evaluated_dir)

cat(sprintf("\n%d evaluated layers found.\n\n", nrow(evaluated_files)))

polygon_areas <- build_area_table(evaluated_files)

cat("\nWriting one CSV per site\n")

write_area_tables_by_site(
  area_table = polygon_areas,
  output_dir = evaluated_dir
)

write.csv(
  polygon_areas,
  file.path(evaluated_dir, "polygon_areas_all_sites.csv"),
  row.names = FALSE
)

area_summary <- summarise_area_table(polygon_areas)

write.csv(
  area_summary,
  file.path(evaluated_dir, "polygon_areas_summary.csv"),
  row.names = FALSE
)

cat("\nMetrics recalculated from the loaded polygons:\n\n")

print(
  area_summary[, c(
    "site", "season", "resolution",
    "tp", "fp", "fn", "precision", "recall", "f1"
  )],
  digits = 3
)

cat(sprintf("\nResults written to: %s\n", evaluated_dir))