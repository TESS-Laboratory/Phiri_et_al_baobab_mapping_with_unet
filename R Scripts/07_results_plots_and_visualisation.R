#-------------------------------------------------------------------------------
# Plotting the graphs to visualise the performance of U-Net baobab detection 
# across spatial resolution,
#-------------------------------------------------------------------------------

# The script plots U-Net baobab detection performance across spatial resolution, 
# dataset, crown size, seasonality, and land-use context. It summarises the 
# model performance metrics and produces plots showing how precision, recall, 
# and F1-score vary across these factors.

# Loag the required packages

library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(patchwork)
library(grid)     

# --------theme-----------------------------------------------------------------

theme_fancy <- function() {
  theme_bw() +
    theme(
      text = element_text(family = "sans"),
      axis.text = element_text(size = 10, colour = "black"),
      axis.title = element_text(size = 10, colour = "black"),
      axis.line.x = element_line(linewidth = 0.5, colour = "black"),
      axis.line.y = element_line(linewidth = 0.5, colour = "black"),
      axis.ticks = element_line(linewidth = 0.4, colour = "black"),
      panel.border = element_blank(),
      panel.grid = element_blank(),
      plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"),
      plot.title = element_text(size = 10, vjust = 1, hjust = 0.5, colour = "black"),
      legend.text = element_text(size = 10, colour = "black"),
      legend.title = element_text(size = 9, colour = "black"),
      legend.key.size = unit(0.9, "line"),
      legend.background = element_rect(colour = "black", fill = "transparent",
                                       linewidth = 2, linetype = "blank")
    )
}

# --- Paths ------------------------------------------------------------------

in_dir  <- "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet/Evaluation_Results"
out_dir <- "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet/Manuscripts Plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load the files -----------------------------------------------------------

csv_files <- list.files(in_dir, pattern = "\\.csv$", full.names = TRUE)
stopifnot(length(csv_files) > 0)

raw <- lapply(csv_files, function(f) {
  df <- read.csv(f, check.names = FALSE, stringsAsFactors = FALSE)
  df$source <- tools::file_path_sans_ext(basename(f))
  df
}) |>
  bind_rows()

message("Files found:   ", paste(basename(csv_files), collapse = ", "))
message("Columns found: ", paste(names(raw), collapse = ", "))

# --- Locate the columns we need -----------------------------------------------

find_col <- function(df, candidates) {
  hit <- which(tolower(names(df)) %in% candidates)
  if (length(hit) == 0) NULL else names(df)[hit[1]]
}

site_col   <- find_col(raw, c("site", "location", "area", "study_site"))
season_col <- find_col(raw, c("season", "dataset", "month", "period"))
res_col    <- find_col(raw, c("resolution", "spatial_resolution", "grain",
                              "gsd", "pixel_size", "res"))

prec_col <- find_col(raw, c("precision", "prec"))
rec_col  <- find_col(raw, c("recall", "rec"))
f1_col   <- find_col(raw, c("f1", "f1_score", "f1-score", "f1score", "fscore"))
stopifnot(!is.null(prec_col), !is.null(rec_col), !is.null(f1_col))

# Fall back to the filename only where a column is absent
site_raw   <- if (is.null(site_col))   raw$source else raw[[site_col]]
season_raw <- if (is.null(season_col)) raw$source else raw[[season_col]]
res_raw    <- if (is.null(res_col))    raw$source else raw[[res_col]]

# --- Standardise --------------------------------------------------------------

tidy_site <- function(x) {
  x <- tolower(trimws(as.character(x)))
  case_when(grepl("kremetart",  x) ~ "Kremetart",
            grepl("mapungubwe", x) ~ "Mapungubwe",
            TRUE ~ NA_character_)
}

tidy_season <- function(x) {
  x <- tolower(trimws(as.character(x)))
  case_when(grepl("combin", x)                ~ "Combined",
            grepl("feb", x) & grepl("jul", x) ~ "Combined",
            grepl("feb", x)                   ~ "February",
            grepl("jul", x)                   ~ "July",
            TRUE ~ NA_character_)
}

tidy_res <- function(x) {
  suppressWarnings(as.numeric(sub(".*?([0-9]+\\.?[0-9]*)\\s*m?$", "\\1",
                                  trimws(as.character(x)))))
}

combined <- raw |>
  mutate(
    site       = tidy_site(site_raw),
    season     = tidy_season(season_raw),
    resolution = tidy_res(res_raw),
    Precision  = as.numeric(.data[[prec_col]]),
    Recall     = as.numeric(.data[[rec_col]]),
    F1         = as.numeric(.data[[f1_col]])
  )

# Report before discarding anything

message("Rows read: ", nrow(combined),
        " | unparsed site: ", sum(is.na(combined$site)),
        " | season: ",        sum(is.na(combined$season)),
        " | resolution: ",    sum(is.na(combined$resolution)))

combined <- combined |> filter(!is.na(site), !is.na(season), !is.na(resolution))
stopifnot(nrow(combined) > 0)

# --- Summary table ----------------------------------------------------------
summary_tbl <- combined |>
  group_by(Site = site, Season = season, Resolution = resolution) |>
  summarise(across(c(Precision, Recall, F1), \(x) mean(x, na.rm = TRUE)),
            .groups = "drop") |>
  arrange(Site, Season, Resolution) |>
  mutate(across(c(Precision, Recall, F1), \(x) round(x, 3)))

write.csv(summary_tbl, file.path(out_dir, "spatial_grain_summary.csv"),
          row.names = FALSE)

# --- Groups to plot (edit here only) ------------------------------------------

wanted_groups <- tibble::tribble(
  ~Site,        ~Season,
  "Kremetart",  "Combined",
  "Kremetart",  "February",
  "Kremetart",  "July",
  "Mapungubwe", "July"
)

summary_selected <- summary_tbl |> semi_join(wanted_groups, by = c("Site", "Season"))
stopifnot(nrow(summary_selected) > 0)

# --- Long format --------------------------------------------------------------

metric_cols  <- c("F1", "Precision", "Recall")
group_levels <- paste(wanted_groups$Site, wanted_groups$Season, sep = " \u2013 ")

group_colours <- setNames(c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"), group_levels)

long <- summary_selected |>
  select(Site, Season, Resolution, all_of(metric_cols)) |>
  pivot_longer(all_of(metric_cols), names_to = "metric", values_to = "score") |>
  mutate(
    metric = factor(metric,
                    levels = c("F1", "Precision", "Recall"),
                    labels = c("F1-score", "Precision", "Recall")),
    group_label = factor(paste(Site, Season, sep = " \u2013 "), levels = group_levels)
  )

# --- Figure -------------------------------------------------------------------

res_breaks <- sort(unique(long$Resolution))

main_plot <- ggplot(long, aes(x = Resolution, y = score,
                              colour = group_label, group = group_label)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3.0) +
  facet_wrap(~ metric, nrow = 1) +
  scale_x_log10(breaks = res_breaks, labels = res_breaks, minor_breaks = NULL) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_colour_manual(values = group_colours, drop = FALSE) +
  labs(x = "Spatial grain (m)", y = "Detection score", colour = "Site\u2013season") +
  theme_fancy() +
  theme(
    strip.text         = element_text(size = 10, face = "bold"),
    axis.title         = element_text(size = 10, face = "bold"),
    legend.position    = "bottom",
    legend.direction   = "horizontal",
    legend.key.width   = unit(1.0, "cm"),
    legend.key.height  = unit(0.4, "cm"),
    legend.box.spacing = unit(0.2, "cm"),
    panel.spacing.x    = unit(1, "lines"),
    plot.margin        = margin(10, 15, 10, 10)
  ) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE, title.position = "top"))

print(main_plot)

ggsave(file.path(out_dir, "spatial_grain_comparison_plot.png"),
       plot = main_plot, width = 21.0, height = 14.85,
       units = "cm", dpi = 600, bg = "white")

# --- Table plus figure --------------------------------------------------------

table_grob <- tableGrob(
  summary_selected,
  rows  = NULL,
  theme = ttheme_minimal(
    core    = list(fg_params = list(fontsize = 10, fontfamily = "sans",
                                    col = "black")),
    colhead = list(fg_params = list(fontsize = 10, fontfamily = "sans",
                                    fontface = "bold", col = "black"))
  )
)


ggsave(file.path(out_dir, "spatial_grain_summary_table.png"),
       plot = table_grob, width = 21.0, height = 14.85, dpi = 600, bg = "white")



# ==============================================================================
# Crown size versus resolution: July acquisitions at Kremetart and Mapungubwe
# ------------------------------------------------------------------------------
# This section compares crown size classes distributed in equal buckets (small,
# medium and large) 

# Input: the polygon area tables written by the area export script,
# Output: bar chart of precision, recall and F1 by crown size class,
# with one bar per resolution.
# ==============================================================================

# ---- Select the folder holding the area data----------------------------------

areas_dir <- file.path(
  "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet",
  "Evaluation_Results"
)

plot_dir <- file.path(
  "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet",
  "Manuscripts Plots"
)

sites_wanted  <- c("Kremetart", "Mapungubwe")
season_wanted <- "July"

resolution_levels <- c("0.3 m", "0.9 m", "2.7 m", "8.1 m")

resolution_colors <- c("0.3 m" = "#264653",
                       "0.9 m" = "#2A9D8F",
                       "2.7 m" = "#E9C46A",
                       "8.1 m" = "#d62728")


# ---- Load the polygon areas --------------------------------------------------

load_polygon_areas <- function(areas_dir) {
  
  combined_file <- file.path(areas_dir, "polygon_areas_all_sites.csv")
  
  if (file.exists(combined_file)) {
    return(read.csv(combined_file, stringsAsFactors = FALSE))
  }
  
  site_files <- list.files(
    areas_dir,
    pattern    = "^polygon_areas_(?!all_sites|summary).+\\.csv$",
    perl       = TRUE,
    full.names = TRUE
  )
  
  if (length(site_files) == 0) {
    stop("No polygon area tables found in: ", areas_dir)
  }
  
  bind_rows(lapply(site_files, read.csv, stringsAsFactors = FALSE))
}

polygon_areas <- load_polygon_areas(areas_dir)

required_columns <- c("site", "season", "resolution",
                      "source", "polygon_id", "status", "area_m2")

missing_columns <- setdiff(required_columns, names(polygon_areas))

if (length(missing_columns) > 0) {
  stop("Missing column(s) in the area table: ",
       paste(missing_columns, collapse = ", "))
}


# ---- Restrict to the July runs at the two sites ------------------------------

crowns <- polygon_areas |>
  filter(site %in% sites_wanted,
         season == season_wanted) |>
  mutate(
    # sprintf("%g") drops any trailing zero, so 0.30 prints as "0.3".
    resolution_label = factor(sprintf("%g m", resolution),
                              levels = resolution_levels),
    site_label = factor(site, levels = sites_wanted)
  )

if (nrow(crowns) == 0) {
  stop("No rows remain for season '", season_wanted,
       "' at: ", paste(sites_wanted, collapse = ", "))
}

if (any(is.na(crowns$resolution_label))) {
  stop("Resolution(s) not listed in resolution_levels: ",
       paste(unique(crowns$resolution[is.na(crowns$resolution_label)]),
             collapse = ", "))
}

cat("Rows retained by site and resolution:\n")
print(table(crowns$site_label, crowns$resolution_label))

cat("\nStatus values by layer:\n")
print(table(crowns$source, crowns$status, useNA = "ifany"))


# ---- Equal-count crown size bins from the reference polygons -----------------

reference_trees <- crowns |>
  filter(source == "reference") |>
  group_by(site_label, polygon_id) |>
  summarise(area_m2 = median(area_m2, na.rm = TRUE), .groups = "drop")

if (nrow(reference_trees) == 0) {
  stop("No reference polygons remain after filtering.")
}

# Four evenly spaced quantiles give three bins holding roughly equal numbers of
# reference trees, so no class is defined by a handful of polygons.

bin_breaks <- quantile(reference_trees$area_m2,
                       probs = seq(0, 1, length.out = 4),
                       na.rm = TRUE)

bin_breaks[1]                  <- floor(min(crowns$area_m2, na.rm = TRUE))
bin_breaks[length(bin_breaks)] <- ceiling(max(crowns$area_m2, na.rm = TRUE))

bin_breaks <- unique(bin_breaks)

if (length(bin_breaks) < 4) stop("Could not form 3 distinct bins.")

bin_labels <- c(
  paste0("Small\n(",  round(bin_breaks[1], 1), "-", round(bin_breaks[2], 1), " m\u00B2)"),
  paste0("Medium\n(", round(bin_breaks[2], 1), "-", round(bin_breaks[3], 1), " m\u00B2)"),
  paste0("Large\n(",  round(bin_breaks[3], 1), "-", round(bin_breaks[4], 1), " m\u00B2)")
)

crowns <- crowns |>
  mutate(crown_bin = cut(area_m2,
                         breaks         = bin_breaks,
                         labels         = bin_labels,
                         include.lowest = TRUE))


# ---- Precision, recall and F1 by size class ----------------------------------

# TP and FP are counted on the predicted side and FN on the reference side, so
# each matched tree contributes to the counts only once.

tp_fp <- crowns |>
  filter(source == "predicted") |>
  group_by(site_label, resolution_label, crown_bin) |>
  summarise(TP = sum(status == "TP", na.rm = TRUE),
            FP = sum(status == "FP", na.rm = TRUE),
            .groups = "drop")

fn <- crowns |>
  filter(source == "reference") |>
  group_by(site_label, resolution_label, crown_bin) |>
  summarise(FN = sum(status == "FN", na.rm = TRUE),
            .groups = "drop")

metrics <- full_join(
  tp_fp, fn,
  by = c("site_label", "resolution_label", "crown_bin")
) |>
  complete(site_label, resolution_label, crown_bin) |>
  mutate(
    across(c(TP, FP, FN), ~ replace_na(.x, 0L)),
    Precision  = ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_),
    Recall     = ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_),
    `F1-score` = ifelse((Precision + Recall) > 0,
                        2 * Precision * Recall / (Precision + Recall),
                        NA_real_)
  )


# ---- Long format -------------------------------------------------------------

metrics_long <- metrics |>
  pivot_longer(cols      = c(Precision, Recall, `F1-score`),
               names_to  = "metric",
               values_to = "value") |>
  mutate(metric = factor(metric,
                         levels = c("F1-score", "Precision", "Recall")))


# ---- Reference tree counts per site and size class ---------------------------

n_labels <- reference_trees |>
  mutate(crown_bin = cut(area_m2,
                         breaks         = bin_breaks,
                         labels         = bin_labels,
                         include.lowest = TRUE)) |>
  group_by(site_label, crown_bin) |>
  summarise(n = n(), .groups = "drop") |>
  crossing(metric = factor(c("F1-score", "Precision", "Recall"),
                           levels = c("F1-score", "Precision", "Recall"))) |>
  mutate(label = paste0("n = ", n))


# ---- Plot --------------------------------------------------------------------

p <- ggplot(metrics_long,
            aes(x = crown_bin, y = value, fill = resolution_label)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(data = n_labels,
            aes(x = crown_bin, y = 1.06, label = label),
            inherit.aes = FALSE,
            size = 3, color = "grey30") +
  facet_grid(site_label ~ metric) +
  # drop = FALSE keeps all four resolutions in the legend even where one is
  # absent from a panel.
  scale_fill_manual(values = resolution_colors, drop = FALSE) +
  scale_y_continuous(limits = c(0, 1.12),
                     breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0))) +
  labs(x    = "Crown size class",
       y    = "Detection Score",
       fill = "Resolution") +
  theme_fancy() +
  theme(
    legend.position = "bottom",
    axis.text.x     = element_text(angle = 45, hjust = 1, size = 10),
    panel.spacing.x = unit(1, "lines"),
    strip.text      = element_text(face = "bold")
  )

print(p)

# ---- Save -------------------------------------------------------------------

dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(
  file.path(plot_dir, "crown_sizes_performance_plot.png"),
  plot   = p,
  width  = 21.0,
  height = 14.85,
  units  = "cm",
  dpi    = 600,
  bg     = "white"
)


# ==============================================================================
# Kremetart: detection score against spatial grain, by imagery seasonality
# acquisition
# Output: one line per season, precision, recall and F1 in separate panels.
# ==============================================================================

# ---- Load the file------------------------------------------------------------

areas_dir <- file.path(
  "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet",
  "Evaluation_Results"
)

plot_dir <- file.path(
  "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet",
  "Manuscripts Plots"
)

site_wanted <- "Kremetart"

composition_lookup <- c(
  "February" = "Single (wet)",
  "July"     = "Single (dry)",
  "Combined" = "Multi (dry+wet)"
)

composition_colours <- c(
  "Single (wet)"    = "lightseagreen",
  "Single (dry)"    = "black",
  "Multi (dry+wet)" = "firebrick"
)

composition_shapes <- c(
  "Single (wet)"    = 17,
  "Single (dry)"    = 4,
  "Multi (dry+wet)" = 18
)


# ---- Load the polygon --------------------------------------------------------

site_file     <- file.path(areas_dir, paste0("polygon_areas_", site_wanted, ".csv"))
combined_file <- file.path(areas_dir, "polygon_areas_all_sites.csv")

areas_file <- if (file.exists(site_file)) site_file else combined_file

if (!file.exists(areas_file)) {
  stop("No polygon area table found in: ", areas_dir)
}

polygon_areas <- read.csv(areas_file, stringsAsFactors = FALSE)

required_columns <- c("site", "season", "resolution", "source", "status")

missing_columns <- setdiff(required_columns, names(polygon_areas))

if (length(missing_columns) > 0) {
  stop("Missing column(s) in the area table: ",
       paste(missing_columns, collapse = ", "))
}


# ---- Restrict to Kremetart and label the seasonality -------------------------

crowns <- polygon_areas |>
  filter(site == site_wanted) |>
  mutate(
    
    composition = factor(
      unname(composition_lookup[season]),
      levels = unname(composition_lookup)
    ),
  
    resolution_m = as.numeric(resolution)
  )

if (nrow(crowns) == 0) {
  stop("No rows found for site: ", site_wanted)
}

unmapped_seasons <- unique(
  polygon_areas$season[polygon_areas$site == site_wanted &
                         !polygon_areas$season %in% names(composition_lookup)]
)

if (length(unmapped_seasons) > 0) {
  stop("Season(s) not listed in composition_lookup: ",
       paste(unmapped_seasons, collapse = ", "))
}

cat("Rows per composition and resolution:\n")
print(table(crowns$composition, crowns$resolution_m))

cat("\nStatus values by layer:\n")
print(table(crowns$source, crowns$status, useNA = "ifany"))


# ---- Precision, recall and F1 ------------------------------------------------

# TP and FP are counted on the predicted side and FN on the reference side, so
# each matched tree is counted exactly once.

tp_fp <- crowns |>
  filter(source == "predicted") |>
  group_by(composition, resolution_m) |>
  summarise(TP = sum(status == "TP", na.rm = TRUE),
            FP = sum(status == "FP", na.rm = TRUE),
            .groups = "drop")

fn <- crowns |>
  filter(source == "reference") |>
  group_by(composition, resolution_m) |>
  summarise(FN = sum(status == "FN", na.rm = TRUE),
            .groups = "drop")

metrics <- full_join(tp_fp, fn, by = c("composition", "resolution_m")) |>
  mutate(
    across(c(TP, FP, FN), ~ replace_na(.x, 0L)),
    Precision  = ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_),
    Recall     = ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_),
    `F1-score` = ifelse((Precision + Recall) > 0,
                        2 * Precision * Recall / (Precision + Recall),
                        NA_real_)
  ) |>
 
  arrange(composition, resolution_m)

cat("\nDetection metrics:\n")
print(as.data.frame(metrics), digits = 3)


# ---- Long format -------------------------------------------------------------

metrics_long <- metrics |>
  pivot_longer(
    cols      = c(Precision, Recall, `F1-score`),
    names_to  = "metric",
    values_to = "value"
  ) |>
  mutate(metric = factor(metric,
                         levels = c("F1-score", "Precision", "Recall")))


# ---- Plot --------------------------------------------------------------------

seasonality <- ggplot(
  metrics_long,
  aes(x = resolution_m, y = value,
      colour = composition, group = composition, shape = composition)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ metric, nrow = 1) +
  scale_colour_manual(values = composition_colours) +
  scale_shape_manual(values = composition_shapes) +
  scale_x_log10(breaks = c(0.3, 0.9, 2.7, 8.1)) +
  scale_y_continuous(limits = c(0, 1),
                     breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(x      = "Spatial grain (m)",
       y      = "Detection score",
       colour = "Imagery composition",
       shape  = "Imagery composition") +
  theme_minimal(base_size = 12) +
  theme_fancy() +
  theme(
    axis.title            = element_text(size = 10, face = "bold"),
    axis.text             = element_text(size = 10, colour = "black"),
    strip.text            = element_text(size = 10, face = "bold"),
    legend.title          = element_text(size = 10),
    legend.text           = element_text(size = 10),
    legend.position       = "bottom",
    legend.direction      = "horizontal",
    legend.box            = "horizontal",
    legend.title.position = "top",
    legend.key.width      = unit(1.0, "cm"),
    legend.key.height     = unit(0.4, "cm"),
    legend.margin         = margin(0, 0, 0, 0),
    legend.box.spacing    = unit(0.2, "cm"),
    panel.grid.minor      = element_blank(),
    panel.spacing.x       = unit(1, "lines"),
    plot.margin           = margin(10, 15, 10, 10)
  ) +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE),
         shape  = guide_legend(nrow = 1, byrow = TRUE))

print(seasonality)

# ---- Save --------------------------------------------------------------------

dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(
  file.path(plot_dir, "seasonality_performance_plot.png"),
  plot   = seasonality,
  width  = 21.0,
  height = 14.85,
  units  = "cm",
  dpi    = 600,
  bg     = "white"
)


# ==============================================================================
# Detection score against spatial grain, by land-use type (July only)
# ------------------------------------------------------------------------------
# This section compares the irrigated cropland, game reserve and national park 
# sites for the July acquisition.
# ==============================================================================

# ---- Load the files-----------------------------------------------------------

areas_dir <- file.path(
  "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet",
  "Evaluation_Results"
)

plot_dir <- file.path(
  "C:/Workspace/Phiri_et_al_baobab_mapping_with_unet",
  "Manuscripts Plots"
)

season_wanted <- "July"

landuse_lookup <- c(
  "Irrigated"    = "Agriculture Land",
  "Game Reserve" = "Game Reserve",
  "Mapungubwe"   = "National Park"
)

landuse_colours <- c(
  "Agriculture Land" = "#000000",
  "Game Reserve"     = "#2A9D8F",
  "National Park"    = "#d62728"
)

landuse_shapes <- c(
  "Agriculture Land" = 4,
  "Game Reserve"     = 17,
  "National Park"    = 18
)


# ---- Load the polygon data----------------------------------------------------

load_polygon_areas <- function(areas_dir) {
  
  combined_file <- file.path(areas_dir, "polygon_areas_all_sites.csv")
  
  if (file.exists(combined_file)) {
    return(read.csv(combined_file, stringsAsFactors = FALSE))
  }
  
  # The negative lookahead excludes the combined and summary files, so that
  # nothing is read twice.
  
  site_files <- list.files(
    areas_dir,
    pattern    = "^polygon_areas_(?!all_sites|summary).+\\.csv$",
    perl       = TRUE,
    full.names = TRUE
  )
  
  if (length(site_files) == 0) {
    stop("No polygon area tables found in: ", areas_dir)
  }
  
  bind_rows(lapply(site_files, read.csv, stringsAsFactors = FALSE))
}

polygon_areas <- load_polygon_areas(areas_dir)

required_columns <- c("site", "season", "resolution",
                      "source", "polygon_id", "status")

missing_columns <- setdiff(required_columns, names(polygon_areas))

if (length(missing_columns) > 0) {
  stop("Missing column(s) in the area table: ",
       paste(missing_columns, collapse = ", "))
}


# ---- Restrict to the three sites at the July acquisition ---------------------

crowns <- polygon_areas |>
  filter(season == season_wanted,
         site %in% names(landuse_lookup)) |>
  mutate(
    
    landuse = factor(
      unname(landuse_lookup[site]),
      levels = unname(landuse_lookup)
    ),
    
    grain_m = as.numeric(resolution)
  )

if (nrow(crowns) == 0) {
  stop("No rows found for season '", season_wanted, "' at the three sites.")
}

missing_sites <- setdiff(names(landuse_lookup), unique(crowns$site))

if (length(missing_sites) > 0) {
  warning("No July rows found for site(s): ",
          paste(missing_sites, collapse = ", "),
          call. = FALSE)
}

cat("Rows per land-use type and grain:\n")
print(table(crowns$landuse, crowns$grain_m))

cat("\nStatus values by layer:\n")
print(table(crowns$source, crowns$status, useNA = "ifany"))


# ---- Precision, recall and F1 ------------------------------------------------

# TP and FP are counted on the predicted side and FN on the reference side, so
# each matched tree is counted exactly once.

tp_fp <- crowns |>
  filter(source == "predicted") |>
  group_by(landuse, grain_m) |>
  summarise(TP = sum(status == "TP", na.rm = TRUE),
            FP = sum(status == "FP", na.rm = TRUE),
            .groups = "drop")

fn <- crowns |>
  filter(source == "reference") |>
  group_by(landuse, grain_m) |>
  summarise(FN = sum(status == "FN", na.rm = TRUE),
            .groups = "drop")

metrics <- full_join(tp_fp, fn, by = c("landuse", "grain_m")) |>
  mutate(
    across(c(TP, FP, FN), ~ replace_na(.x, 0L)),
    Precision  = ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_),
    Recall     = ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_),
    `F1-score` = ifelse((Precision + Recall) > 0,
                        2 * Precision * Recall / (Precision + Recall),
                        NA_real_)
  ) |>
  # geom_line() joins points in row order, so the table is sorted by grain.
  
  arrange(landuse, grain_m)

cat("\nDetection metrics:\n")
print(as.data.frame(metrics), digits = 3)


# ---- Sample sizes: reference crowns per land-use type ------------------------

sample_sizes <- crowns |>
  filter(source == "reference") |>
  group_by(landuse) |>
  summarise(n = n_distinct(polygon_id), .groups = "drop")

landuse_labels <- setNames(
  sprintf("%s (n = %d)", sample_sizes$landuse, sample_sizes$n),
  as.character(sample_sizes$landuse)
)


# ---- Long format -------------------------------------------------------------

metrics_long <- metrics |>
  pivot_longer(cols      = c(Precision, Recall, `F1-score`),
               names_to  = "metric",
               values_to = "value") |>
  mutate(metric = factor(metric,
                         levels = c("F1-score", "Precision", "Recall")))


# ---- Plot --------------------------------------------------------------------

landscape_complexity <- ggplot(
  metrics_long,
  aes(x = grain_m, y = value,
      colour = landuse, group = landuse, shape = landuse)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ metric, nrow = 1) +
  scale_x_log10(breaks = c(0.3, 0.9, 2.7, 8.1)) +
  scale_colour_manual(values = landuse_colours, labels = landuse_labels) +
  scale_shape_manual(values = landuse_shapes, labels = landuse_labels) +
  scale_y_continuous(limits = c(0, 1),
                     breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(x      = "Spatial grain (m)",
       y      = "Detection score",
       colour = "Land-use type",
       shape  = "Land-use type") +
  guides(colour = guide_legend(nrow = 1, byrow = TRUE),
         shape  = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_minimal(base_size = 12) +
  theme_fancy() +
  theme(
    axis.title            = element_text(size = 10, face = "bold"),
    axis.text             = element_text(size = 10, colour = "black"),
    strip.text            = element_text(size = 10, face = "bold"),
    legend.title          = element_text(size = 10),
    legend.text           = element_text(size = 10),
    legend.position       = "bottom",
    legend.direction      = "horizontal",
    legend.box            = "horizontal",
    legend.title.position = "top",
    legend.key.width      = unit(1.0, "cm"),
    legend.key.height     = unit(0.4, "cm"),
    legend.margin         = margin(0, 0, 0, 0),
    legend.box.spacing    = unit(0.2, "cm"),
    panel.grid.minor      = element_blank(),
    panel.spacing.x       = unit(1, "lines"),
    plot.margin           = margin(10, 15, 10, 10)
  )

print(landscape_complexity)


# ---- Save --------------------------------------------------------------------

dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(
  file.path(plot_dir, "landuse_type_perfomance_plot.png"),
  plot   = landscape_complexity,
  width  = 21.0,
  height = 14.85,
  units  = "cm",
  dpi    = 600,
  bg     = "white"
)