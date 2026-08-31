# Towards scalable monitoring of African baobab populations using satellite imagery and a U-Net model.


This repository contains the data and code used to reproduce the analyses, figures and tables presented in:

> Chafika Phiri, Hugh A. Graham, Sarah Venter, Dawn Cory-Toussaint, Dave Hodgson, and Andrew M. Cunliffe. (Year). *Towards scalable monitoring of African baobab populations using satellite imagery and a U-Net model*. Journal. DOI: https://doi.org/...

A permanent version of this repository is archived at [![DOI](https://zenodo.org/badge/DOI/...svg)](...)

Use of this code is licensed under [![Licence: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](...)

Contact Chafika Phiri at EMAIL: cp844@exeter.ac.uk


---

## Project overview

The main aim of this study was to test and evaluate deep learning (U-Net) and satellite imagery to map baobab tree distribution across heterogeneous landscapes.

We trained a U-Net deep learning model and evaluated its performance across two contrasting sites in South Africa: Kremetart Game Farm in the Soutpansberg and Mapungubwe National Park in Limpopo.

We used fine-resolution WorldView-3 satellite imagery and 613 field-validated GPS locations of baobab trees to assess detection performance across multiple spatial resolutions 0.3 m, 0.9 m, 2.7 m, and 8.1 m.

---

## Repository contents

This repository contains the following folders:

### Scripts

| Script | Description |
|---------|-------------|
| *01a_image_mosaicking_coregistration_masking.qmd* | Mosaics, co-registration, and cloud/shadow masking.|
| *01b_gdal_pansharpening.txt* | Command lines used in gdal for pansharpening the imagery.|
| *02_image_rescalling_and_resampling.R* | Stacks, rescales imagery to 8-bit, and resamples to four spatial grains (0.3, 0.9, 2.7, 8.1 m).|
| *03_image_and_masks_patch extraction.R* | Extracts 256×256 image and mask patches for U-Net training.|
| *04a_U_Net_model_training_kremetart_feb.R* | Trains the U-Net on Kremetart February (wet-season) imagery.|
| *04b_U_Net_model_training_july.R* | Trains the U-Net on July (dry-season) imagery.|
| *04c_U_Net_model_training_kremetart_feb_july.R* | Trains the U-Net on the combined Kremetart February+July (16-bands).|
| *05_model_prediction.R* | Runs a trained model over a full landscape images and produces predictions.|
| *06_performance_evaluation_on_each_site.R* | Matches predicted and reference crowns and calculates precision, recall and F1 for each site.|
| *07_results_plots_and_visualisation.R* | Produces manuscript figures and supplementary outputs.|


### Data

| File | Description |
|------|-------------|
| *Kremetart_Tree_Polygons.gpkg* | Training polygons from Kremetart.|
| *Mapungubwe_Tree_Polygons.gpkg* | Training polygons from Mapungubwe.|
| *Kremetart_negatives_examples.gpkg* | Negative examples from Kremetart.|
| *Mapungubwe_Non_baobab_points.gpkg* | Negative examples from Mapungubwe.|


---

## Getting started

Clone this repository and review the project overview and repository structure before running the analysis. 

TESS Lab projects typically use [renv](https://rstudio.github.io/renv/) to record package dependencies and software versions. 
Where an renv.lock file is included, restore the project environment before running any analyses:
```r
renv::restore()
```

## Running the analysis

The scripts should be run in this order:

1. *01a_image_mosaicking_coregistration_masking.qmd*
2. *01b_gdal_pansharpening.txt*
3. *03_image_and_masks_patch extraction.R*
4. *04a_U_Net_model_training_kremetart_feb.R*
5. *04b_U_Net_model_training_kremetart_july.R*
6. *04c_U_Net_model_training_kremetart_feb_july.R*
6. *05_model_prediction.R*
5. *06_performance_evaluation_on_each_site.R*
6. *07_results_plots_and_visualisation.R*

---


## Acknowledgements

We gratefully acknowledge financial support from Few and Far Collection within the Oppenheimer Programme in African Landscape Systems (OPALS) (https://opals-exeter.org/), which enabled this research. We sincerely thank landowners at Kremetart, Wild Estate WE (PTY) Ltd, ZZ2, Francois Pauer, and South African National Parks (SANParks) for granting access and permission to conduct fieldwork. We thank Alan Nare and Tapiwa Gumbo for their support during data collection. We also thank Oscar Hountondji, Remy Vandaele, and Scott Barningham for their valuable insights during the development and implementation of the of U-Net modelling workflow.

