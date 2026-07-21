# Phiri_et_al_baobab_mapping_with_unet

## About this repository

This repository contains R code used in our manuscript:

“[Title of our Manuscript]”
Chafika Phiri, Hugh A. Graham, Sarah Venter, Dawn Cory-Toussaint, Dave Hodgson, and Andrew M. Cunliffe

This study explores the use of deep learning (U-Net) and satellite imagery to map baobab tree distribution across heterogeneous landscapes.

## Study overview

We trained a U-Net deep learning model and evaluated its performance across two contrasting sites in South Africa: Kremetart Game Farm in the Soutpansberg and Mapungubwe National Park in Limpopo.

We used fine-resolution WorldView-3 satellite imagery and 755 field-validated GPS locations of baobab trees to assess detection performance across multiple spatial resolutions 0.3 m, 0.9 m, 2.7 m, and 8.1 m.

## Repository structure

This repository contains the following folders:

* `R Scripts`: Contains the R scripts.
            1. WorldView_3_mosaicking_and_Cloud_and_Shadow_Masking.qmd 
            2. WorldView3_Image_Coregistration.R
            3. Image_rescaling_and_resampling.R
            4. Image_Mask_Patch_Extraction.R
            5. U_Net_Model_Archictecture_and_Training.R
            6. Performance_evaluation_visualisation.qmd
* `Manuscripts Plots`: Contains the plots used in the manuscript.

## Data

Input datasets including image chips, and training labels are available publicly at Zenodo: [Zenodo link]

## Citation

If you use this code, please cite as:

Phiri, C., Graham, H. A., Venter, S., Cory-Toussaint, D., Hodgson, D., & Cunliffe, A. M. (Year).
[Title of Manuscript]. 
DOI: [Insert DOI]

## Contact

For questions, collaboration, or data access requests:

Chafika Phiri
Email: cp844@exeter.ac.uk

## Acknowledgements

We gratefully acknowledge financial support from Few and Far Collection within the Oppenheimer Programme in African Landscape Systems (OPALS) (https://opals-exeter.org/), which enabled this research. We sincerely thank landowners at Kremetart, Wild Estate WE (PTY) Ltd, ZZ2, Francois Pauer, and South African National Parks (SANParks) for granting access and permission to conduct fieldwork. We thank Alan Nare and Tapiwa Gumbo for their support during data collection. We also thank Oscar Hountondji, Remy Vandaele, and Scott Barningham for their valuable insights during the development and implementation of the of U-Net modelling workflow.

