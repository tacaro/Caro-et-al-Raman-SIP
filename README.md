# Supporting Information

This repository contains all source code needed to reproduce the calculations and plots of the following manuscript: *Caro et al. (in review)*. (Citation to be updated upon publication).

DOI: (To be updated upon publication).


# What can I do with this code?

In publishing this repository, our hope is that this code is useful to other members of the scientific community. This repository is released under a Creative Commons BY (CC-BY) license, which means that all code published here can be shared and adapted for any purposes so long as appropriate credit and citation of the original paper is given. See attribution section for details.

# How do I run this code?

1. Download and install R for your operating system.
2. Download and install RStudio for your operating system.
3. Download a zip file of this repository and decompress it in a directory of your choosing on your computer.
4. Navigate to the directory and open the `.Rproj` file to start Rstudio and load this project's files.
5. Open the script(s) you would like to run. Scripts are numbered in the order they should be executed e.g, 01, 02, 03. Duplicate numbers mean those scripts can be run in any order relative to each other.
6. Ensure that you have all of the required libraries installed by inspecting the `Setup` chunks. In these scripts, we note the CRAN/GitHub version/release that was used. If any libraries fail to install, note the name of the library and attempt to manually install its most recent version via CRAN or GitHub.
7. To generate an HTML report, select File --> Knit from the menu.


# Scripts

## Directory

- `01_nanoSIMS_data_reduction.Rmd`: Load NanoSIMS ROI data, load cell metadata, join cell_metadata to nanoSIMS ROI data, save to cache.
- `01_Raman_Data_Reduction.Rmd`: Load Raman Data, assign metadata, join the tables, Summarize, save to cache.
- `02_CD2F_Co-Registration.Rmd`: Read the two `01` dataframes from cache, join the dataframes together, save to cache.
- `03_CD2F_Analysis`: Load the `02` data from cache, visualize, cache.
- `03_CD2F_Sensitivity.Rmd`: Growth modeling and data vizualization.
- `03_CD2F_cell_abund_cd.qmd` tests with cell abundance data.
- `04_molecules_H.qmd`: H mass balancing in model cells.
- `05_SERS_data_reduction.qmd`: SERS data reduction and plotting.
- `98_SIMS_viz.ipynb`: Jupyter notebook for visualizing SIMS data.
- `99_CD2F_DataViz.Rmd`: Plots
- `99_Modeling_Growth.Rmd`: Tests with modeling of growth
- `nanosims_processor.py` : Script for extracting .im files and tabulating ROI information
