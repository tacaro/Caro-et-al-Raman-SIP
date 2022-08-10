# README

## Conda installation

Install miniconda and mamba if not already:
 * Miniconda: https://docs.conda.io/en/latest/miniconda.html [follow install directions]
 * https://mamba.readthedocs.io/en/latest/installation.html#installation
 
Install nanosims environment:
 * conda env create -f nanosims_env.yml
 * conda activate nanosims-processing

Shouldn't need to, but if something crashes manually install sims via pip
   pip install sims --no-deps

Fix a bug: 
 1. Run the script to make it generate an error, e.g. run nanosims_processor_lite.py --help
 2. find the next-to-last line that says "... from skimage.feature import register_translation"
 3. open the sims/utils.py file from that line with nano or vi, e.g. for me it looks like
    nano /Users/dutter/opt/miniconda3/envs/nanosims-processing/lib/python3.7/site-packages/sims/utils.py
 4. Use the arrow keys to move to anad edit the line near the top that says "from skimage.feature import register_translation" to say "from skimage.registration import phase_cross_correlation"
 5. Edit the line several pages down that starts with "sh = register_translation(" to be "sh = phase_cross_correlation("
 6. Save and exit.
 
You're good to go!

## Data processing

1. Extract single-channel images from .im files. You can do this with `loop_channel_xtractor.sh`. You can edit the folder name in this shell script to match the folder you'd like to batch process. Or you can do it manually.

2. Draw ROIs: use an image editing software (recommend: GIMP, Adobe Photoshop, etc.) and use pure RGB colors (Red, Green, or Blue) to draw ROIs. Draw conservatively, don't get near the edge of the cell where background bleed is a concern. Save ROIs as transparent images.

3. Use ROI.png files to generate single-cell isotope ratios. You can do this with the `loop_ROI_processor.sh` script. Same as before, edit the folder name in this shell script to match the folder you'd like to batch process. Or you can do it manually.

4. Use `collect_all_tsv.sh` to collect all .tsv output files in the nanoSIMS_roi and all subdirectories. This script copies them to tsv_output folder.

## ROI drawing rules

1. Pure RGB red (255, 0, 0) for cells that can clearly be distinguished.

2. Pure RGB green (0, 255, 0) for cells that cannot be distinguished "clumps" etc.

3. Pure RGB blue is not used.

4. Cells with scanline defects are excluded.

5. Avoid drawing the "shadow"/"halo" around cells: be conservative and only draw over the center of the cell.
