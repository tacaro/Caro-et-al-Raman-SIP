#!/usr/bin/env python3

import argparse
import re
import pandas as pd
import numpy as np
import xarray as xr
import sims
from imageio import imread
from scipy import ndimage
from skimage import measure, transform, morphology, segmentation, filters, color, feature
#from aicsimageio import AICSImage
from matplotlib import colors
from matplotlib.widgets import CheckButtons, Button
import sys
if sys.platform == 'darwin':
    try:
        get_ipython().__class__.__name__ != 'ZMQInteractiveShell'
    except NameError:
        #TK_SILENCE_DEPRECATION=1
        import matplotlib
        matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from pathlib import Path
from yaml import dump, safe_load

parser = argparse.ArgumentParser(description='Tool for parsing Cameca NanoSIMS .im files to produce ratios. \
                                 Basic usage is to read a .im file, and produce single-channel PNGs and/or ratios of \
                                 isotope_1 / (isotope_1 + isotope_2). Also can read in a .png as a mask for ROIs.')
group0 = parser.add_argument_group('INPUT / OUTPUT', '')
group1 = parser.add_argument_group('BASIC CONTROL', '')
group2 = parser.add_argument_group('PLOTTING OPTIONS', '')
group3 = parser.add_argument_group('ROI OPTIONS', '')

group0.add_argument('-i', '--input', metavar='PATH',  # required=True,
                    default='',
                    dest='input', help='File for input (a .im file)')
group0.add_argument('-o', '--output', metavar='PATH',
                    default=None, help='Prefix for output [default: --input without the .im ending]')
group0.add_argument('-y', '--config', default=None, metavar='YAML FILE',
                    help='Path to a config file (yaml format). Will override the --frames and -i argument if specified. If doing the two-step workflow, this will be produced by the first step for use in the second step.')
group0.add_argument('--batch', default=False, action='store_true',
                    help='Batch mode (skip interactive mode, just run set flags)')

group1.add_argument('-c', '--compare1', metavar='"STRING"', default='15N 12C',
                    help='Focal element or trolley to be the numerator (use quotes if has spaces) [default: "15N 12C"]')
group1.add_argument('-C', '--compare2', metavar='"STRING"', default='14N 12C',
                    help='Comparand element or trolley to be summed with --compare1 for denominator \
                    [default: "14N 12C"]')
group1.add_argument('-f', '--frame', metavar='INT', default=None,
                    help='Frame index (e.g., "0" would be the first frame) to focus, \
                    default behavior is to loop through all. Can comma separate multiple values. [default: None]')
group1.add_argument('--add', default=False, action='store_true',
                    help='Add all frame(s) matching by -f argument. [default: do not add]')
group1.add_argument('-F', '--filter_method', metavar='STRING', default=None,
                    help='Add a filter step, currently confined to one of "gaussian", "median", "tophat" [default: None]')
group1.add_argument('-s', '--sigma', metavar='FLOAT', type=float, default=1.0,
                    help='Parameter for filtration (sigma for gaussian, size for median) [default: 1.0]')

group2.add_argument('--rough', action='store_true',
                    help='Only rough plot, then exit [default: False]')
group2.add_argument('-w', '--whole_image', action='store_true',
                    help='Calculate ratio of specified channels for the full image (all pixels)')
group2.add_argument('-n', '--no_show', action='store_true',
                    help='Add this to not show plots to your screen, only save. Useful when batch processing')
group2.add_argument('--minquant', type=float, default=0.00,
                    help='The minimum percentile to scale the colorbar to, e.g., 0.1 would blackout the lowest 10% of pixels')
group2.add_argument('--maxquant', type=float, default=1.00,
                    help='The maximum percentile to scale the colorbar to, e.g., 0.9 would saturate the top 10% of pixels')
group2.add_argument('--minvalue', type=float, default=None,
                    help='Specify lower limit for whole-image-ratio colormap display')
group2.add_argument('--maxvalue', type=float, default=None,
                    help='Specify upper limit for whole-image-colormap colormap display')


group3.add_argument('-r', '--roi', metavar='PATH', default=None,
                    help='Path to corresponding manually drawn ROI file (mask) [default: None]')
group3.add_argument('--ipad', action='store_true',
                    help='Legacy mode (treat blue as non-ROI category for cropping)')
group3.add_argument('--count_stat', metavar='STRING', default='mean',
                    help='Method for aggregating pixel values in each ROI to report in the summary table. \
                    Does not affect ratios. Current options: mean, sum [default: mean]')
group3.add_argument('-a', '--autoroi', default=False, action='store_true',
                    help='Auto-generate an ROI by attempting to segment the image with Otsu + watershed method. \
                    Only saves data in the absence of the --roi flag (to not overwrite user-specified ROIs).')
group3.add_argument('-M', '--min_area', default=100, metavar='INT',
                    help='Minimum area for --autoroi to recognize a region (to cut down on tiny bright specks/debris)')
group3.add_argument('--otsu_mult_high', default=1.2, metavar='FLOAT', type=float,
                    help='The --autoroi uses Otsu thresholding to determing "object" and "background" intensities, so \
                         this value is multiplied by the Otsu threshold to determine the UPPER bound')
group3.add_argument('--otsu_mult_low', default=0.5, metavar='FLOAT', type=float,
                    help='The --autoroi uses Otsu thresholding to determing "object" and "background" intensities, so \
                         this value is multiplied by the Otsu threshold to determine the LOWER bound')
group3.add_argument('--shuffle_roi', default=0, action='store_true',
                    help='pick random spots matching ROIs as a test')
group3.add_argument('--czi', default=None, metavar='FILE.czi', help='CZI tilescan for ROI')

args = parser.parse_args()


def load_image(args=args, config=None, path='', fr=None, add=False):
    if config is not None:
        with open(config, 'r') as f:
            c = safe_load(f)
        args = vars(args)
        args.update(c)
        args = argparse.Namespace(**args)
        path = args.input
        fr = args.frame
        add = args.add

    image = sims.SIMS(path)
    aligned_image, shifts = sims.utils.align(image)  # account for inter-frame shifts

    # remove 1px border on all sides because Cameca
    aligned_image = aligned_image.drop_isel(x=[0, -1], y=[0, -1])

    if fr is not None:
        print('subsetting to frame(s): ' + str(fr))
        frs = [int(x) for x in str(fr).split(',')]
        aligned_image = aligned_image.sel(frame=frs)
    if add:
        print('Adding frame(s) [all surviving -f]')
        aligned_image = accumulate(aligned_image)

    return aligned_image

def apply_filter(img, filt_method, sigma):
    img2 = img
    if filt_method == 'gaussian':
        for f in img.frame.values:
            for specie in img.species.values:
                img2.loc[specie, f, :, :] = ndimage.gaussian_filter(img.loc[specie, f].data, sigma=sigma)
        return img
    if filt_method == 'median':
        for f in img.frame.values:
            for specie in img.species.values:
                img.loc[specie, f, :, :] = ndimage.median_filter(img.loc[specie, f].data, size=int(sigma))
    if filt_method == 'tophat':
        for f in img.frame.values:
            for specie in img.species.values:
                footprint = morphology.disk(1)
                res = morphology.white_tophat(img.loc[specie, f].data, footprint)
                img.loc[specie, f, :, :] = img.loc[specie, f].data - res
    return img


def accumulate(img):
    img = img.sum('frame')
    img = img.assign_coords({'frame': 'a'})
    img = img.expand_dims('frame', 1)

    return img


def add_pseudoframe(img, name='cells', s1='12C', s2='14N 12C'):
    new_frames = []
    for f in img.frame.values:
        new_frame = img.loc[s1, f] * img.loc[s2, f]
        new_frame = new_frame.assign_coords({'species': s1 + 'x' + s2, 'frame': f})
        new_frame = new_frame.expand_dims(dim=['species','frame'], axis=[0,1])
        new_frames.append(new_frame)

    new_frames = xr.concat(new_frames, dim='frame')
    img = xr.concat([img, new_frames], dim='species')

    return img


def strip_multi(img, collapse_img, frames=True, min_quant=0, max_quant=1):
    plt.rcParams['figure.figsize'] = [1.3*len(img.species.values), 1.5*(len(img.frame.values) + frames)]
    fig, ax = plt.subplots(len(img.frame.values) + 1*frames, len(img.species.values))

    for j,specie in enumerate(img.species.values):
        for i,f in enumerate(img.frame.values):
            sub_img = img.loc[specie, f]
            norm = colors.Normalize(np.quantile(sub_img, min_quant), np.quantile(sub_img, max_quant), clip=False)
            ax[i,j].imshow(sub_img, cmap='gray', interpolation='none', norm=norm)
            ax[i,j].set(ylabel='Frame ' + str(f), xlabel=specie)
            ax[i,j].label_outer()
            ax[i,j].xaxis.set_ticks_position('none')
            ax[i,j].yaxis.set_ticks_position('none')
            ax[i,0].yaxis.set_ticks_position('left')

        sub_img = collapse_img.loc[specie,'a']
        norm = colors.Normalize(np.quantile(sub_img, min_quant), np.quantile(sub_img, max_quant), clip=False)
        ax[i+1,j].imshow(sub_img, cmap='gray', interpolation='none', norm=norm)
        ax[i+1,j].set(ylabel='Accumulated', xlabel=specie)
        ax[i+1,j].yaxis.set_ticks_position('none')
        ax[i+1,j].label_outer()
        ax[i+1,0].yaxis.set_ticks_position('none')

    return fig, ax


def strip_single(collapse_img, cmap='gray', min_quant=0, max_quant=1):
    plt.rcParams['figure.figsize'] = [1.4*len(collapse_img.species.values), 6]
    fig, axs = plt.subplots(2, int(np.ceil(len(collapse_img.species.values)/2)))
    for s,ax in enumerate(axs.reshape(-1)):
        if s > len(collapse_img.species.values):
            next
        sub_img = collapse_img.loc[collapse_img.species.values[s-1],'a']
        norm = colors.Normalize(np.quantile(sub_img, min_quant), np.quantile(sub_img, max_quant), clip=True)
        ax.imshow(sub_img, cmap=cmap, interpolation='none', norm=norm)
        ax.set(ylabel='Accumulated', xlabel=None, title=collapse_img.species.values[s-1])
        ax.yaxis.set_ticks_position('none')
        ax.label_outer()
    axs[0,0].yaxis.set_ticks_position('left')
    axs[1,0].yaxis.set_ticks_position('left')

    return fig, axs

def rough_plot(img, frames=True, cmap='gray', save=True, output='', no_show=False, min_quant=0, max_quant=1):
    img = add_pseudoframe(img, 'cells', '12C', '14N 12C')
    collapse_img=accumulate(img)

    if len(img.frame.values) > 1:
        fig, ax = strip_multi(img, collapse_img, min_quant=min_quant, max_quant=max_quant)
    else:
        fig, ax = strip_single(collapse_img, cmap=cmap, min_quant=min_quant, max_quant=max_quant)

    plt.tight_layout()
    if not no_show:
        plt.show()
    if save:
        fig.savefig(output + "_tiled.png", dpi=300)


def save_plot(img, frames=True, output='', no_show=False, min_quant=0, max_quant=1):
    rough_plot(img, frames=frames, save=True, no_show=True, output=output, min_quant=min_quant, max_quant=max_quant)
    plt.cla()
    img = add_pseudoframe(img, 'cells', '12C', '14N 12C')
    for f in img.frame.values:
        for specie in img.species.values:
            sub_img = img.loc[specie, f]
            norm = colors.Normalize(np.quantile(sub_img, min_quant), np.quantile(sub_img, max_quant), clip=False)
            plt.cla()
            plt.gray()
            plt.axis('off')
            plt.title(specie)
            plt.imsave(fname=output + "_f" + str(f) + "_" + specie + ".png",
                       arr=sub_img, cmap='gray')
    plt.cla()
    try:
        accum=accumulate(img)
        for specie in accum.species.values:
            sub_img = accum.loc[specie,'a']
            norm = colors.Normalize(np.quantile(sub_img, min_quant), np.quantile(sub_img, max_quant), clip=False)
            plt.cla()
            plt.gray()
            plt.axis('off')
            plt.title(specie)
            plt.imsave(fname=output + "_acc_" + specie + ".png",
                       arr=sub_img, cmap='gray')
    except KeyError:
        pass
    print('Saved!')


def gradient_sharpness(img_2d):
    gy, gx = np.gradient(100* img_2d / np.max(img_2d))
    overall_gradient = np.sqrt(gx**2 + gy**2)

    return np.average(overall_gradient)
def evaluate_frame_sharpness(img, channels=['SE','12C','14N 12C']):
    frame_scores = []
    for f in img.frame.values:
        pass


def jupyter_plot(img, inp, output, min_quant=0, max_quant=1):
    rough_plot(img=img, frames=True, save=False, min_quant=min_quant, max_quant=max_quant)

    labels = [str(f) for f in img.frame.values]
    visibility = [True for f in img.frame.values]

    plt.tight_layout()

    checkboxes = [widgets.Checkbox(value=True, description=x) for x in labels]
    check_accum = widgets.Button(description="Check\nAccumulation", button_style='success')
    save_and_done = widgets.Button(description="Done", button_style='warning')

    #plt.show()

    control_box = widgets.HBox([widgets.VBox([check_accum, save_and_done]),
                               widgets.HBox(children=checkboxes)])

    display(control_box)

    out=widgets.Output(layout={'border': '1px solid black'}, wait=True)
    display(out)

    def draw_proposed(x):
        visibility = [int(c.description) for c in checkboxes if c.value]
        proposed = accumulate(img.loc[:,visibility])
        out.clear_output()
        with out:
            display(rough_plot(proposed, frames=False, cmap='cividis', save=False, output=output, min_quant=min_quant, max_quant=max_quant))

    def save_data(x):
        visibility = [int(c.description) for c in checkboxes if c.value]
        accumulated = accumulate(img.loc[:,visibility])
        Path(re.sub(r'^(.*/).*$', '\\1', output)).mkdir(exist_ok=True)
        with out:
            save_plot(accumulated, frames=False, output=output, min_quant=min_quant, max_quant=max_quant)
        with open(output + "-config.yaml", 'w') as f:
            dump({'input': inp, 'add': True, 'frame': ','.join(str(x) for x in img.frame.values[visibility])}, f)

    check_accum.on_click(draw_proposed)
    save_and_done.on_click(save_data)


def interactive_plot(img, inp, output, min_quant=0, max_quant=1):
    try:  # jupyter format interaction
        if get_ipython().__class__.__name__ != 'ZMQInteractiveShell':
            print('Please use in either jupyter notebook or regular python')
            return
        if isinstance(img, list):
            if not isinstance(inp, list) or not isinstance(output, list):
                print("Your img argument is a list (presumably of several .im files)\nbut your input/output args are not lists,\neverything needs to be a list or not")
                return
            img_list = img
            img = img_list[0]
            def f(x):
                jupyter_plot(img=img_list[x], inp=inp[x], output=output[x], min_quant=min_quant, max_quant=max_quant)
            img_slide = widgets.interactive(f, x=widgets.IntSlider(min=0, max=len(img_list)-1, step=1, value=0, continuous_update=False));
            display(img_slide)
        else:
            jupyter_plot(img=img, inp=inp, output=output, min_quant=min_quant, max_quant=max_quant)

    except NameError:  # matplotlib in terminal
        collapse_img=accumulate(img)

        plt.rcParams['figure.figsize'] = [1.3*len(img.species.values), 2.2*len(img.frame.values)]
        fig, ax = plt.subplots(len(img.frame.values) + 1, len(img.species.values))

        for j,specie in enumerate(img.species.values):
            for i,f in enumerate(img.frame.values):
                sub_img = img.loc[specie, f]
                norm = colors.Normalize(np.quantile(sub_img, min_quant), np.quantile(sub_img, max_quant), clip=False)
                ax[i,j].imshow(sub_img, cmap='gray', interpolation='none', norm=norm)
                ax[i,j].set(ylabel='Frame ' + str(f), xlabel=specie)
                ax[i,j].label_outer()

            sub_img = collapse_img.loc[specie,'a']
            norm = colors.Normalize(np.quantile(sub_img, min_quant), np.quantile(sub_img, max_quant), clip=False)
            ax[i+1, j].imshow(sub_img, cmap='gray', interpolation='none', norm=norm)
            ax[i+1,j].set(ylabel='Accumulated', xlabel=specie)
            ax[i+1,j].label_outer()

        labels = [str(f) for f in img.frame.values]
        visibility = [True for f in img.frame.values]

        plt.tight_layout()

        plt.subplots_adjust(right=0.94)

        rax = plt.axes([0.95, 0.47, 0.04, 0.1])
        check = CheckButtons(rax, labels, visibility)
        bax = plt.axes([0.945, 0.40, 0.05, 0.05])
        go_button = Button(bax, 'Check')#

        def func(label):
            p = check.get_status()
            proposed = accumulate(img.loc[:,p])
            rough_plot(proposed, frames=False, cmap='cividis', save=False, output=output, min_quant=min_quant, max_quant=max_quant)

        go_button.on_clicked(func)
        plt.show()
        pattern = check.get_status()

        if all(not x for x in pattern):
            sys.exit("You unchecked every plane, exiting under the assumption you don't want to process this experiment")

        accumulated = accumulate(img.loc[:,pattern])

        with open(output + "-config.yaml", 'w') as f:
            dump({'input': inp, 'add': True,
            'frame': ','.join(str(x) for x in img.frame.values[pattern])}, f, encoding='utf-16-be')

        save_plot(accumulated, frames=False, output=output, min_quant=min_quant, max_quant=max_quant)



def get_image_from_raw_rois(roi, im, ipad=False):
    if roi.shape[0:2] == im.data.shape[2:4]:
        print("The ROI png has the same pixel dimension as the image, great! [no ROI cropping]")
    elif roi.shape[0] > im.data.shape[2]:
        print("ROI png dimensions are larger than the image, trying to correct...")

        edges = np.concatenate([roi[[0, -1], 0, :], roi[[0, -1], 0, :]]).reshape((4, 4))
        if np.all(edges == edges[0, :]):  # check if edges are all the same RGB (likely border)
            print("Edge pixels are all the same color, assuming this is background so trimming these solid edges")

            if ipad:
                print("Using iPad ROI mode (long story), removing all blue-ish pixels - next time, don't add blue border")
                not_outside = np.any(roi != edges[0,:], axis=-1)# * 1
                blue = roi[:, :, 2]-roi[:, :, 1]
                not_blue = np.invert(blue > 100)
                not_outside = not_outside * not_blue
                roi = roi[np.array(not_outside, dtype='bool'), :]
                #print(np.argwhere(np.sum(not_outside,0) > 1))
            else:
                not_outside_0 = np.all(roi != edges[0, :], axis=0)[:,0]
                not_outside_1 = np.all(roi == edges[0, :], axis=1)[:,0]
                roi = roi[~not_outside_0, :, :]
                roi = roi[:, ~not_outside_1, :]

            width_roi = roi.shape[0]
            width_image = im.data.shape[3]

            roi = roi.reshape((width_roi, width_roi, 4))

        # check if this fixed it
        if roi.shape[0:2] == im.data.shape[2:4]:
            print("Cropped successfully, no rescaling!")
        else:
            print("FYI... ROI was still too big @ "+str(width_roi)+"px vs. "+str(width_image)+"px so scaling")
            roi = transform.resize(roi, (width_image, width_image, 3), anti_aliasing=True)*255
    else:
        sys.exit("Your ROI is smaller? Stopping so you can fix. Image: " + im.data.shape + " vs roi: " + roi.shape)

    return roi


def parse_ROIs(objects, grp_col, c1, c2, annotated_im, im, stats, count_stat):
    #for obj in range(objects[1]):
    i=1
    for reg in measure.regionprops(measure.label(objects)):
        #obj_x, obj_y = np.where(objects[0] == (obj + 1))

        #pts = np.where(objects[0] == (obj + 1))
        #pts = np.reshape(pts, (2, len(pts[0])))
        #pts = [pts[:, x] for x in range(pts.shape[1])]
        pts = reg.coords
        #print(pts)
        counts1 = sum([c1.data[x, y] for x, y in pts])
        counts2 = sum([c2.data[x, y] for x, y in pts])

        if count_stat == 'mean':
            all_counts = [v for v in np.array([im.data[:, x, y] for x, y in pts]).sum(axis=0)]
        elif count_stat == 'sum':
            all_counts = [v for v in np.array([im.data[:, x, y] for x, y in pts]).mean(axis=0)]
        else:
            sys.exit("Somehow there is no stat? fix the code so args.count_stat is not: " + count_stat)

        rat_im = counts1/(counts1 + counts2)
        #print(rat_im)
        #annotated_im[obj_x, obj_y] = rat_im
        #xs = np.array(x for x,y in pts)
        x_arr = np.array([x for x,y in pts])
        y_arr = np.array([y for x,y in pts])

        annotated_im[x_arr, y_arr] = rat_im

        #obj_stats = [grp_col, obj]
        obj_stats = [grp_col, i]
        i+=1
        obj_stats.extend(all_counts)
        obj_stats.append(rat_im)
        obj_stats.append(reg.area)
        obj_stats.append(reg.axis_major_length)
        obj_stats.append(reg.axis_minor_length)

        stats.append(obj_stats)
    return annotated_im, stats


def summary_ratio_img_and_table(annotated_image, stats_table, species, frame='a', compare1='15N 12C', compare2='14N 12C', output='', no_show=True):
        stats_columns = ["Group", "ROI"]
        stats_columns.extend(species)
        stats_columns.append("Ratio_" + compare1 + "x" + compare2)
        stats_columns.append('area')
        stats_columns.append('axis_major_length')
        stats_columns.append('axis_minor_length')
        stats_columns = [re.sub(" ", "_", x) for x in stats_columns]
        stats_table = pd.DataFrame(stats_table, columns=stats_columns)

        stats_table.to_csv(output + "_f" + str(frame) +
                           "_ratio" + re.sub(" ", "_", compare1) + "-x-" + re.sub(" ", "_", compare2) +
                           ".tsv", sep="\t", index=False)
        plt.clf()
        plt.imshow(annotated_image, interpolation='none', cmap='cividis')
        plt.axis('off')
        plt.colorbar().set_label("Fraction (per ROI)")
        plt.title(sims.utils.format_species(compare1) + " / (" + sims.utils.format_species(compare1)+" + " +
                  sims.utils.format_species(compare2) + ")")
        plt.savefig(fname=output + "_f" + str(frame) +
                          "_ratio" + re.sub(" ", "_", compare1) + "-x-" + re.sub(" ", "_", compare2) + ".png")
        if not no_show:
            plt.show(block=True)


def autosegment_threshold(img, thresh, low_mult, high_mult, too_small, method='otsu', pre_gaus=0, clip_min=0, clip_max=1,
                          no_show=False, output=''):
    img = np.clip(img, clip_min, clip_max)

    if pre_gaus > 0:
        img = filters.gaussian(img, sigma=pre_gaus)

    if method=='otsu':
        markers = np.zeros_like(img)
        markers[img < low_mult*thresh] = 1
        markers[img > (high_mult*thresh)] = 2
        label_image = segmentation.watershed(filters.sobel(img), markers) - 1
    elif method=='canny':
        label_image = feature.canny(img, sigma=low_mult)
        label_image = morphology.binary_closing(label_image, (np.ones(round(high_mult)),np.ones(round(high_mult))))
        label_image = ndimage.binary_fill_holes(label_image)

    # drop segments with small area
    label_image = morphology.area_opening(label_image, area_threshold=too_small)

    auto_rois = measure.label(label_image, return_num=True)

    if not no_show:
        plt.clf()
        plt.imshow(img, cmap='Greys_r', interpolation=None)
        plt.imshow(color.label2rgb(label_image, bg_label=0), interpolation=None, alpha=0.3)
        for reg in measure.regionprops(auto_rois[0]):
            ry, rx = reg.centroid
            plt.text(rx + 0, ry - 0, s=reg.label, ha='center', va='center', c='white')#, bbox=dict(fc='white', ec='none', pad=2))
        if output not in [None, '']:
            plt.savefig(fname=output + "_auto_roi_watershed.png")
        plt.show(block=True)

    return label_image, auto_rois


def autosegment(full_img, comp1='15N 12C', comp2='14N 12C', annotated_im=None, low_mult=0.8, high_mult=1.2, too_small=100, no_show = False, output='', count_stat = 'mean', seg_channel='12Cx14N 12C', method='otsu'):
    full_img = add_pseudoframe(full_img)
    img = np.array(full_img.loc[seg_channel, full_img.frame.values[0]])

    img = img/img.max()
    #img = 255*(img/img.max())

    thresh = filters.threshold_otsu(img)

    try:  # jupyter format interaction
        if get_ipython().__class__.__name__ != 'ZMQInteractiveShell':
            print('Please use in either jupyter notebook or regular python')
            return

        min_slide = widgets.FloatSlider(min=0, max=2, step=0.1, value=low_mult, continuous_update=False, description='O*Lo/C gs')
        max_slide = widgets.FloatSlider(min=0, max=5, step=0.1, value=high_mult, continuous_update=False, description='O*Hi/C k')
        area_slide = widgets.IntSlider(min=0, max=500, step=1, value=too_small, continuous_update=False, description='Min A')
        done_button = widgets.Button(description="Done", button_style='success')

        clip_min = widgets.FloatSlider(min=0, max=1, step=0.01, value=0, continuous_update=True, description='Clip Min')
        clip_max = widgets.FloatSlider(min=0, max=1, step=0.01, value=1, continuous_update=True, description='Clip Max')
        pre_gaus = widgets.FloatSlider(min=0, max=3, step=0.01, value=0, continuous_update=True, description='Pre-Gauss')
        method_box = widgets.Dropdown(options=['otsu', 'canny'], value=method, description='Method:')

        otsu_box=widgets.HBox([min_slide, max_slide, area_slide, done_button])
        clip_box=widgets.HBox([clip_min, clip_max, pre_gaus, method_box])


        def update_plot(min_slide, max_slide, area_slide, clip_min, clip_max, pre_gaus):
            autosegment_threshold(img=img, thresh=thresh, low_mult=min_slide, high_mult=max_slide, too_small=area_slide, pre_gaus=pre_gaus,
                                  clip_min=clip_min, clip_max=clip_max, no_show=False, output=None, method=method_box.value)
        def save_autoroi(x, annotated_im=annotated_im):
            accum_img = accumulate(full_img)
            label_image, auto_rois = autosegment_threshold(img=img, thresh=thresh, low_mult=min_slide.value, high_mult=max_slide.value,
                                                           too_small=area_slide.value, clip_min=clip_min.value, clip_max=clip_max.value,
                                                           pre_gaus=pre_gaus.value, no_show=True, output=output, method=method_box.value)
            if annotated_im is None:
                annotated_im = np.zeros(accum_img.shape[-2:])
            ann_im, stats_table = parse_ROIs(objects=label_image, grp_col='auto', c1=accum_img.loc[comp1, 'a', :, :], c2=accum_img.loc[comp2, 'a', :, :],
                                             annotated_im=annotated_im, im=accum_img.sel(frame='a'), stats=list(), count_stat=count_stat)

            summary_ratio_img_and_table(annotated_image=ann_im, stats_table=stats_table, species=full_img.species.values, frame='a', compare1=comp1,
                                        compare2=comp2, output=output, no_show=no_show)


        out = widgets.interactive_output(update_plot, {'min_slide':min_slide, 'max_slide':max_slide, 'area_slide':area_slide,
                                                       'clip_min':clip_min, 'clip_max':clip_max, 'pre_gaus':pre_gaus})
        display(clip_box, otsu_box, out)

        done_button.on_click(save_autoroi)


    except NameError:  # matplotlib in terminal
        label_image, auto_rois = autosegment_threshold(img=img, thresh=thresh, low_mult=low_mult, high_mult=high_mult, too_small=too_small,
                                                       no_show=no_show, output=output)
        return label_image, auto_rois


def rescale_CZI(f_img, aligned_image, f_pix_size=0.04, n_pix_size=0.068):
    # make a dummy dataset with goal resolution
    print(f_img.shape)
    new_x_pix = int(np.ceil(f_pix_size*f_img.shape[-1]/n_pix_size))
    new_y_pix = int(np.ceil(f_pix_size*f_img.shape[-2]/n_pix_size))
    mask_res = xr.DataArray(data=np.zeros((new_y_pix, new_x_pix)), #f_img.shape[-2:]),
                            dims=['Y','X'],
                            coords={'X': np.arange(new_x_pix)*n_pix_size, #np.linspace(0,f_img.shape[-1]-1, f_img.shape[-1]) * (size / aligned_image.data.shape[-1]),
                                    'Y': np.arange(new_y_pix)*n_pix_size})#np.linspace(0,f_img.shape[-2]-1, f_img.shape[-2]) * (size / aligned_image.data.shape[-1])})
    print(mask_res.shape)
    mask_res = mask_res.to_dataset(name='mask')
    #print(mask_res)
    #print(mask_res.coords)
    # correct CZI resolution to .IM resolution
    f_img2 = f_img.interp_like(mask_res, method='linear', assume_sorted=True).dropna(dim='X', how='all').dropna(dim='Y', how='all')
    print('done rescaling to microns')
    return f_img2

def parse_CZI(czi, aligned_image, im_plane='14N 12C', size=9.993):
    f_img = AICSImage(czi)
    try:
        f_pix_size = [float(c.text) for c in f_img.metadata[0][1][1][0][0] if c.tag.endswith('ScalingX')][0]*1e6 #if c.tag.endswith('Scaling')]
    except IndexError:
        try:
            f_pix_size = float(f_img.metadata[0][2][0].text.split(',')[1])/100
        except IndexError:
            sys.exit('Pixel metdata not found either way, did CZI metadata format change? debug this fresh evil')

    n_pix_size = float(size/(aligned_image.shape[-1] + 2))
    f_img = f_img.xarray_data.assign_coords({'X': f_img.xarray_data.coords['X'],
                                             'Y': f_img.xarray_data.coords['Y']})
    #print(f_img.coords['Y'])

    aligned_image = add_pseudoframe(aligned_image)
    # add coordinates
    aligned_image = aligned_image.rename({'x': 'X', 'y': 'Y'})
    aligned_image = aligned_image.assign_coords({'X': aligned_image['X'].values * (size / aligned_image.data.shape[3]),
                                                 'Y': aligned_image['Y'].values * (size / aligned_image.data.shape[3])})

    return f_img, f_pix_size
    for c in [0,1,3]:
        prev_y = 0
        f_img_t = f_img.sel(C=f_img.C.values[c],T=0,Z=0).data
        #f_img_t = rescale_CZI(f_img.sel(C=f_img.C.values[c],T=0,Z=0), aligned_image, size=size).data
        for y_seg in np.linspace(0, f_img_t.shape[-1], num=5)[1:]:
            prev_x = 0
            y_seg = round(y_seg)
            for x_seg in np.linspace(0, f_img_t.shape[-2], num=5)[1:]:
                #working_f = f_img_t.data[0,c,0,prev_x:x_seg, prev_y:y_seg]
                x_seg = round(x_seg)
                #print(f_img_t)
                working_f = f_img_t[prev_x:x_seg, prev_y:y_seg]
                #print(working_f)
                working_f = np.transpose(working_f)  # if off by 90?
                working_f = working_f / np.max(working_f)
                prev_x = x_seg

                matched = feature.match_template(working_f, working_al)
                print('matched')
                print(matched.shape)
                print(np.max(matched))
                x, y = np.unravel_index(np.argmax(matched), matched.shape)

                working_f = working_f[x:(x+254), y:(y+254)]

                fig, (ax1, ax2) = plt.subplots(1,2)
                ax1.imshow(working_al, cmap='Greys_r', interpolation='none')
                ax1.set_title('NanoSIMS image')
                ax2.imshow(working_f, alpha=1, cmap='Greys_r', interpolation='none')
                ax2.set_title('Zeiss image: c' + str(c) + " x" + str(x_seg) + " y" + str(y_seg))
                plt.show()

            prev_y = y_seg

    sys.exit(':()')



def process_nanosims(inp='', output=None, fr=None, args=args, add=False, config=None, batch=False, rough=False, no_show=False,
                     compare1='15N 12C', compare2='14N 12C', filter_method=None, sigma=1, whole_image=False, roi=None, autoroi=False,
                     count_stat='mean', min_area=100, otsu_mult_high=0.5, otsu_mult_low=1.2, min_quant=0, max_quant=1, czi=None, ipad=False,
                     min_value=None, max_value=None):

    aligned_image = load_image(path=inp, config=config, args=args, fr=fr, add=add)

    if config is not None:
        with open(config, 'r') as f:
            c = safe_load(f)
            inp = c['input']
            fr = c['frame']

    if output is None:
        prefix = re.sub(".im$", "", inp)
        Path(prefix).mkdir(exist_ok=True)
        output = prefix + '/' + re.sub("^.*/", "", prefix)
    if filter_method:
        aligned_image = apply_filter(img=aligned_image, filt_method=filter_method, sigma=sigma)
    if args.czi is not None:
        parse_CZI(args.czi, aligned_image)
        print('done')
        sys.exit()
    if not batch:
        interactive_plot(aligned_image, inp, output)
        try:
            get_ipython().__class__.__name__  # jupyter format interaction:
            print("Good luck with ROIs :')")
            return
        except NameError:
            sys.exit("Good luck with ROIs :')")
    if rough:
        save_plot(aligned_image, output=output, no_show=no_show, min_quant=min_quant, max_quant=max_quant)
        if not autoroi:
            print('Exiting without trying to auto-roi, please use with --autoroi flag if you want it to try')
            return


    # PREP STATS
    stats_table = list()
    annotated_image = np.zeros(aligned_image.data.shape[2:4])

    # PARSE ROIS OR AUTOSEGMENT TO GENERATE ROIS
    if autoroi:
        auto_rois, auto_objects = autosegment(full_img=aligned_image, low_mult=otsu_mult_low, high_mult=otsu_mult_high, too_small=min_area, no_show = no_show, output=output)

    if roi is not None:
        rois = imread(roi, pilmode='RGBA')

        # hack becasue DPI is different between saved png and imported roi
        rois = get_image_from_raw_rois(roi=rois, im=aligned_image, ipad=ipad)

        # 3 groups of rois - red, green, blue
        red_rois = rois[:, :, 0]-rois[:, :, 2]
        red_rois = 1*(red_rois > 220)
        green_rois = rois[:, :, 1]-rois[:, :, 0]
        green_rois = 1*(green_rois > 80)
        if not ipad:
            blue_rois = rois[:, :, 2]-rois[:, :, 1]
            blue_rois = 1*(blue_rois > 200)
        else:
            blue_rois = np.zeros(red_rois.shape)
        # segment by contiguous colors by category
        red_objects = measure.label(red_rois, return_num=True)
        green_objects = measure.label(green_rois, return_num=True)
        blue_objects = measure.label(blue_rois, return_num=True)

        # make a nice annotated image so rois are numbered to correspond to future stats_table
        obj = np.where(red_objects[0] > 0, red_objects[0], 0) + \
              np.where(green_objects[0] > 0, green_objects[0] + red_objects[1], 0) + \
              np.where(blue_objects[0] > 0, blue_objects[0] + red_objects[1] + green_objects[1], 0)

        colors_needed = np.hstack((np.repeat('black', 1), np.repeat('red', red_objects[1]), np.repeat('green', green_objects[1]), np.repeat('blue', blue_objects[1])))

        cmap, norm = colors.from_levels_and_colors( levels=range(len(colors_needed)), extend='max', colors= colors_needed) #np.unique(obj)
        plt.imshow(obj, cmap=cmap, norm=norm, interpolation='none')

        for region in measure.regionprops(measure.label(red_rois)):
            ry, rx = region.centroid
            plt.text(rx, ry, s=region.label, ha='center', va='center', c='white')
        for region in measure.regionprops(measure.label(green_rois)):
            ry, rx = region.centroid
            plt.text(rx, ry, s=region.label, ha='center', va='center', c='white')
        for region in measure.regionprops(measure.label(blue_rois)):
            ry, rx = region.centroid
            plt.text(rx, ry, s=region.label, ha='center', va='center', c='white')
        plt.savefig(fname=output + "_roi_table.png")
        if not no_show:
            plt.show(block=True)
        plt.cla()
    elif not whole_image and not autoroi:
        print("No ROI, saving pngs for each channel so you can use for making ROIs (or rerun with -r),\n \
        OR use with --autoroi feature (running now so you can see what it looks like)")
        save_plot(aligned_image, output=output, no_show=no_show, min_quant=min_quant, max_quant=max_quant)
        auto_rois, auto_objects = autosegment(full_img=aligned_image, low_mult=otsu_mult_low, high_mult=otsu_mult_high, too_small=min_area, no_show = no_show, output=output)

    # PROCESS ROIS COMPUTED ABOVE
    for frame in aligned_image.frame.values:
        # calculate ratio of desired vs (desired + ref)
        comp1 = aligned_image.loc[compare1, frame, :, :]
        comp2 = aligned_image.loc[compare2, frame, :, :]

        if whole_image:
            ratio_image = comp1/(comp1 + comp2)
            ratio_image = np.nan_to_num(ratio_image)

            plt.cla()
            norm = colors.Normalize(min_value, max_value, clip=True)
            plt.imshow(ratio_image, interpolation='none', norm=norm, cmap='cividis')
            plt.axis('off')
            plt.colorbar().set_label("Fraction: " + sims.utils.format_species(compare1) + " / (" +
                                     sims.utils.format_species(compare1)+" + " +
                                     sims.utils.format_species(compare2) + ")")
            plt.title(sims.utils.format_species(compare1) + " / (" + sims.utils.format_species(compare1) + " + " +
                      sims.utils.format_species(compare2) + ")")
            plt.savefig(fname=output + "_whole_f" + str(frame) +
                              "_ratio" + re.sub(" ", "_", compare1) + "-x-" + re.sub(" ", "_", compare2) + ".png")
            if not no_show:
                plt.show(block=True)
            if not (roi or autoroi):
                continue

        if roi:
            try:
                #annotated_image, stats_table = parse_ROIs(objects=red_objects, grp_col='red', c1=comp1, c2=comp2,
                annotated_image, stats_table = parse_ROIs(objects=red_rois, grp_col='red', c1=comp1, c2=comp2,
                                                          im=aligned_image.loc[:, frame, :, :],
                                                          annotated_im=annotated_image, stats=stats_table, count_stat=count_stat)
            except (TypeError, NameError):
                print("No red, continuing")
            try:
                annotated_image, stats_table = parse_ROIs(objects=green_rois, grp_col='green', c1=comp1, c2=comp2,
                                                          im=aligned_image.loc[:, frame, :, :],
                                                          annotated_im=annotated_image, stats=stats_table, count_stat=count_stat)
            except (TypeError, NameError):
                print("No green, continuing")
            try:
                annotated_image, stats_table = parse_ROIs(objects=blue_rois, grp_col='blue', c1=comp1, c2=comp2,
                                                          im=aligned_image.loc[:, frame, :, :],
                                                          annotated_im=annotated_image, stats=stats_table, count_stat=count_stat)
            except (TypeError, NameError):
                print("No blue huh?")
            if args.shuffle_roi:
                plt.cla()
                size = annotated_image.shape[-1]
                background_95 = np.quantile(aligned_image.loc[compare1, 0, :, :].sum() / (aligned_image.loc[compare1, 0, :, :].sum() + aligned_image.loc[compare2, 0, :, :].sum()), 0.95)
                for r in stats_table:
                    pad = r[-2] + 2
                    minor = r[-1]
                    major = r[-2]
                    rand_centers = np.random.randint(pad, size-pad, (1000,2))
                    outlist = np.empty(1000)
                    for i,c in enumerate(rand_centers):
                        xx, yy = draw.ellipse(c[0], c[1], minor, major)
                        comp1 = aligned_image.loc[compare1, 0, xx, yy]
                        comp2 = aligned_image.loc[compare2, 0, xx, yy]
                        outlist[c] = comp1.sum() / (comp1.sum() + comp2.sum())
                    outlist = np.nan_to_num(outlist)

                    sns.kdeplot(outlist).set(title=r[0] + " " + str(r[1]) + ": " + str(r[-4]))
                    plt.axvline(np.quantile(outlist, 0.95), 0, 1, color='red')
                    plt.axvline(background_95, 0, 1, color='blue')
                    plt.axvline(r[-4], 0, 1, color='black')
                    plt.xlabel('Fraction (out of 1)')
                    plt.show()
        elif autoroi or not whole_image:
                annotated_image, stats_table = parse_ROIs(objects=auto_rois, grp_col='auto', c1=comp1, c2=comp2,
                                                          im=aligned_image.loc[:, frame, :, :],
                                                          annotated_im=annotated_image, stats=stats_table, count_stat=count_stat)

        summary_ratio_img_and_table(annotated_image=annotated_image, stats_table=stats_table, species=aligned_image.species.values, frame=frame,
                                    compare1=compare1, compare2=compare2, output=output, no_show=no_show)




##############
#    BODY    #
##############
if __name__ == "__main__":
    process_nanosims(inp=args.input, output=args.output, fr=args.frame, args=args, add=args.add, config=args.config, batch=args.batch,
                     compare1=args.compare1, compare2=args.compare2, filter_method=args.filter_method, sigma=args.sigma,
                     rough=args.rough, whole_image=args.whole_image, no_show=args.no_show, roi=args.roi, autoroi=args.autoroi,
                     count_stat=args.count_stat, min_area=args.min_area, min_quant=args.minquant, max_quant=args.maxquant,
                     otsu_mult_high=args.otsu_mult_high, otsu_mult_low=args.otsu_mult_low, czi=args.czi, ipad=args.ipad,
                     min_value=args.minvalue, max_value=args.maxvalue)
