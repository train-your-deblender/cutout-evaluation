import sys
import glob
import datetime
import os
from os.path import exists, join, isdir, split, abspath, dirname, basename, splitext
import argparse
import subprocess
from collections import namedtuple, defaultdict
from functools import partial, wraps
from itertools import chain
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
matplotlib.pyplot.style.use('ggplot')
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = 'nearest'
matplotlib.rcParams['image.cmap'] = 'viridis'
matplotlib.cm.viridis.set_bad()

import numpy as np
from scipy.ndimage.interpolation import zoom
from astropy import log
log.setLevel('WARNING')
# log.disable_warnings_logging()
from astropy import convolution
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D

# Where does this pipeline live?
PROJECT_ROOT = abspath(dirname(__file__))

# SExtractor has a max of 1024 elements in a convolution kernel array, 32x32
# but we need an odd-numbered mask dimension
SEXTRACTOR_MAX_MASK_DIM = 31

# Unphysical redshift used to signal absence of data in merged CANDELS catalogs
SENTINEL_ZSPEC = -99.0

# Result types
KernelResult = namedtuple('KernelResult', ['fits', 'table', 'fwhm_arcsec'])
SExtractionResult = namedtuple('SExtractionResult', ['catalog', 'segmap'])
CutoutsResult = namedtuple('CutoutsResult', ['filesets', 'table'])
Dataset = namedtuple('Dataset', ['drz', 'rms'])

def step(func):
    @wraps(func)
    def inner(*args, **kwargs):
        msg = func.__name__ + '('
        if len(args) > 0:
            msg += ', '.join(map(repr, args))
        if len(args) > 0 and len(kwargs) > 0:
            msg += ', '
        if len(kwargs) > 0:
            ', '.join('{}={}'.format(k, repr(v)) for k, v in kwargs.items())
        msg += ')'
        print(msg)
        return func(*args, **kwargs)
    return inner

def dataset_exists(ds):
    for member in ds:
        if not exists(member):
            return False
    return True

def ensure_dir(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError:
        if not isdir(dirpath):
            raise
    return dirpath

def _clean_candels_header(hdr):
    bad_keys = (
        'CPERR1',
        'CPDIS1',
        'DP1.EXTVER',
        'DP1.NAXES',
        'DP1.AXIS.1',
        'DP1.AXIS.2',
        'CPERR2',
        'CPDIS2',
        'DP2.EXTVER',
        'DP2.NAXES',
        'DP2.AXIS.1',
        'DP2.AXIS.2',
        'A_1_2',
        'A_0_3',
        'A_0_2',
        'B_2_1',
        'A_3_0',
        'A_2_1',
        'B_3_0',
        'B_2_0',
        'A_1_3',
        'A_ORDER',
        'B_0_4',
        'A_2_2',
        'A_0_4',
        'B_ORDER',
        'B_1_1',
        'B_1_2',
        'A_1_1',
        'A_3_1',
        'B_4_0',
        'B_2_2',
        'A_2_0',
        'B_0_3',
        'B_1_3',
        'B_0_2',
        'A_4_0',
        'B_3_1'
    )
    current_keys = list(hdr.keys())
    for k in bad_keys:
        if k in current_keys:
            hdr.remove(k)
    return hdr

# def process_file(fn):
#     fitshdul = fits.open(fn, mode='update')
#     for hdu in fitshdul:
#         clean_header(hdu.header)
#     fitshdul.close()

@step
def clean_candels_headers(data_root, datasets):
    output_datasets = []
    output_dir = join(data_root, 'scratch', 'clean_headers')
    ensure_dir(output_dir)
    for dataset in datasets:
        _, rms = split(dataset.rms)
        _, drz = split(dataset.drz)
        output_datasets.append(Dataset(
            drz=join(output_dir, drz),
            rms=join(output_dir, rms)
        ))

    for dataset, output_dataset in zip(datasets, output_datasets):
        if dataset_exists(output_dataset):
            continue
        for attr in ('drz', 'rms'):
            image = getattr(dataset, attr)
            output = getattr(output_dataset, attr)

            # clean header
            hdul = fits.open(image)
            for header_unit in hdul:
                _clean_candels_header(header_unit.header)
            hdul[0].header['BASENAME'] = basename(dataset.drz).replace('_drz.fits', '')
            hdul.writeto(output)
    return output_datasets

@step
def collect_candels_originals(field_name):
    drz_pattern = '*_drz.fits'
    glob_pattern = join(PROJECT_ROOT, 'candels', field_name, 'originals', drz_pattern)
    drz_files = glob.glob(glob_pattern)
    output_datasets = []
    for fn in drz_files:
        assert exists(fn), 'lol wut? {}'.format(fn)
        rms_fn = fn.replace('_drz.fits', '_rms.fits')
        assert exists(rms_fn), "{} has no rms image {}".format(fn, rms_fn)
        output_datasets.append(Dataset(drz=fn, rms=rms_fn))
    return output_datasets

@step
def exclusion_list_for(data_root, field_name, kernel_fwhm_arcsec):
    list_filename = join(data_root, 'scratch', 'exclude_{}_{:2.2f}arcsec.txt'.format(field_name, kernel_fwhm_arcsec))
    if exists(list_filename):
        return list_filename
    with open(list_filename, 'w') as f:
        f.write(
            '# Exclude these blend IDs from the final list (one per line)\n'
        )
    return list_filename

@step
def create_kernel(kernel_fwhm_arcsec, pixel_scale):
    """
    Basically, SExtractor has a maximum kernel size. The stddev of the kernel
    is the same as the one used for convolution, but we need to generate one
    one evaluated only on 31x31 px for use in the detection step.
    """
    outdir = ensure_dir(join(PROJECT_ROOT, 'output', 'kernels'))
    kernel_table_filename = join(outdir, 'gauss_{0:2.2f}_{1:1.3f}_{2}x{2}.conv'.format(kernel_fwhm_arcsec, pixel_scale, SEXTRACTOR_MAX_MASK_DIM))
    kernel_fits_filename = join(outdir, 'gauss_{:2.2f}_{:1.3f}.fits'.format(kernel_fwhm_arcsec, pixel_scale))

    if exists(kernel_fits_filename) and exists(kernel_table_filename):
        return KernelResult(fits=kernel_fits_filename, table=kernel_table_filename, fwhm_arcsec=kernel_fwhm_arcsec)

    kernel_fwhm_pix = kernel_fwhm_arcsec / pixel_scale
    kernel_sigma_pix = kernel_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))

    kernel_small = convolution.Gaussian2DKernel(
        stddev=kernel_sigma_pix,
        x_size=SEXTRACTOR_MAX_MASK_DIM,
        y_size=SEXTRACTOR_MAX_MASK_DIM
    )
    with open(kernel_table_filename, 'w') as f:
        f.write(
            'CONV NORM\n'
            '# {1}x{1} convolution mask of a Gaussian PSF with FWHM = {0:2.2f} px\n'.format(kernel_fwhm_pix, SEXTRACTOR_MAX_MASK_DIM)
        )
        for row in kernel_small.array:
            outstr = ' '.join(list(map(lambda x: '{:1.9e}'.format(x), row)))
            f.write(outstr + '\n')

    k = convolution.Gaussian2DKernel(stddev=kernel_sigma_pix)
    imhdu = fits.ImageHDU(k.array)
    imhdu.header['FWHMPIX'] = (kernel_fwhm_pix, 'Kernel FWHM in pixels')
    imhdu.header['FWHMAS'] = (kernel_fwhm_arcsec, 'Kernel FWHM in arcseconds')
    imhdu.header['PIXELSCL'] = (pixel_scale, 'arseconds per pixel')
    imhdu.writeto(kernel_fits_filename, clobber=True)

    return KernelResult(fits=kernel_fits_filename, table=kernel_table_filename, fwhm_arcsec=kernel_fwhm_arcsec)

@step
def get_phosim_kernel(bandpass):
    lookup = {
        'u': 'phosim_0.90arcsec_u',
        'g': 'phosim_0.82arcsec_g',
        'r': 'phosim_0.78arcsec_r',
        'i': 'phosim_0.76arcsec_i',
        'z': 'phosim_0.74arcsec_z',
        'y': 'phosim_0.73arcsec_y',
    }
    kernel_fits_filename = join('.', 'data', 'phosim_psfs', lookup[bandpass.lower()] + '.fits')
    kernel_fwhm_arcsec = fits.getval(kernel_fits_filename, 'FWHMAS', ext=1)
    kernel_table_filename = join('.', 'data', 'phosim_psfs', lookup[bandpass.lower()] + '_31x31.conv')
    assert exists(kernel_fits_filename)
    assert exists(kernel_table_filename)
    return KernelResult(fits=kernel_fits_filename, table=kernel_table_filename, fwhm_arcsec=kernel_fwhm_arcsec)

def convolve_in_place(fits_kernel_filename, kernel_fwhm_arcsec, image_filename):
    kernel = fits.getdata(fits_kernel_filename)
    image = fits.open(image_filename, mode='update', memmap=False)
    if image[0].header['CONVOLVE'] == 'COMPLETE':
        return image_filename
    else:
        d = image[0].data
        dmin, dmean, dmax = np.nanmin(d), np.nanmean(d), np.nanmax(d)
        if dmin == dmean == dmax:
            print('DEBUG: no data in {}, skipping'.format(image_filename))
            return image_filename

        mark_start = datetime.datetime.now()
        smeared_data_fft = convolution.convolve_fft(d, kernel)
        mark_end_convolve = datetime.datetime.now()
        print('Convolved {} in {}'.format(image_filename, mark_end_convolve - mark_start))
        image[0].data = smeared_data_fft
        image[0].header['CONVOLVE'] = 'COMPLETE'
        image.close()
    return image_filename

def make_bboxes(shape, box_dimension, pad_pix):
    pix_up, pix_across = shape
    boxes_across = pix_across // box_dimension
    if pix_across % box_dimension > 0:
        boxes_across += 1
    boxes_up = pix_up // box_dimension
    if pix_up % box_dimension > 0:
        boxes_up += 1

    box_bottom = np.arange(boxes_up) * box_dimension
    box_left = np.arange(boxes_across) * box_dimension
    assert box_bottom[0] == 0
    assert box_left[0] == 0

    box_top = box_bottom + box_dimension
    box_top[-1] = pix_up
    box_right = box_left + box_dimension
    box_right[-1] = pix_across

    for (bottom, top) in zip(box_bottom, box_top):
        for (left, right) in zip(box_left, box_right):
            pad_bottom, pad_top = max(bottom - pad_pix, 0), min(top + pad_pix, pix_up)
            pad_left, pad_right = max(left - pad_pix, 0), min(right + pad_pix, pix_across)
            yield ((pad_left, pad_bottom), (pad_right, pad_top))

def carve_image_to_tiles(filename, tile_size, pad_pix):
    dest_dir = filename.replace('.fits', '_tiles')
    ensure_dir(dest_dir)
    mosaic = fits.open(filename)
    tile_filenames = []
    for idx, ((left, bottom), (right, top)) in enumerate(make_bboxes(mosaic[0].data.shape, tile_size, pad_pix)):
        outfile = os.path.join(dest_dir, 'tile_{:04}.fits'.format(idx))
        tile_filenames.append(outfile)
        if os.path.exists(outfile):
            continue
        chunk = fits.PrimaryHDU(mosaic[0].data[bottom:top, left:right])
        chunk.header['SRCIMG'] = os.path.split(filename)[1]
        chunk.header['BOTTOM'] = bottom
        chunk.header['TOP'] = top
        chunk.header['LEFT'] = left
        chunk.header['RIGHT'] = right
        chunk.header['CONVOLVE'] = 'PERFORM'
        hdulist = fits.HDUList([chunk])
        hdulist.writeto(outfile, output_verify='exception', clobber=True)
        hdulist.close()
        print(outfile)
    return tile_filenames


def reassemble_tile_set(image_filename, tile_set, output_filename, tile_size, pad_pix, kernel_fwhm_arcsec):
    if os.path.exists(output_filename):
        raise RuntimeError("Output destination file {} exists!".format(output_filename))
    mosaic = fits.open(image_filename)
    padded_bboxes = make_bboxes(mosaic[0].data.shape, tile_size, pad_pix)
    bboxes = make_bboxes(mosaic[0].data.shape, tile_size, 0)
    for idx, (bbox, padded_bbox, tile_path) in enumerate(zip(bboxes, padded_bboxes, sorted(tile_set))):
        print('bbox:', bbox)
        print('padded_bbox:', padded_bbox)
        (left, bottom), (right, top) = bbox
        (padded_left, padded_bottom), (padded_right, padded_top) = padded_bbox
        data = fits.getdata(tile_path)

        # edge tiles have padding < pad_pix on up to two sides
        # so make sure *up to* pad_pix of excess are trimmed
        # (I guess there's the case of a tile size larger than the entire image, but that's unlikely)
        left_padding = left - padded_left if padded_left != left else None
        right_padding = right - padded_right if padded_right != right else None
        top_padding = top - padded_top if padded_top != top else None
        bottom_padding = bottom - padded_bottom if padded_bottom != bottom else None
        print(bottom_padding, top_padding, ',', left_padding, right_padding)

        mosaic[0].data[bottom:top, left:right] = data[bottom_padding:top_padding, left_padding:right_padding]
        # mosaic[0].data[bottom:top, left:right] = data[bottom_padding:bottom_padding + tile_size, left_padding:tile_size]
    mosaic[0].header['KERNFWHM'] = (kernel_fwhm_arcsec, 'Seeing-sim kernel FWHM in arcseconds')
    mosaic_basename = splitext(basename(output_filename))[0]
    mosaic_basename = mosaic_basename.replace('_drz', '').replace('_rms', '')
    mosaic[0].header['BASENAME'] = (mosaic_basename, 'base name for mosaic dataset')
    mosaic.writeto(output_filename)
    return output_filename

@step
def make_seeing_sim(data_root, kernel_result, input_datasets, tile_size, pad_pix):
    kernel_fwhm_arcsec = kernel_result.fwhm_arcsec
    kernel_filename = kernel_result.fits
    output_datasets = []
    output_dir = join(data_root, 'scratch', 'make_seeing_sim')
    ensure_dir(output_dir)
    for ds in input_datasets:
        _, rms = split(ds.rms)
        _, drz = split(ds.drz)
        output_ds = Dataset(
            drz=join(output_dir, drz.replace('_drz.fits', '_{:2.2f}arcsec_drz.fits'.format(kernel_fwhm_arcsec))),
            rms=join(output_dir, rms.replace('_rms.fits', '_{:2.2f}arcsec_rms.fits'.format(kernel_fwhm_arcsec)))
        )
        output_datasets.append(output_ds)

        if dataset_exists(output_ds):
            continue
        for input_fn, output_fn in zip(ds, output_ds):
            convolver = partial(convolve_in_place, kernel_filename, kernel_fwhm_arcsec)
            tile_set = list(map(convolver, carve_image_to_tiles(input_fn, tile_size, pad_pix)))
            reassemble_tile_set(input_fn, tile_set, output_fn, tile_size, pad_pix, kernel_fwhm_arcsec)

    return output_datasets


def downsample(img, downsample_factor):
    '''Downsample 2D array `img` by a factor of `downsample_factor` along each axis'''
    rows, cols = img.shape
    new_shape = rows // downsample_factor, downsample_factor, cols // downsample_factor, downsample_factor
    return img.reshape(new_shape).sum(-1).sum(1)

# modified from scale_image() (from QUIP via stginga)
# https://github.com/spacetelescope/stginga/blob/192411a2e4344f588a4ceeacfbbb402ba601498d/stginga/utils.py#L271
# now uses integer-width bins to resample image and conserve flux
def downsample_image(hdu_list, downsample_factor, ext='PRIMARY', debug=False):
    """Rescale the image size in the given extension
    by the given zoom factor and adjust WCS accordingly.
    Both PC and CD matrices are supported. Distortion is not
    taken into account; therefore, this does not work on an
    image with ``CTYPE`` that ends in ``-SIP``.

    .. note::

        WCS transformation provided by Mihai Cara.
        Some warnings are suppressed.

    Parameters
    ----------
    hdu : fits.HDUList
        FITS Image with WCS information in the `ext` header
    downsample_factor : int
        number of pixels to bin along each axis
        (e.g. 2 means 2x2 pixels correspond to 1 output pixel)
    ext : str
        Extension to transform (default: 'PRIMARY')
    debug : bool
        If `True`, print extra information to screen.

    Raises
    ------
    ValueError
        Unsupported number of dimensions or invalid WCS.
    """
    prihdr = hdu_list['PRIMARY'].header
    hdr = hdu_list[ext].header

    if hdu_list[ext].data.ndim != 2:
        raise ValueError('Unsupported ndim={0}'.format(data.ndim))

    # Scale the data.
    hdu_list[ext].data = downsample(hdu_list[ext].data, downsample_factor)

    # Adjust WCS
    # slice_factor = 1 / zoom_factor
    slice_factor = downsample_factor
    old_wcs = WCS(hdr)

    if 'SIP' in old_wcs.wcs.ctype[0] or 'SIP' in old_wcs.wcs.ctype[1]:
        raise ValueError('Unsupported projection/distortion type: {}'.format(old_wcs.wcs.ctype))

    # Supress RuntimeWarning about ignoring cdelt because cd is present.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # Update if cropping
        new_wcs = old_wcs.slice(
            (np.s_[::slice_factor], np.s_[::slice_factor]))

    if old_wcs.wcs.has_pc():  # PC matrix
        wshdr = new_wcs.to_header()
    elif old_wcs.wcs.has_cd():  # CD matrix
        # Supress RuntimeWarning about ignoring cdelt because cd is present
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            new_wcs.wcs.cd *= new_wcs.wcs.cdelt

        new_wcs.wcs.set()
        wshdr = new_wcs.to_header()

        for i in range(1, 3):
            for j in range(1, 3):
                key = 'PC{0}_{1}'.format(i, j)
                if key in wshdr:
                    wshdr.rename_keyword(key, 'CD{0}_{1}'.format(i, j))
    else:
        raise ValueError('Missing CD or PC matrix for WCS')

    hdr.extend(wshdr.cards, update=True)

    if debug:
        old_wcs.printwcs()
        new_wcs.printwcs()

    return hdu_list

@step
def downsample_datasets(data_root, input_datasets, pixel_scale, downsample_factor):
    output_datasets = []
    for ds in input_datasets:
        pixel_scale_file_part = '{:03.0f}mas'.format(pixel_scale * 1000)
        assert pixel_scale_file_part in ds.drz
        assert pixel_scale_file_part in ds.rms
        new_scale = '{:03.3f}arcsec_per_px'.format(downsample_factor * pixel_scale)
        output_ds = Dataset(
            drz=ds.drz.replace(pixel_scale_file_part, new_scale),
            rms=ds.rms.replace(pixel_scale_file_part, new_scale),
        )
        output_datasets.append(output_ds)
        if dataset_exists(output_ds):
            continue

        # handle drz image
        drz_hdul = fits.open(ds.drz)
        total_flux = np.nansum(drz_hdul[0].data)
        drz_hdul = downsample_image(drz_hdul, downsample_factor)
        new_total_flux = np.nansum(drz_hdul[0].data)
        assert np.allclose(new_total_flux, total_flux), 'lost too much flux in downsampling (total flux in new_data: {}, total original flux: {})'.format(new_total_flux, total_flux)
        drz_hdul[0].header['BASENAME'] = basename(output_ds.drz).replace('_drz.fits', '')
        drz_hdul.writeto(output_ds.drz)

        # handle rms image
        # > For downsampling RMS, if you are using a block-sum for the images then the
        # > *variance* just sums as well. So that means the RMS is the square root of the
        # > sum of the variances. So simplest is to square the RMS, block-sum it, and take
        # > the square root.
        rms_hdul = fits.open(ds.rms)
        variance = rms_hdul[0].data ** 2
        rms_hdul[0].data = variance
        rms_hdul = downsample_image(rms_hdul, downsample_factor)
        rms_hdul[0].data = np.sqrt(rms_hdul[0].data)
        rms_hdul[0].header['BASENAME'] = basename(output_ds.drz).replace('_drz.fits', '')
        rms_hdul.writeto(output_ds.rms)

    return output_datasets

def _create_sexconfig_file(data_root, mosaic, sextractor_config_filename, kernel_table_filename,
                           kernel_fwhm_arcsec, pixel_scale, sextractor_config_template_filename):
    template = open(os.path.join(PROJECT_ROOT, 'config', 'sextractor', sextractor_config_template_filename)).read()
    mosaic_fits = fits.open(mosaic)
    flag_filename = join(data_root, 'scratch', 'sextract', mosaic_fits[0].header['BASENAME'] + '_flag.fits')
    if not os.path.exists(flag_filename):
        print("Creating fake/empty flag file {}".format(flag_filename))
        mosaic_fits[0].data = np.zeros(mosaic_fits[0].data.shape, dtype=np.uint16)
        mosaic_fits.writeto(flag_filename)

    conf_file = template.format(
        MOSAIC_BASENAME=mosaic_fits[0].header['BASENAME'],
        MOSAIC_PATH_NO_SUFFIX=mosaic.replace('_drz.fits', ''),
        ## N.B. Catalog name format is specified in two places: the config template and the calling function
        PARAMETERS_NAME=join(PROJECT_ROOT, 'config', 'sextractor', 'detection.param'),
        FILTER_NAME=abspath(kernel_table_filename),
        STARNNW_NAME=join(PROJECT_ROOT, 'config', 'sextractor', 'default.nnw'),
        PIXEL_SCALE=pixel_scale,
        SEEING_FWHM='{:2.2f}'.format(kernel_fwhm_arcsec),
        DATA_ROOT=data_root,
    )
    with open(sextractor_config_filename, 'w') as f:
        f.write(conf_file)
        print("Saved new config file to {}".format(sextractor_config_filename))

@step
def sextract(data_root, mosaic_dataset, kernel_result, pixel_scale, sextractor_config_template_filename):
    kernel_fwhm_arcsec = kernel_result.fwhm_arcsec
    print("Processing mosaic {}".format(mosaic_dataset))
    mosaic = mosaic_dataset.drz
    mosaic_basename = fits.getval(mosaic, 'BASENAME')
    output_dir = ensure_dir(join(data_root, 'scratch', 'sextract'))

    sextractor_config_filename = join(output_dir, '{}.sex'.format(mosaic_basename))
    ## N.B. Catalog name format is specified in two places: the config template and this function
    catalog_filename = join(output_dir, '{}.cat'.format(mosaic_basename))
    segmap_filename = join(output_dir, '{}_seg.fits'.format(mosaic_basename))

    if exists(catalog_filename) and exists(segmap_filename):
        print("Catalog and segmap exist for {}, remove from scratch/sextract/ to re-SExtract".format(mosaic))
        return SExtractionResult(catalog=catalog_filename, segmap=segmap_filename)
    else:
        print("Couldn't find:", catalog_filename, segmap_filename)

    if os.path.exists(sextractor_config_filename):
        print("Using existing {}".format(sextractor_config_filename))
    else:
        print("Creating SExtractor config: {}".format(sextractor_config_filename))
        _create_sexconfig_file(data_root, mosaic, sextractor_config_filename,
                               kernel_result.table, kernel_fwhm_arcsec, pixel_scale,
                               sextractor_config_template_filename)

    print("Starting SExtractor on {} with {}".format(mosaic, sextractor_config_filename))
    args = ['sex', mosaic, '-c', sextractor_config_filename]
    print("$ {}".format(' '.join(args)))
    proc = subprocess.Popen(
        args,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )
    for line in proc.stdout:
        sys.stdout.write(line.decode('utf8'))
    return_code = proc.wait()
    if return_code != 0:
        raise Exception("Failed to SExtract {}".format(mosaic))
    print("SExtraction complete on {}".format(mosaic))
    return SExtractionResult(catalog=catalog_filename, segmap=segmap_filename)

def write_association_table(seg2cat, outpath):
    with open(outpath, 'w') as f:
        f.write('# seg_id,cat_id_1,cat_id_2,...\n')
        for seg_id in seg2cat.keys():
            f.write('{},'.format(seg_id) + ','.join(map(str, seg2cat[seg_id])) + '\n')

def read_association_table(inpath):
    association = {}
    for line in open(inpath):
        if line[0] == '#':
            continue
        line_parts = tuple(map(int, line.split(',')))
        seg_id = line_parts[0]
        cat_ids = line_parts[1:]
        association[seg_id] = set(cat_ids)
    return association

@step
def normalize_catalog_columns(data_root, table_filename):
    _, fn = split(table_filename)
    output_filename = join(data_root, 'scratch', fn)
    if exists(output_filename):
        return output_filename
    table = Table.read(table_filename)
    irregular_names_map = {
        # irregular : normalized
        'zspec_q': 'q_zspec',
        'zspec_1': 'zspec',
        'x_image': 'X_IMAGE',
        'y_image': 'Y_IMAGE',
        'RAdeg': 'RA',
        'DECdeg': 'DEC',
    }
    for irregular, normalized in irregular_names_map.items():
        if irregular in table.colnames and normalized not in table.colnames:
            table.rename_column(irregular, normalized)
    table.write(output_filename)
    return output_filename

def _load_exclusion_list(filename):
    excludes = []
    for line in open(filename):
        if line[0] == '#':
            continue
        excludes.append(line.strip())
    return set(excludes)

@step
def identify_blends(data_root, sextraction_result, field_name, catalog_filename, exclusion_list_filename, downsample_factor, kernel_result):
    output_dir = ensure_dir(join(data_root, 'scratch', 'blends'))
    _, sextractor_catalog_filename = split(sextraction_result.catalog)
    filename_base = join(output_dir, sextractor_catalog_filename.replace('.cat', ''))
    blends_to_catalog_map_filename = filename_base + '_seg2cat.txt'
    sextractor_subset_filename = filename_base + '_blends.txt'
    candels_subset_filename = filename_base + '_blends_candels.txt'
    region_overlay_filename = filename_base + '_blends.reg'
    all_blends_list_filename = filename_base + '_blend_names.txt'
    if all(map(exists, (blends_to_catalog_map_filename, sextractor_subset_filename, region_overlay_filename, candels_subset_filename, all_blends_list_filename))):
        return blends_to_catalog_map_filename
    catalog = Table.read(catalog_filename)
    catalog.add_index('ID')
    sexcat = Table.read(sextraction_result.catalog, format='ascii.sextractor')
    sexcat.add_index('NUMBER')
    segmap = fits.getdata(sextraction_result.segmap)
    excludes = _load_exclusion_list(exclusion_list_filename)

    segid_to_catid = defaultdict(set)
    for row in catalog:
        x, y = int(row['X_IMAGE'] // downsample_factor), int(row['Y_IMAGE'] // downsample_factor)
        segid = segmap[y, x]
        if segid != 0:
            segid_to_catid[segid].add(row['ID'])

    nspec, ngood, nreally_good = 0, 0, 0
    really_good = {}

    n_segs = len(segid_to_catid.items())
    all_good_blend_names = []
    for idx, (segid, catids) in enumerate(segid_to_catid.items()):
        # skip segments with a single corresponding catalog entry
        if len(catids) < 2:
            continue
        # implement the exclusion_list
        row = _get_row(sexcat, segid)
        blend_name = format_candels_blend_name(field_name, row['ALPHA_J2000'], row['DELTA_J2000'], kernel_result.fwhm_arcsec)
        if blend_name in excludes:
            continue

        print('evaluating', blend_name, idx + 1, "/", n_segs)
        # subset = catalog[list(catids)]  # think this was producing bogus output
        subset = catalog.loc[list(catids)]
        if np.max(subset['zspec']) == SENTINEL_ZSPEC:
            # print("No redshift from spectra for any component of blend {}".format(segid))
            continue
        zbmin, zbmax = np.min(subset['zbest']), np.max(subset['zbest'])
        # print("seg {} - {} < z < {}".format(segid, zbmin, zbmax))
        nspec += 1
        if np.any(subset['q_zspec'] == 1):
            ngood += 1
            if zbmax - zbmin >= 0.5:
                nreally_good += 1
                really_good[segid] = catids
                all_good_blend_names.append(blend_name)

    print("writing", sextractor_subset_filename)
    # clobber sextractor_subset_filename if exists
    if exists(sextractor_subset_filename):
        os.remove(sextractor_subset_filename)

    sextractor_subset_ids = list(sorted(really_good.keys()))
    sextractor_subset = sexcat.loc[sextractor_subset_ids]
    sextractor_subset.write(sextractor_subset_filename, format='ascii.basic')

    print("writing", candels_subset_filename)
    # clobber candels_subset_filename if exists
    if exists(candels_subset_filename):
        os.remove(candels_subset_filename)

    # add a 'segnum' column to the CANDELS catalog with the SExtractor source segment number
    candels_subset_tables = []
    for sextractor_number, candels_subset_ids in really_good.items():
        candels_subset = catalog.loc[list(sorted(candels_subset_ids))]
        candels_subset['SEGNUM'] = sextractor_number
        candels_subset_tables.append(candels_subset)

    candels_subset_table = vstack(candels_subset_tables)
    candels_subset_table.write(candels_subset_filename, format='ascii.basic')

    print("writing", region_overlay_filename)
    # write region overlay
    with open(region_overlay_filename, 'w') as f:
        f.write('''# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
''')
        for row in candels_subset:
            f.write("point({x},{y}) # point=cross color=magenta text={{z={z}}}\n".format(
                x=row['RA'],
                y=row['DEC'],
                z=row['zbest']
            ))
        for row in sextractor_subset:
            f.write('circle({x},{y},5") # text={{{num}}}\n'.format(
                x=row['ALPHA_J2000'],
                y=row['DELTA_J2000'],
                num=row['NUMBER']
            ))

    print("writing", all_blends_list_filename)
    with open(all_blends_list_filename, 'w') as f:
        f.write('# Name for every good blend identified in this field\n')
        for blend_name in all_good_blend_names:
            f.write(blend_name + '\n')

    print("writing", blends_to_catalog_map_filename)
    write_association_table(really_good, blends_to_catalog_map_filename)
    return blends_to_catalog_map_filename

def format_candels_blend_name(field_name, alpha, delta, kernel_fwhm):
    return '{} {:3.8f}{:+2.7f} at {:2.2f} arcsec'.format(field_name.upper(), alpha, delta, kernel_fwhm)

def _get_row(table, id, id_column='NUMBER'):
    rows = table[table[id_column] == id]
    assert len(rows) == 1
    return rows[0]

def cutout_to_hdu(cutout):
    hdu = fits.ImageHDU(
        data=cutout.data,
        header=cutout.wcs.to_header()
    )
    return hdu

def save_cutout_fits(cutouts_dir, blend_name, detection_thumb, seeing_thumb, segmap_thumb):
    outfile = join(cutouts_dir, blend_name + '.fits')
    seeing_img = fits.ImageHDU(
        data=seeing_thumb.data,
        header=seeing_thumb.wcs.to_header()
    )
    seeing_img.header['EXTNAME'] = 'SEEING'
    detection_img = fits.ImageHDU(
        data=detection_thumb.data,
        header=detection_thumb.wcs.to_header()
    )
    detection_img.header['EXTNAME'] = 'DETECT'
    segmap_img = fits.ImageHDU(
        data=segmap_thumb.data,
        header=segmap_thumb.wcs.to_header()
    )
    segmap_img.header['EXTNAME'] = 'SEGMAP'
    hdul = fits.HDUList([seeing_img, detection_img, segmap_img])
    hdul.writeto(outfile)
    return outfile


def _get_filter_from_hdul(hdul):
    for hdu in hdul:
        if 'FILTER' in hdu.header:
            return hdu.header['FILTER']
        elif 'FILTER1' in hdu.header:
            if 'CLEAR' in hdu.header['FILTER1']:
                return hdu.header['FILTER2']
            elif 'CLEAR' in hdu.header['FILTER2']:
                return hdu.header['FILTER1']
            else:
                raise RuntimeError
    assert False

# ugh
FILTER_TO_WAVELENGTH_UM = {
    'F275W': 0.275,
    'F435W': 0.435,
    'F606W': 0.606,
    'F775W': 0.775,
    'F814W': 0.814,
    'F850L': 0.850,
    'F850LP': 0.850,
    'F098M': 0.98,
    'F105W': 1.05,
    'F110W': 1.10,
    'F125W': 1.25,
    'F127M': 1.27,
    'F139M': 1.39,
    'F140W': 1.40,
    'F153M': 1.53,
    'F160W': 1.60,
}

def _read_drz_sorted(dataset_list):

    loaded_drzes = []
    for dataset in dataset_list:
        drz = fits.open(dataset.drz)
        filtername = _get_filter_from_hdul(drz)
        assert filtername in FILTER_TO_WAVELENGTH_UM.keys()
        wavelength = FILTER_TO_WAVELENGTH_UM[filtername]
        loaded_drzes.append((wavelength, drz))
    loaded_drzes.sort()
    return [a[1] for a in loaded_drzes]

def _make_cutout_dimensions(sexcat_row, pad, convert_pix_factor):
    x_min, x_max = sexcat_row['XMIN_IMAGE'], sexcat_row['XMAX_IMAGE']
    y_min, y_max = sexcat_row['YMIN_IMAGE'], sexcat_row['YMAX_IMAGE']
    x_center, y_center = (x_max + x_min) // 2, (y_max + y_min) // 2
    width = x_max - x_min + pad
    height = y_max - y_min + pad
    return (x_center * convert_pix_factor,
            y_center * convert_pix_factor,
            height * convert_pix_factor,
            width * convert_pix_factor)

def _make_cutout(sexcat_row, image_hdul, pad, convert_pix_factor):
    x_center, y_center, height, width = _make_cutout_dimensions(sexcat_row, pad, convert_pix_factor)
    cutout = Cutout2D(
        image_hdul[0].data,
        (x_center, y_center),
        (height, width),  # yes, this is the order in the docs
        wcs=WCS(image_hdul[0].header, relax=False)
    )
    return cutout

def _detection_filter(image_hduls, detection_image):
    detection_base = basename(detection_image)
    for idx, img_hdul in enumerate(image_hduls):
        orig_filename = basename(img_hdul.fileinfo(0)['filename'])
        if detection_base == orig_filename:
            filtername = _get_filter_from_hdul(img_hdul)
            filter_wavelength = FILTER_TO_WAVELENGTH_UM[filtername]
            return filtername, filter_wavelength, idx
    raise Exception('Detection image not found!')

@step
def make_cutouts(data_root, blends_to_catalog_map_filename, sextraction_result, detection_image, field_name, catalog_filename, original_datasets, seeing_sim_datasets, kernel_result, cutout_pad, downsample_factor):
    kernel_fwhm_arcsec = kernel_result.fwhm_arcsec
    seg2cat = read_association_table(blends_to_catalog_map_filename)
    sexcat = Table.read(sextraction_result.catalog, format='ascii.sextractor')

    assert len(original_datasets) > 0
    assert len(seeing_sim_datasets) > 0
    originals = _read_drz_sorted(original_datasets)
    filters = list(map(_get_filter_from_hdul, originals))
    wavelengths = [FILTER_TO_WAVELENGTH_UM[filtername] for filtername in filters]
    detection_filtername, detection_filter_wavelength, detection_filter_idx = _detection_filter(originals, detection_image)
    seeing_sim = _read_drz_sorted(seeing_sim_datasets)
    assert len(originals) == len(seeing_sim)
    assert len(originals) > 0
    assert len(seeing_sim) > 0
    assert len(filters) > 0
    n_bands = len(originals)
    segmap = fits.open(sextraction_result.segmap)
    catalog = Table.read(catalog_filename)
    catalog.add_index('ID')

    cutout_files = []
    outdir = ensure_dir(join(data_root, 'products'))

    for seg_id, cat_ids in seg2cat.items():
        row = _get_row(sexcat, seg_id)
        blend_name = format_candels_blend_name(field_name, row['ALPHA_J2000'], row['DELTA_J2000'], kernel_fwhm_arcsec)
        filename = blend_name.replace(' ', '_') + '.fits'
        outpath = join(outdir, filename)
        cutout_files.append(outpath)

        if exists(outpath):
            continue

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['NAME'] = blend_name
        primary_hdu.header['FWHMAS'] = (kernel_fwhm_arcsec, 'Kernel FWHM in arcseconds')
        primary_hdu.header['DETIMAGE'] = (basename(detection_image), 'Detection image')
        primary_hdu.header['DETFILTR'] = (detection_filtername, 'Detection image filter name')
        primary_hdu.header['DETWAVEL'] = (detection_filter_wavelength, 'Detection image filter center wavelength')
        primary_hdu.header['DETINDEX'] = (detection_filter_idx, 'Index of cube plane with detection filter')
        extensions = [primary_hdu]

        _, _, truth_cutout_height, truth_cutout_width = _make_cutout_dimensions(row, cutout_pad, convert_pix_factor=downsample_factor)
        _, _, blend_cutout_height, blend_cutout_width = _make_cutout_dimensions(row, cutout_pad, convert_pix_factor=1)

        truth = np.zeros((n_bands, truth_cutout_height, truth_cutout_width))
        blend = np.zeros((n_bands, blend_cutout_height, blend_cutout_width))
        # filters = []
        # wavelengths = []

        for idx, (orig, sl, filtername) in enumerate(zip(originals, seeing_sim, filters)):
            # filtername = _get_filter_from_hdul(orig)
            # filters.append(filtername)
            # wavelengths.append()
            seeing_thumb = _make_cutout(row, sl, cutout_pad, convert_pix_factor=1)
            blend[idx] = seeing_thumb.data

            orig_thumb = _make_cutout(row, orig, cutout_pad, convert_pix_factor=downsample_factor)
            truth[idx] = orig_thumb.data

        # for loops don't make a scope, so use the last seeing_thumb and orig_thumb
        # to add WCS
        blend_hdu = fits.ImageHDU(
            data=blend,
            header=seeing_thumb.wcs.to_header()
        )
        blend_hdu.header['EXTNAME'] = 'BLEND'
        blend_hdu.header['FWHMAS'] = (kernel_fwhm_arcsec, 'Kernel FWHM in arcseconds')
        extensions.append(blend_hdu)

        truth_hdu = fits.ImageHDU(
            data=truth,
            header=orig_thumb.wcs.to_header()
        )
        truth_hdu.header['EXTNAME'] = 'TRUTH'
        extensions.append(truth_hdu)

        # Filter names and wavelengths
        filters_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name='filter', format='5A', array=filters),
            fits.Column(name='wavelength', format='E', array=wavelengths),
        ])
        filters_hdu.header['EXTNAME'] = 'FILTERS'
        extensions.append(filters_hdu)

        # This might be really stupid, but I don't want to assume CANDELS FITS
        # catalogs have consecutive indices...
        catalog_subset = catalog.loc[list(cat_ids)]
        catalog_subset_hdu = fits.table_to_hdu(catalog_subset)
        catalog_subset_hdu.header['EXTNAME'] = 'CATALOG'
        extensions.append(catalog_subset_hdu)

        segmap_thumb = _make_cutout(row, segmap, cutout_pad, convert_pix_factor=1)
        segmap_img = fits.ImageHDU(
            data=segmap_thumb.data,
            header=segmap_thumb.wcs.to_header()
        )
        segmap_img.header['EXTNAME'] = 'SEGMAP'
        extensions.append(segmap_img)

        hdul = fits.HDUList(extensions)
        hdul.writeto(outpath)

    return cutout_files

def plot_for_blend(filename, output_filename):
    blend_fits = fits.open(filename)
    detection_filter_index = blend_fits[0].header['DETINDEX']

    fig, (ax_det, ax_sl, ax_seg) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax_det.set_title('Original')
    ax_sl.set_title('Seeing Limited')
    ax_seg.set_title('Segmap')
    panels = [
        (ax_det, WCS(blend_fits['TRUTH'].header), blend_fits['TRUTH'].data[detection_filter_index]),
        (ax_sl, WCS(blend_fits['BLEND'].header), blend_fits['BLEND'].data[detection_filter_index]),
        (ax_seg, WCS(blend_fits['SEGMAP'].header), blend_fits['SEGMAP'].data),
    ]
    for ax, wcs, data in panels:
        ax.imshow(data)
        ax.set_xlim(0, data.shape[1] - 1)
        ax.set_ylim(0, data.shape[0] - 1)

    # overplot original components only on segmap
    for row in blend_fits['CATALOG'].data:
        ax_seg.scatter(*wcs.all_world2pix(row['RA'], row['DEC'], 0),
                       s=5 + 100 * row['zbest']**2,
                       c='red',
                       alpha=0.5)

    fig.suptitle(basename(filename))
    fig.savefig(output_filename, dpi=150)
    plt.close(fig)

@step
def make_products(cutout_files):
    for filename in cutout_files:
        output_filename = filename.replace('.fits', '.png')
        print(output_filename)
        if not exists(output_filename):
            plot_for_blend(filename, output_filename)

def run_candels(field_name, detection_image, pixel_scale, downsample_factor,
                kernel_fwhm_arcsec, use_phosim, tile_size, pad_tiles, pad_cutouts):
    catalog_filename = join(PROJECT_ROOT, 'candels', field_name, field_name + '_merged_catalog.fits')
    data_root = join(PROJECT_ROOT, 'candels', field_name)
    detection_image_rms = detection_image.replace('_drz.fits', '_rms.fits')

    # intermediate/re-used results
    if use_phosim and (kernel_fwhm_arcsec is not None):
        raise RuntimeError("Can't mix --phosim and --kernel-fwhm")
    if kernel_fwhm_arcsec is not None:
        seeing_kernel = create_kernel(
            kernel_fwhm_arcsec,
            pixel_scale * downsample_factor
        )
        print("DEBUG using a gaussian kernel")
    else:
        seeing_kernel = get_phosim_kernel('r')
        assert abs(pixel_scale * downsample_factor - 0.2) / 0.2 < 0.3, "not even getting CLOSE to the LSST pixel scale at which this PSF was simulated..."
        print("DEBUG using a PhoSim-derived PSF (in 'r' band)")

    detection_dataset = make_seeing_sim(
        data_root,
        seeing_kernel,
        downsample_datasets(
            data_root,
            clean_candels_headers(data_root, [Dataset(drz=detection_image, rms=detection_image_rms), ]),
            pixel_scale,
            downsample_factor
        ),
        tile_size,
        pad_tiles,
    )[0]

    sextraction_result = sextract(
        data_root,
        detection_dataset,
        seeing_kernel,
        pixel_scale,
        'detect_in_seeing_limited.sex.template'
    )

    original_datasets = clean_candels_headers(data_root, collect_candels_originals(field_name))
    seeing_sim_datasets = make_seeing_sim(
        data_root,
        seeing_kernel,
        downsample_datasets(data_root, original_datasets, pixel_scale, downsample_factor),
        tile_size,
        pad_tiles,
    )
    catalog_file = normalize_catalog_columns(data_root, catalog_filename)

    return make_products(
        make_cutouts(
            data_root,
            identify_blends(
                data_root,
                sextraction_result,
                field_name,
                catalog_file,
                exclusion_list_for(
                    data_root,
                    field_name,
                    seeing_kernel.fwhm_arcsec
                ),
                downsample_factor,
                seeing_kernel,
            ),
            sextraction_result,
            detection_image,
            field_name,
            catalog_file,
            original_datasets,
            seeing_sim_datasets,
            seeing_kernel,
            pad_cutouts,
            downsample_factor,
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Positional
    parser.add_argument("data_root",
                        help="Reference/original data root")
    parser.add_argument("field_name",
                        help="Name of the survey field being processed")
    parser.add_argument("catalog_filename",
                        help="FITS catalog filename corresponding to this field")

    # Flags
    parser.add_argument("-d", "--detection-image", required=True,
                        help="Filename of original image whose seeing-sim version will be used to detect sources")
    parser.add_argument("-s", "--pixel-scale", type=float, required=True,
                        help="Pixel scale in arcseconds per pixel")
    parser.add_argument("-f", "--downsample-factor", type=int, required=True,
                        help="Integer downsampling factor along each axis (e.g. factor of 2 turns 4 px into 1 px)")
    parser.add_argument("-k", "--kernel-fwhm", type=float, required=False,
                        help="Gaussian kernel FWHM in arcsec used to generate these images (incompatible with --phosim)")
    parser.add_argument("-ps", "--phosim", action='store_true', required=False,
                        help="Use PhoSim PSF (hardcoded for 'r' band currently, incompatible with --kernel-fwhm)")
    parser.add_argument("-t", "--tile-size", type=int, required=True,
                        help="Size of tiles in pixels for the tiled convolution")
    parser.add_argument("-pt", "--pad-tiles", type=int, required=True,
                        help="Margin to leave on edge of tiles for convolution (in original px)")
    parser.add_argument("-pc", "--pad-cutouts", type=int, required=True,
                        help="Margin to leave on edge of blend cutouts (in final downsampled px)")

    # ex:
    # python pipeline.py --detection-image ./data/egs_all_acs_wfc_f606w_060mas_v1.1_drz.fits --pixel-scale 0.06 --kernel-fwhm 0.7 --tile-size 1024 ./data/ egs ./data/merged_catalogs/egs.fits
    args = parser.parse_args()
    main(args)

    print("Complete!")
