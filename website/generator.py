#!/usr/bin/env python3
from __future__ import division, unicode_literals, print_function
import sys
import os

# astro data wrangling
import matplotlib
matplotlib.use('Agg')
# Set up plotting style defaults
from matplotlib import pyplot as plt
plt.style.use('ggplot')
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.cmap'] = 'magma'
plt.cm.magma.set_bad('black')
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['image.interpolation'] = 'nearest'
from matplotlib.colors import LogNorm

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from photutils.utils import random_cmap
import numpy as np
from PIL import Image

# file, template, and config wrangling
import os.path
from os.path import isdir, splitext, basename, exists, join, abspath, dirname
import errno
import shutil
import subprocess
import glob
import yaml
import markdown2
import logging
from jinja2 import Environment, FileSystemLoader

if sys.version_info > (3, 2):
    def ensure_dir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        return dirpath
else:
    def ensure_dir(dirpath):
        if not isdir(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
        return dirpath

ROOT = abspath(dirname(__file__))
TEMPLATES_ROOT = join(ROOT, 'templates')
STATIC_ROOT = join(ROOT, 'static')
SCSS_FILE_PATH = join(STATIC_ROOT, 'styles.scss')
CSS_OUTPUT_PATH = join(STATIC_ROOT, 'styles.css')
BUILD_ROOT = ensure_dir(join(ROOT, '_build'))

try:
    import blendcollection
except ImportError:
    blendcollection_dir = abspath(join(ROOT, '..', 'blendcollection'))
    sys.path.insert(1, blendcollection_dir)
    import blendcollection

ENV = Environment(loader=FileSystemLoader(TEMPLATES_ROOT))
from markdown2 import Markdown
MD = Markdown()

from jinja2 import evalcontextfilter, Markup, escape

@evalcontextfilter
def markdown(eval_ctx, value):
    result = MD.convert(value)
    if eval_ctx.autoescape:
        result = Markup(result)
    return result

ENV.filters['markdown'] = markdown

def compile_scss():
    command = ['sassc', '-m', SCSS_FILE_PATH, CSS_OUTPUT_PATH]
    print("Running:", ' '.join(command))
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        bufsize=1,
        cwd=STATIC_ROOT,
        universal_newlines=True,
    )
    buf = []
    for line in iter(proc.stdout.readline, ''):
        buf.append(line)
        logging.info(line, end='')  # uncomment this to enable output
    proc.stdout.close()
    returncode = proc.wait()
    if returncode != 0:
        logging.info('*'*72)
        logging.info('sassc exited with returncode != 0 (was {})'.format(returncode))
        logging.info('*'*72)
        logging.info('\n'.join(buf))
        logging.info('*'*72)
    logging.info('Build completed.')

def render_one_cutout(collection_shortname, filename, preview_bands, catalog_columns):
    # open file
    cutout = blendcollection.load(filename)
    image_filenames_base = join(ensure_dir(join(BUILD_ROOT, collection_shortname, 'images')), cutout.id)
    # generate RGB quick-look image
    preview_rgb_filename = image_filenames_base + '_rgb.png'
    if not exists(preview_rgb_filename):
        rgb_image = blendcollection.quick_rgb(*(cutout.truth[b] for b in preview_bands))
        image = Image.fromarray(rgb_image)
        image.save(preview_rgb_filename)
    # generate individual band quick-look images
    preview_all_filename = image_filenames_base + '_preview.png'
    if not exists(preview_all_filename):
        fig, (blended_axs, truth_axs) = plt.subplots(nrows=2, ncols=len(cutout.bands), figsize=(1.25 * len(cutout.bands), 2.5))
        for data, axs in zip((cutout.truth, cutout.blended), (truth_axs, blended_axs)):
            for band, ax in zip(cutout.bands, axs):
                ax.imshow(blendcollection.quick_mono(data[band]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
        for band, ax in zip(cutout.bands, truth_axs):
            ax.set_xlabel(band)

        truth_axs[0].set_ylabel('Truth')
        blended_axs[0].set_ylabel('Blended')
        plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.05, right=1, top=1.0 - 0.05, bottom=0.1)
        fig.savefig(preview_all_filename)
        plt.close(fig)
        #
    # preview_truth_filename = image_filenames_base + '_truth.png'
    # if not exists(preview_truth_filename):
    #     fig_truth, truth_axs = plt.subplots(nrows=1, ncols=len(cutout.bands), figsize=(3 * len(cutout.bands), 4))
    #     for band, ax in zip(cutout.bands, truth_axs):
    #         ax.set_title(band)
    #         ax.imshow(blendcollection.quick_mono(cutout.truth[band]), cmap='magma', origin='lower')
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #     fig_truth.savefig(preview_truth_filename)
    #     plt.close(fig_truth)
    #
    # preview_blended_filename = image_filenames_base + '_blended.png'
    # if not exists(preview_blended_filename):
    #     fig_blended, blended_axs = plt.subplots(nrows=1, ncols=len(cutout.bands), figsize=(3 * len(cutout.bands), 4))
    #     for band, ax in zip(cutout.bands, blended_axs):
    #         ax.set_title(band)
    #         ax.imshow(blendcollection.quick_mono(cutout.blended[band]), cmap='magma', origin='lower')
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #     fig_blended.savefig(preview_blended_filename)
    #     plt.close(fig_blended)
    # truth_preview_filenames = []
    # blend_preview_filenames = []
    # for band in cutout.bands:
    #     band_truth_filename = image_filenames_base + '_{}_truth.png'.format(band)
    #     truth_preview_filenames.append(basename(band_truth_filename))
    #     band_blended_filename = image_filenames_base + '_{}_blended.png'.format(band)
    #     blend_preview_filenames.append(basename(band_blended_filename))
    #     if not exists(band_truth_filename):
    #         band_truth_img = blendcollection.quick_mono(cutout.truth[band])
    #         plt.imsave(band_truth_filename, band_truth_img, cmap='magma')
    #     if not exists(band_blended_filename):
    #         band_blended_img = blendcollection.quick_mono(cutout.blended[band])
    #         plt.imsave(band_blended_filename, band_blended_img, cmap='magma')
    # construct iterable from catalog and columns
    table = []
    for row in cutout.catalog:
        subset_row = []
        for colname, coldesc in catalog_columns:
            subset_row.append((colname, row[colname]))
        table.append(dict(subset_row))
    # return template context dict
    return {
        'cutout': cutout,
        'cutout_filename': basename(filename),
        # 'width_arcsec': width_arcsec,
        # 'height_arcsec': height_arcsec,
        'preview_rgb_filename': basename(preview_rgb_filename),
        'preview_all_filename': basename(preview_all_filename),
        # 'preview_blended_filename': basename(preview_blended_filename),
        # 'truth_preview_filenames': truth_preview_filenames,
        # 'blend_preview_filenames': blend_preview_filenames,
        'red_band': preview_bands[0],
        'green_band': preview_bands[1],
        'blue_band': preview_bands[2],
        'catalog_columns': catalog_columns,
    }


def generate_collection_page(collection):
    collection_dir = ensure_dir(join(BUILD_ROOT, collection['shortname']))
    # collect cutouts from collection
    collection_pattern = join(ROOT, 'collections', collection['directory'], collection['glob_pattern'])
    cutout_filenames = glob.glob(collection_pattern)
    collection['count'] = len(cutout_filenames)
    print('Found', collection['count'], 'cutouts for', collection['shortname'])
    if collection['count'] == 0:
        return
    cutouts = []
    for idx, cutout_filename in enumerate(cutout_filenames):
        print(basename(cutout_filename), idx + 1, '/', collection['count'])
        cutout_context = render_one_cutout(
            collection['shortname'],
            cutout_filename,
            collection['display']['rgb']['bands'],
            collection['display']['catalog']['columns'],
        )
        cutouts.append(cutout_context)

    collection['cutouts'] = cutouts
    # make collection zip file rollup
    collection['archive_filename'] = shutil.make_archive(collection['shortname'], 'zip', join(ROOT, 'collections', collection['directory']))
    collection['archive_size'] = os.path.getsize(collection['archive_filename'])
    print('Made', collection['archive_filename'], 'size', collection['archive_size'], 'bytes')
    shutil.move(collection['archive_filename'], join(collection_dir, basename(collection['archive_filename'])))
    collection['archive_filename'] = basename(collection['archive_filename'])
    # copy over preview image for homepage display
    shutil.copy(join(ROOT, 'collections', collection['directory'], collection['preview_image']), join(collection_dir, collection['preview_image']))
    # render collection page with all contexts
    collection_t = ENV.get_template('collection.html')
    with open(join(collection_dir, 'index.html'), 'w') as f:
        f.write(collection_t.render(
            collection=collection,
            cutouts=cutouts,
        ))
    print("Wrote", join(collection_dir, "index.html"))
    return collection


def generate():
    # Collect collection specs
    collection_specs = glob.glob(join(ROOT, 'collections', '*.yaml'))
    print('Found', len(collection_specs), 'collection specifications')
    collections = []
    for collection_filename in collection_specs:
        with open(collection_filename) as f:
            collection = yaml.load(f)
        collections.append(generate_collection_page(collection))
    # For each collection
    #  - generate collection page
    #    - for each cutout
    #      - render fragment
    #      - generate quicklook image
    # - generate zip bundle 
    # - render homepage fragment
    # Build SASS
    compile_scss()
    # Collect static files
    build_static = join(BUILD_ROOT, 'static')
    if isdir(build_static):
        shutil.rmtree(build_static, ignore_errors=True)
    shutil.copytree(join(ROOT, 'static'), build_static)
    # Render homepage template
    index_t = ENV.get_template('index.html')
    with open(join(BUILD_ROOT, 'index.html'), 'w') as f:
        f.write(index_t.render(
            collections=collections
        ))

if __name__ == "__main__":
    generate()