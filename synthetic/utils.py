import lsst.afw.table
import lsst.afw.image
import lsst.afw.math
import lsst.meas.algorithms
import lsst.meas.base
import lsst.meas.deblender
import os

# Hack to import PySynphot
from os.path import exists, isdir, basename

_pysynphot_ref_file_roots = (
    '/Users/hcferguson/data/cdbs',
    '/Users/jlong/cdbs',
    '/grp/hst/cdbs'
)
if not os.environ.get('PYSYN_CDBS', False):
    for location in _pysynphot_ref_file_roots:
        if isdir(location):
            os.environ['PYSYN_CDBS'] = location

assert os.environ.get('PYSYN_CDBS', False)
import pysynphot


import numpy as np
from astropy.io import fits
import astropy.modeling.models as models
from scipy import stats

STELLAR_FLUX_BANDS = dict([
    ('u', 4906.8876155282305),
    ('g', 12374.179158464998),
    ('r', 9997.8779664504054),
    ('i', 8443.9622555133319),
    ('z', 7255.7890343376985)
])

def deblend_run(image_array, variance_array, psf_array):
    schema = lsst.afw.table.SourceTable.makeMinimalSchema()
    detect = lsst.meas.algorithms.SourceDetectionTask(schema=schema)
    deblend = lsst.meas.deblender.SourceDeblendTask(schema=schema)
    measure = lsst.meas.base.SingleFrameMeasurementTask(schema=schema)

    image = lsst.afw.image.ImageF(image_array.astype(np.float32))
    variance = lsst.afw.image.ImageF(variance_array.astype(np.float32))
    masked_image = lsst.afw.image.MaskedImageF(image, None, variance)

    psf_image = lsst.afw.image.ImageD(psf_array.astype(np.float64))
    psf_kernel = lsst.afw.math.FixedKernel(psf_image)
    psf = lsst.meas.algorithms.KernelPsf(psf_kernel)

    exposure = lsst.afw.image.ExposureF(masked_image)
    exposure.setPsf(psf)

    table = lsst.afw.table.SourceTable.make(schema)  # this is really just a factory for records, not a table
    detect_result = detect.run(table, exposure)
    catalog = detect_result.sources   # this is the actual catalog, but most of it's still empty
    deblend.run(exposure, catalog, exposure.getPsf())
    measure.run(catalog, exposure)
    #catalog.writeFits('./temp.fits')
    #t = Table.read('./detections.fits')
    #os.remove('./temp.fits')
    astropy_table = lsst.afw.table._syntax.BaseCatalog_asAstropy(catalog)
    return astropy_table

def two_gaussian_image(kernel_fwhm_arcsec,
                       separation_arcsec,
                       band='r',
                       pixel_scale_arcsec_per_px=0.06):
    kernel_fwhm_px = kernel_fwhm_arcsec / pixel_scale_arcsec_per_px
    kernel_sigma_arcsec = kernel_fwhm_arcsec / (2 * np.sqrt(2 * np.log(2)))
    kernel_sigma_px = kernel_fwhm_px / (2 * np.sqrt(2 * np.log(2)))
    width_px = height_px = int(max(4 * separation_arcsec / pixel_scale_arcsec_per_px, 100))
    width_arcsec = height_arcsec = width_px * pixel_scale_arcsec_per_px  # int() might have rounded things, work back to arcsec

    image = np.zeros((height_px, width_px), dtype=np.float32)
    x, y = np.meshgrid(
        np.linspace(-width_arcsec/2, width_arcsec/2, width_px),
        np.linspace(-height_arcsec/2, height_arcsec/2, height_px)
    )
    kernel_mod = models.Gaussian2D(
        amplitude=1.0,
        x_stddev=kernel_sigma_arcsec,
        y_stddev=kernel_sigma_arcsec,
        x_mean=0,
        y_mean=0
    )
    kernel = kernel_mod(x, y)
    star1 = models.Gaussian2D(
        amplitude=STELLAR_FLUX_BANDS[band],
        x_stddev=kernel_sigma_arcsec,
        y_stddev=kernel_sigma_arcsec,
        x_mean=-separation_arcsec/2,
        y_mean=0
    )
    star2 = models.Gaussian2D(
        amplitude=STELLAR_FLUX_BANDS[band],
        x_stddev=kernel_sigma_arcsec,
        y_stddev=kernel_sigma_arcsec,
        x_mean=separation_arcsec/2,
        y_mean=0
    )
    image += star1(x, y)
    image += star2(x, y)
    noisy_image = stats.norm(image, np.sqrt(image)).rvs(image.shape)
    return kernel, noisy_image