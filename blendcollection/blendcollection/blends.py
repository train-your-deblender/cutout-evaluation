from functools import wraps
from collections import OrderedDict
from os.path import basename

import numpy as np
from astropy.io import fits
# from astropy.visualization import make_lupton_rgb
from astropy.visualization import LogStretch, ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize

__all__ = [
    'quick_rgb',
    'quick_mono',
    'validate',
    'load',
    'load_blend_fits',
    'axes_optional',
    'Cube',
    'Blend'
]

def quick_mono(image, contrast=0.25):
    interval = ZScaleInterval(contrast=contrast)
    vmin, vmax = interval.get_limits(image)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch(), clip=True)
    # image = image.copy()
    # image[~np.isfinite(image)] = np.nanmedian(image)
    return norm(image)

def quick_rgb(image_red, image_green, image_blue, contrast=0.25):
    # Determine limits for each channel
    interval = ZScaleInterval(contrast=contrast)
    red_min, red_max = interval.get_limits(image_red)
    green_min, green_max = interval.get_limits(image_green)
    blue_min, blue_max = interval.get_limits(image_blue)
    # Determine overall limits
    vmin, vmax = min(red_min, green_min, blue_min), max(red_max, green_max, blue_max)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch(), clip=True)
    # Make destination array
    rgbim = np.zeros(image_red.shape + (3,), dtype=np.uint8)
    for idx, im in enumerate((image_red, image_green, image_blue)):
        rescaled = (norm(im) * 255).astype(np.uint8)
        rgbim[:,:,idx] = rescaled
    return rgbim

def validate(hdulist):
    # use Astropy to validate FITS standard compliance
    hdulist.verify(option='exception')

    required_extensions = ('PRIMARY', 'BLENDED', 'TRUTH', 'BANDS', 'CATALOG')
    ext_error_message = "Blend cutouts must have at least these extensions as the first four in the file: {}".format(required_extensions)
    for idx, extname in enumerate(required_extensions):
        try:
            if hdulist.index_of(extname) != idx:
                raise ValueError(ext_error_message)
        except KeyError:
            raise ValueError(ext_error_message)

    if hdulist['BLENDED'].data.shape[0] != hdulist['TRUTH'].data.shape[0]:
        raise ValueError("BLENDED and TRUTH cubes must contain the same number of planes")

    if len(hdulist['BANDS'].data) != hdulist['BLENDED'].data.shape[0]:
        raise ValueError("BANDS must contain one row per plane in blend cube")

    cols = hdulist['BANDS'].columns
    if cols[0].name != 'name' or cols[1].name != 'wavelength':
        raise ValueError("BANDS table must have 'name' and 'wavelength' as the first two columns")

    if 'BLEND-ID' not in hdulist['PRIMARY'].header:
        raise ValueError("PRIMARY header must contain 'BLEND-ID' to uniquely identify this blend")

    if 'NAME' not in hdulist['PRIMARY'].header:
        raise ValueError("PRIMARY header must contain 'NAME' to provide a human-friendly name")

    return hdulist

def load(file_path_or_hdulist):
    return Blend(file_path_or_hdulist)

def load_blend_fits(file_path_or_hdulist):
    if not isinstance(file_path_or_hdulist, fits.HDUList):
        hdulist = fits.open(file_path_or_hdulist)
    else:
        hdulist = file_path_or_hdulist

    return validate(hdulist)

def axes_optional(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if 'ax' not in kwargs or kwargs['ax'] is None:
            from matplotlib import pyplot as plt
            kwargs['ax'] = plt.gca()
        return func(*args, **kwargs)
    return inner

class Cube(object):
    def __init__(self, datacube, bands):
        self._datacube = datacube
        self.bands = bands
    def __getitem__(self, key):
        if not key in self.bands:
            raise ValueError("Unknown band (these bands are present: {})".format(', '.join(self.bands.keys())))
        names = tuple(self.bands.keys())
        plane_idx = names.index(key)
        return self._datacube[plane_idx]
    def __iter__(self):
        for i in range(self._datacube.shape[0]):
            yield self._datacube[i]

    def __repr__(self):
        return '<Cube: {dimensions} (bands: {bands})>'.format(
            dimensions='x'.join(map(str, self._datacube.shape)),
            bands=', '.join(self.bands.keys())
        )

class Blend(object):
    _IMSHOW_KWARGS = {
        'origin': 'lower',
        'interpolation': 'nearest',
    }
    def __init__(self, file_path_or_object):
        self._hdulist = load_blend_fits(file_path_or_object)
        self.catalog = self._hdulist['CATALOG'].data
        self.bands = OrderedDict(zip(
            self._hdulist['BANDS'].data['name'],
            self._hdulist['BANDS'].data['wavelength']
        ))
        
        self.blended = Cube(self._hdulist['BLENDED'].data, self.bands)
        self.truth = Cube(self._hdulist['TRUTH'].data, self.bands)

    @property
    def filename(self):
        return basename(self._hdulist.filename())

    @property
    def id(self):
        return self._hdulist['PRIMARY'].header['BLEND-ID']

    @property
    def name(self):
        return self._hdulist['PRIMARY'].header['NAME']

    @property
    def width_arcsec(self):
        return self._hdulist['PRIMARY'].header['NAME']

    @property
    def height_arcsec(self):
        return self._hdulist['PRIMARY'].header['NAME']

    @property
    def extent_arcsec(self):
        return self._hdulist['PRIMARY'].header['NAME']

    def display(self, overlay=True, red='i', green='r', blue='g', fig=None):
        if fig is None:
            from matplotlib import pyplot as plt
            fig = plt.gcf()
        blended_ax = fig.add_subplot(1, 2, 1)
        truth_ax = fig.add_subplot(1, 2, 2)
        self.display_blended(overlay=overlay, red=red, green=green, blue=blue, ax=blended_ax)
        self.display_truth(overlay=overlay, red=red, green=green, blue=blue, ax=truth_ax)
        return fig

    @axes_optional
    def display_blended(self, overlay=True, red='i', green='r', blue='g', ax=None):
        image_r = self.blended[red]
        image_g = self.blended[green]
        image_b = self.blended[blue]
        rgb_image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        
        img = ax.imshow(rgb_image, **Blend._IMSHOW_KWARGS)
        if overlay:
            pass

    @axes_optional
    def display_truth(self, overlay=True, red='i', green='r', blue='g', ax=None):
        image_r = self.truth[red]
        image_g = self.truth[green]
        image_b = self.truth[blue]
        rgb_image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        
        img = ax.imshow(rgb_image, **Blend._IMSHOW_KWARGS)
        if overlay:
            pass