from __future__ import print_function, division
import glob
from collections import defaultdict
from astropy.io import fits
import astropy.modeling.models as models
import numpy as np
from scipy import stats
import astropy.stats
import astropy.table
from astropy import wcs
from astropy import convolution
import img_scale
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
from os.path import exists, isdir, basename, split

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

GALAXY_A_PATH = os.environ['PYSYN_CDBS'] + '/grid/bc95/templates/bc95_c_10E9.fits'
GALAXY_B_PATH = os.environ['PYSYN_CDBS'] + '/grid/bc95/templates/bc95_c_10E7.fits'
STAR_PATH = os.environ['PYSYN_CDBS'] + '/grid/pickles/dat_uvk/pickles_uk_9.fits'
SOLAR_PATH = os.environ['PYSYN_CDBS']  + '/calspec/sun_reference_001.fits'

LSST_FWHM_ARCSEC = 0.7
LSST_PIXEL_SCALE = 0.2
SDSS_BANDS = ('u', 'g', 'r', 'i', 'z')

DEFAULT_OVERSAMPLING = 2

def create_photometry(spectrum, redshift, abmag, bands):
    n_bands = len(bands)
    effective_wavelengths = np.zeros(n_bands)
    fluxes = np.zeros(n_bands)
    counts = np.zeros(n_bands)
    zp = 25. * np.ones(n_bands)
    exptime = 100. * np.ones(n_bands)
    spec = pysynphot.FileSpectrum(spectrum)
    spec = spec.redshift(redshift)
    spec = spec.renorm(abmag, 'abmag', pysynphot.ObsBandpass('sdss,r'))

    for idx, (bandpass, zeropoint, exposure_time) in enumerate(zip(bands,zp,exptime)):
        obs = pysynphot.Observation(spec, pysynphot.ObsBandpass('sdss,'+bandpass))
        effective_wavelengths[idx] = obs.efflam()  # efflam = ? effective wavelength
        logcts = (zeropoint - obs.effstim('abmag')) / 2.5  # effstim ?
        fluxes[idx] = 10. ** logcts
        counts[idx] = exposure_time * 10. ** logcts

    return effective_wavelengths, counts

# RGB image with flexible scaling of color of the brightest pixels
def makeRGB(rimg,gimg,bimg,minsigma=1.,maxpercentile=99.9,color_scaling=None,sigmaclip=3,iters=20,nonlinear=8.):
    ''' minsigma -- black level is set this many clipped-sigma below sky
       maxpercentile -- white level is set at this percentile of the pixel levels
       color_scaling -- list or array: maximum value is divided by this;
                        so if you want the brightest pixels to be reddish, try, say 1.0,0.2, 0.1
       sigmaclip -- clipping threshold for iterative sky determination
       iters -- number of iterations for iterative sky determination
    '''
    bands = ['r','g','b']
    # Color scaling
    if color_scaling == None:
        cfactor = {'r':1.,'g':1.0,'b':1.0}
    else:
        cfactor = {}
        for i,b in enumerate(bands):
            cfactor[b] = color_scaling[i]
    images = {'r':rimg, 'g':gimg, 'b':bimg}
    rescaled_img = {}
    for b in ['r','g','b']:
        mean, median, stddev = astropy.stats.sigma_clipped_stats(images[b],sigma=sigmaclip,iters=iters)
        imin = median - minsigma * stddev
        imax = np.percentile(images[b], maxpercentile)
        rescaled_img[b] = img_scale.asinh(images[b], scale_min=imin, scale_max=imax/cfactor[b], non_linear=nonlinear)
    rgbimg = np.zeros((rescaled_img['r'].shape[0],rescaled_img['r'].shape[1],3),dtype=np.float64)
    rgbimg[:,:,0] = rescaled_img['r']
    rgbimg[:,:,1] = rescaled_img['g']
    rgbimg[:,:,2] = rescaled_img['b']
    return rgbimg

class ChromaticGaussianPSF(object):
    def __init__(self, fwhm_arcsec, reference_wavelength,
                 array_size=25):
        self.fwhm_arcsec = fwhm_arcsec
        self.reference_wavelength = reference_wavelength
        self.array_size = array_size

    def get_kernel(self, wavelength, pixel_scale_arcsec_per_px):
        sigma = (self.fwhm_arcsec / pixel_scale_arcsec_per_px) / (2 * np.sqrt(2 * np.log(2)))
        return convolution.Gaussian2DKernel(
            sigma * wavelength / self.reference_wavelength,
            x_size=self.array_size,
            y_size=self.array_size
        )
    def get_model(self, wavelength, pixel_scale_arcsec_per_px):
        sigma = self.fwhm_arcsec / (2 * np.sqrt(2 * np.log(2)))
        return models.Gaussian2D(
            x_stddev=sigma * wavelength / self.reference_wavelength,
            y_stddev=sigma * wavelength / self.reference_wavelength,
        )
    def fits_for_bands(self, bands, pixel_scale_arcsec_per_px):
        obsbandpasses = [pysynphot.ObsBandpass('sdss,'+band) for band in bands]
        extensions = [fits.PrimaryHDU()]
        extensions[0].header['FWHM'] = (
            self.fwhm_arcsec,
            'FWHM in arcsec at reference wavelength'
        )
        extensions[0].header['PIXELSCL'] = (
            pixel_scale_arcsec_per_px,
            'pixel scale in arcseconds per pixel'
        )
        extensions[0].header['REFWAVE'] = (
            self.reference_wavelength / 10,
            'reference wavelength in nanometers'
        )

        for band, bpass in zip(bands, obsbandpasses):
            k = self.get_kernel(bpass.avgwave(), pixel_scale_arcsec_per_px)
            imghdu = fits.ImageHDU(k.array)
            imghdu.header['WAVELEN'] = (
                bpass.avgwave(),
                'wavelength at which PSF was simulated'
            )
            imghdu.header['FWHM'] = (
                self.fwhm_arcsec * bpass.avgwave() / self.reference_wavelength,
                'FWHM of PSF at this wavelength (arcsec)'
            )
            imghdu.header['EXTNAME'] = band.upper()
            extensions.append(imghdu)
        hdul = fits.HDUList(extensions)
        return hdul

class Catalog(object):
    def __init__(self, bands=SDSS_BANDS):
        def col(name, format='E', dim=None, unit=None):
            return {'name': name, 'format': format, 'dim': dim, 'unit': unit}
        self._column_specs = {x['name']: x for x in (
            col('ra'),
            col('dec'),
            col('x', unit='arcsec'),
            col('y', unit='arcsec'),
            col('x_image', unit='pixel'),
            col('y_image', unit='pixel'),
            col('model', format='20A'),
            col('abmag'),
            col('redshift'),
            col('spectrum', format='20A'),
            col('sed', format='{}E'.format(len(bands)), dim=(len(bands),), unit='counts'),
            col('sersic_n', format='I'),
            col('r_eff', unit='pixel'),
            col('ellipticity'),
            col('theta')
        )}
        self._column_names = tuple(self._column_specs.keys()) # [spec['name'] for spec in self._column_specs]
        self._columns = defaultdict(list)

    def add_row(self, **kwargs):
        if set(kwargs.keys()) != set(self._column_names):
            raise ValueError("All arguments are required: {}".format(', '.join(self._column_names)))

        for name in self._column_names:
            self._columns[name].append(kwargs[name])

    def to_bintable_hdu(self):
        fits_columns = []
        for name in self._column_names:
            fits_columns.append(fits.Column(
                array=self._columns[name],
                **self._column_specs[name]
            ))
        return fits.BinTableHDU.from_columns(fits_columns)

    def __getitem__(self, key):
        if key in self._columns:
            return self._columns[key]
        else:
            raise ValueError("Unknown column {} (should be one of {})".format(key, ', '.join(self._column_names)))

class Scene(object):
    def __init__(self, chromatic_psf, width_px, height_px,
                 pixel_scale_arcsec_per_px=LSST_PIXEL_SCALE, oversample=DEFAULT_OVERSAMPLING):
        self.chromatic_psf = chromatic_psf
        self.pixel_scale_arcsec_per_px = pixel_scale_arcsec_per_px
        self._pixel_scale_oversampled = self.pixel_scale_arcsec_per_px / oversample
        self.oversample = oversample
        self.width_px, self.height_px = width_px, height_px
        self.bands = SDSS_BANDS
        self.bands_to_indices = dict(zip(
            self.bands, range(len(self.bands))
        ))
        self._obsbandpasses = [pysynphot.ObsBandpass('sdss,{}'.format(band))
                               for band in self.bands]

        # from Harry:
        # A typical ground-based sky background might be ~ 22 mag per
        # square arsecond. I just arbitrarily picked a pixel scale of 0.05 arcsec/pixel.
        # LSST has 0.2 arc sec per pixel, so that would be a better choice, actually.
        SKY_ABMAG_PER_ARCSEC = 22.
        _, sky_counts = create_photometry(SOLAR_PATH, 0., SKY_ABMAG_PER_ARCSEC, self.bands)
        # sky counts per pixel in each band
        self.sky_counts = sky_counts * self.pixel_scale_arcsec_per_px * self.pixel_scale_arcsec_per_px
        
        self._models = []
        self._canvas = None
        self._dirty = True

        self._catalog = Catalog()
        
        # bogus row for sky model SED
        # from counts per pixel to total counts in image
        # sky_row = (0, 0, 0, 0, 0, 0, 0, 'sky')  # id, ra, dec, x_arcsec, y_arcsec, x_image, y_image, model
        # sky_row += (SKY_ABMAG_PER_ARCSEC, 0.0, basename(SOLAR_PATH))  # abmag, redshift, spectrum
        # sky_row += tuple(self.width_px * self.height_px * self.sky_counts)  # counts_{} for each band
        # sky_row += (0, 0, 0, 0)  # sersic_n, r_eff_px, ellipticity, theta
        # self._catalog.add_row(sky_row)
        #
        # construct bogus WCS to set angular scale
        bogus_wcs = wcs.WCS(naxis=2)
        bogus_wcs.wcs.crpix = [self.width_px // 2, self.height_px // 2]
        bogus_wcs.wcs.cdelt = [
            self.pixel_scale_arcsec_per_px / 60 / 60, 
            self.pixel_scale_arcsec_per_px / 60 / 60
        ]
        bogus_wcs.wcs.crval = [0, 0]
        bogus_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        self.wcs = bogus_wcs

    @property
    def width_arcsec(self):
        return self.width_px * self.pixel_scale_arcsec_per_px

    @property
    def height_arcsec(self):
        return self.height_px * self.pixel_scale_arcsec_per_px

    def pixels_to_arcsecs(self, x_px, y_px):
        return x_pixel * self.pixel_scale_arcsec_per_px, y_pixel * self.pixel_scale_arcsec_per_px

    def arcsecs_to_pixels(self, x_arcsec, y_arcsec):
        return (
            self.width_px // 2 + x_arcsec / self.pixel_scale_arcsec_per_px,
            self.height_px // 2 + y_arcsec / self.pixel_scale_arcsec_per_px
        )

    def grid(self, oversample=None):
        if oversample is None:
            oversample = self.oversample
        x_min, x_max, y_min, y_max = self.extent
        x_coords = np.linspace(x_min, x_max, self.width_px * oversample)
        y_coords = np.linspace(y_min, y_max, self.height_px * oversample)
        return np.meshgrid(x_coords, y_coords)

    @property
    def extent(self):
        return (-self.width_arcsec / 2, self.width_arcsec / 2, -self.height_arcsec / 2, self.height_arcsec / 2)

    def _downsample_planes(self, canvas, downsample_factor):
        assert len(canvas.shape) == 3
        new_shape = (
            canvas.shape[0],
            canvas.shape[1] // downsample_factor,
            downsample_factor,
            canvas.shape[2] // downsample_factor,
            downsample_factor
        )
        downsampled_canvas = canvas.reshape(new_shape).sum(axis=(2, 4))
        return downsampled_canvas

    def render(self):
        canvas = np.zeros((
            len(self.bands),
            self.height_px * self.oversample,
            self.width_px * self.oversample
        ))
        y_coords_arcsec, x_coords_arcsec = self.grid()
        
        for model, sed in zip(self._models, self._catalog['sed']):
            for band_idx, band_counts in enumerate(sed):
                print("{} model counts = {}".format(self.bands[band_idx], band_counts))
                # TODO remove
                temp = model(y_coords_arcsec, x_coords_arcsec)
                temp /= np.sum(temp)
                temp *= band_counts
                assert np.allclose(np.sum(temp), band_counts)
                canvas[band_idx] += temp

        self._truth_canvas = canvas.copy()

        for idx, (band, bpass) in enumerate(zip(self.bands, self._obsbandpasses)):
            # canvas is in oversampled pixels, sky_counts in counts / pixel
            # so we divide the sky counts into the sky counts per subpixel
            canvas[idx] += self.sky_counts[idx] / self.oversample / self.oversample
            npix = len(canvas[idx])
            print("{} sky level = {}".format(band, np.sum(canvas[idx])))

            # apply PSF with convolution
            psf = self.chromatic_psf.get_kernel(bpass.avgwave(), self.pixel_scale_arcsec_per_px)
            canvas[idx] = convolution.convolve(canvas[idx], psf)

        canvas_downsampled = self._downsample_planes(canvas, self.oversample)
        # apply noise
        for idx, band in enumerate(self.bands):
            canvas_downsampled[idx] = stats.norm(
                canvas_downsampled[idx],
                np.sqrt(canvas_downsampled[idx])
            ).rvs(
                canvas_downsampled[idx].shape
            )
        
        self._dirty = False
        self._canvas = canvas_downsampled
        return self._canvas.copy(), self._truth_canvas.copy()

    def to_hdu_list(self, clobber=False):
        primary_hdu = fits.PrimaryHDU()
        hdr = primary_hdu.header
        hdr['PIXELSCL'] = (self.pixel_scale_arcsec_per_px, 'pixel scale in arcseconds per pixel')
        # for band, bpass in zip(self.bands, self._obsbandpasses):
        #     hdr['WAVE_{}'.format(band.upper())] = (
        #         bpass.avgwave(),
        #         'average wavelength for {} bandpass'.format(band)
        #     )
        #
        # extensions = [primary_hdu,]
        # for idx, band in enumerate(self.bands):
        #     imghdu = fits.ImageHDU(self.canvas[idx], header=self.wcs.to_header())
        #     imghdu.header['SKYCTS'] = (self.sky_counts[idx], 'counts from sky background per pixel')
        #     imghdu.header['EXTNAME'] = band.upper()
        #     extensions.append(imghdu)

        # 'BLENDED', 'TRUTH', 'BANDS', 'CATALOG'
        blend_hdu = fits.ImageHDU(self.canvas)
        blend_hdu.header['EXTNAME'] = 'BLENDED'

        truth_hdu = fits.ImageHDU(self.truth_canvas)
        truth_hdu.header['EXTNAME'] = 'TRUTH'
        
        wavelengths = []
        for bp in self._obsbandpasses:
            wavelengths.append(bp.avgwave() / 10.)
        
        bands_hdu = fits.BinTableHDU.from_columns([
            fits.Column(
                array=self.bands,
                format='{}A'.format(max(map(len, self.bands))),
                name='name',
            ),
            fits.Column(
                array=wavelengths,
                format='E',
                unit='nanometer',
                name='wavelength',
            ),
        ])
        bands_hdu.header['EXTNAME'] = 'BANDS'

        catalog_hdu = self._catalog.to_bintable_hdu()
        catalog_hdu.header['EXTNAME'] = 'CATALOG'

        hdu_list = fits.HDUList([
            primary_hdu,
            blend_hdu,
            truth_hdu,
            bands_hdu,
            catalog_hdu,
        ])
        return hdu_list

    def write(self, filename, clobber=False):
        hdu_list = self.to_hdu_list()
        hdu_list.writeto(filename, clobber=clobber)

    def _add_to_catalog(self, x_arcsec, y_arcsec, model_name, abmag, redshift, spectrum,
                        counts_for_bands,
                        sersic_n=0, r_eff=0, ellipticity=0, theta=0):
        x_image_px, y_image_px = self.arcsecs_to_pixels(x_arcsec, y_arcsec)
        self._catalog.add_row(
            ra=np.mod(360. + x_arcsec / 60 / 60, 360.),
            dec=y_arcsec / 60 / 60,
            x=x_arcsec,
            y=y_arcsec,
            x_image=x_image_px,
            y_image=x_image_px,
            model=model_name,
            abmag=abmag,
            redshift=redshift,
            spectrum=spectrum,
            sed=counts_for_bands,
            sersic_n=sersic_n,
            r_eff=r_eff,
            ellipticity=ellipticity,
            theta=theta,
        )
        
    @property
    def canvas(self):
        if self._canvas is not None and not self._dirty:
            return self._canvas.copy()
        else:
            return self.render()[0]

    @property
    def truth_canvas(self):
        if self._truth_canvas is not None and not self._dirty:
            return self._truth_canvas.copy()
        else:
            return self.render()[1]

    def display(self, band=None, ax=None, **kwargs):
        if band is None:
            band = self.bands[0]
        if ax is None:
            ax = plt.gca()
        plane = self.canvas[self.bands_to_indices[band]]
        new_kwargs = {
            'origin': 'lower',
            'extent': self.extent,
            'cmap': 'gray',
        }
        new_kwargs.update(kwargs)
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')
        ax.set_title(band)
        return ax.imshow(plane, **new_kwargs)

    def display_rgb(self, red='i', green='r', blue='g', ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        new_kwargs = {
            'origin': 'lower',
            'extent': self.extent,
        }
        new_kwargs.update(kwargs)
        rgbimg = makeRGB(
            self.canvas[self.bands_to_indices[red]],
            self.canvas[self.bands_to_indices[green]],
            self.canvas[self.bands_to_indices[blue]],
            color_scaling=[1.0, 0.1, 0.05]
        )
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')
        ax.set_title('Composite (R={}, G={}, B={})'.format(red, green, blue))
        return ax.imshow(rgbimg, **new_kwargs)

    def display_as_panels(self, fig=None, **kwargs):
        if fig is None:
            fig = plt.gcf()
        new_kwargs = {
            'origin': 'lower',
            'extent': self.extent,
            'cmap': 'gray',
        }
        new_kwargs.update(kwargs)
        for idx, band in enumerate(self.bands):
            ax = fig.add_subplot(1, len(self.bands), idx + 1)
            ax.set_title(band)
            ax.imshow(self.canvas[idx], **new_kwargs)
        return fig

    def add_sersic_galaxy(self, spectrum, redshift, abmag,
                          r_eff_arcsec, sersic_index=1, ellipticity=0.5, theta=0.,
                          x_arcsec=0., y_arcsec=0.):
        model = models.Sersic2D(
            amplitude=1.0,
            r_eff=r_eff_arcsec,
            n=sersic_index,
            x_0=x_arcsec,
            y_0=y_arcsec,
            ellip=ellipticity,
            theta=theta,
        )
        wavelengths, counts_for_bands = create_photometry(
            spectrum,
            redshift,
            abmag,
            self.bands
        )

        self._models.append(model)

        self._add_to_catalog(
            x_arcsec, y_arcsec, 'sersic', abmag, redshift, basename(spectrum),
            counts_for_bands,
            sersic_n=sersic_index,
            r_eff=r_eff_arcsec / self.pixel_scale_arcsec_per_px,
            ellipticity=ellipticity,
            theta=theta,
        )
        self._dirty = True

    def add_star(self, spectrum, abmag, x_arcsec=0., y_arcsec=0.):
        wavelengths, counts_for_bands = create_photometry(
            spectrum,
            redshift=0.0,
            abmag=abmag,
            bands=self.bands
        )

        # Model a star as a very tight gaussian (~1 subpixel)
        self._models.append(models.Gaussian2D(
            x_mean=x_arcsec,
            y_mean=y_arcsec,
            x_stddev=self._pixel_scale_oversampled,
            y_stddev=self._pixel_scale_oversampled,
        ))
            
        self._add_to_catalog(x_arcsec, y_arcsec, 'gaus2d',
                             abmag, 0.0, basename(spectrum),
                             counts_for_bands)
        self._dirty = True
