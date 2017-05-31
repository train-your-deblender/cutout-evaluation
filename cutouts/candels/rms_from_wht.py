#!/usr/bin/env python
from astropy.io import fits
from os.path import exists
import numpy as np

def rms_from_wht(filename):
    assert '_wht.fits' in filename
    outfile = filename.replace('_wht.fits', '_rms.fits')
    if exists(outfile):
        return
    hdul = fits.open(filename)
    assert len(hdul) == 1
    # variance = rms^2
    # 1/variance = 1/(rms^2)
    # wht = 1/(rms^2)
    # 1/wht = rms^2
    # sqrt(1/wht) = rms
    hdul[0].data = np.sqrt(1. / hdul[0].data)
    mask = np.isfinite(hdul[0].data)
    hdul[0].data[~mask] = 0.0
    hdul.writeto(outfile)
    print(outfile)

if __name__ == "__main__":
    import sys
    for filename in sys.argv[1:]:
        assert exists(filename)
        print(filename)
        rms_from_wht(filename)
