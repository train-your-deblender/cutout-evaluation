import argparse
from pipeline import run_candels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Positional
    parser.add_argument("field_name",
                        help="Name of the survey field being processed")

    # Flags
    parser.add_argument("-d", "--detection-image", required=True,
                        help="Filename of original image whose seeing-sim version will be used to detect sources")
    parser.add_argument("-s", "--pixel-scale", type=float, required=True,
                        help="Pixel scale of originals in arcseconds per pixel")
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
    # python run_candels.py --detection-image ./data/egs_all_acs_wfc_f606w_060mas_v1.1_drz.fits \
    #    --pixel-scale 0.06 --kernel-fwhm 0.7 --tile-size 1024 \
    #    egs
    args = parser.parse_args()
    kernel_fwhm_arcsec = args.kernel_fwhm
    use_phosim = args.phosim
    tile_size = args.tile_size
    pad_tiles = args.pad_tiles
    detection_image = args.detection_image
    field_name = args.field_name
    pixel_scale = args.pixel_scale
    downsample_factor = args.downsample_factor
    pad_cutouts = args.pad_cutouts
    run_candels(field_name, detection_image, pixel_scale, downsample_factor,
                kernel_fwhm_arcsec, use_phosim, tile_size, pad_tiles, pad_cutouts)

    print("Complete!")