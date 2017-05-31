#!/bin/bash
set -eo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p "$DIR/cosmos/originals" "$DIR/egs/originals" "$DIR/uds/originals" "$DIR/goods-n/originals" "$DIR/goods-s/originals"

for fn in /astro/candels1/www/data/*/mosaics/current/60mas/*.fits;  do
#          /astro/candels1/www/data/goodss/mosaics/current/goods_s_acs_v3/*.fits \
#          /astro/candels1/www/data/goodss/mosaics/current/goods_s_all_combined_v1.0/60mas/*.fits; do
    echo "$fn"
    if [[ $fn == *"allfilters"* ]]; then
        continue
    fi
    if [[ $fn == *"no_udf"* ]]; then
        continue
    fi
    if [[ $fn == *"_presm4_"* ]]; then
        continue  # ignore GOODS-S pre-SM4 duplicate mosaic bands
    fi
    if [[ $fn == *"_ers_"* ]]; then
        continue  # ignore GOODS-S UDF duplicate mosaic bands
    fi
    if [[ $fn == *"cos_"* ]]; then
        field="cosmos"
    elif [[ $fn == *"goodsn_"* ]]; then
        field="goods-n"
    elif [[ $fn == *"egs_"* ]]; then
        field="egs"
    elif [[ ( $fn == *"gs_"* ) || ( $fn == *"goodss_"* ) ]]; then
        field="goods-s"
    elif [[ $fn == *"uds_"* ]]; then
        field="uds"
    else
        echo "what field is this? $fn"
        exit 1
    fi
    destpath="$DIR/$field/originals/$(basename $fn)"
    destpath=${destpath/_60mas/_060mas}

    if [ ! -L "$destpath" ]; then
        ln -vs "$fn" "$destpath"
    fi
done

# the new GOODS-S CANDELS images have _wht.fits which in one of the Readmes
# are apparently weight or inverse-variance maps,
# so make RMS maps from them
# > Images are <b>*_drz.fits</b> (e/s), <b>*_wht.fits</b>
# > (inverse variance), and <b>*_rms.fits</b>.
# - /astro/candels1/www/data/goodss/mosaics/current/Readme.html


# python "$DIR/rms_from_wht.py" "$DIR"/goods-s/originals/gs_all_candels*_wht.fits

# On 2/19/16, 11:15 AM, "Harry Ferguson" <ferguson@stsci.edu> wrote:
#
#
# > The box repository for the CANDELS catalogs is at https://stsci.box.com/s/5waibanyznc9od28p8xtt6a3t8vatr62
# > The easiest files to work with are the .hdf5 files in the merged_catalogs directory. You can read using, for example:
# >
# >     from astropy.table import Table
# >     gds = Table.read(‘gds.hdf5’)
# > The README files describing the columns are in the directories for the
# individual
# fields. For the photometry, the most complete README file is in the UDS
# subdirectory.
# >
# > -Harry
#
# These are copied to <field>_merged_catalog.fits in the individual field dirs.
