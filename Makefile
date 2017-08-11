CANDELS_OPTIONS=--downsample-factor 4 --pad-tiles 128 --pad-cutouts 30 --pixel-scale 0.06 --kernel-fwhm 0.7 --tile-size 1024

.PHONY: clean-cutouts-candels cutouts-candels cutouts clean clean-website pre-website website

all: cutouts pre-website website

clean: clean-cutouts

clean-cutouts-candels:
	rm -rf ./cutouts/candels/cosmos/products/*
	rm -rf ./cutouts/candels/cosmos/scratch/*
	rm -rf ./cutouts/candels/egs/products/*
	rm -rf ./cutouts/candels/egs/scratch/*
	rm -rf ./cutouts/candels/goods-n/products/*
	rm -rf ./cutouts/candels/goods-n/scratch/*
	rm -rf ./cutouts/candels/goods-s/products/*
	rm -rf ./cutouts/candels/goods-s/scratch/*
	rm -rf ./cutouts/candels/uds/products/*
	rm -rf ./cutouts/candels/uds/scratch/*

clean-website:
	rm -rf ./website/_build/*

cutouts-candels:
	bash ./cutouts/candels/collect_data.sh
	python ./cutouts/run_candels.py \
		--detection-image ./cutouts/candels/cosmos/originals/cos_2epoch_acs_f606w_060mas_v1.0_drz.fits \
		$(CANDELS_OPTIONS) \
		cosmos
	python ./cutouts/run_candels.py \
		--detection-image ./cutouts/candels/egs/originals/egs_all_acs_wfc_f606w_060mas_v1.1_drz.fits \
		$(CANDELS_OPTIONS) \
		egs
	python ./cutouts/run_candels.py \
		--detection-image ./cutouts/candels/goods-n/originals/goodsn_all_acs_wfc_f606w_060mas_v2.0_drz.fits \
		$(CANDELS_OPTIONS) \
		goods-n
	python ./cutouts/run_candels.py \
		--detection-image ./cutouts/candels/goods-s/originals/goodss_all_acs_wfc_f606w_060mas_v1.5_drz.fits \
		$(CANDELS_OPTIONS) \
		goods-s
	python ./cutouts/run_candels.py \
		--detection-image ./cutouts/candels/uds/originals/uds_all_acs_f606w_060mas_v1.0_drz.fits \
		$(CANDELS_OPTIONS) \
		uds

cutouts: cutouts-candels

website:
	python website/generator.py

deploy-website:
	rsync -av ./website/_build/ /astro/ferguson1/ferguson/deblending_web/

pre-website: website-collection-candels

website-collection-candels: cutouts-candels
	ln -sfv ./cutouts/candels/cosmos/products/ ./website/collections/cosmos
	ln -sfv ./cutouts/candels/egs/products/ ./website/collections/egs
	ln -sfv ./cutouts/candels/goods-n/products/ ./website/collections/goods-n
	ln -sfv ./cutouts/candels/goods-s/products/ ./website/collections/goods-s
	ln -sfv ./cutouts/candels/uds/products/ ./website/collections/uds
