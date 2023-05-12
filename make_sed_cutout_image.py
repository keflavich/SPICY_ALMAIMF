import numpy as np
import re
import os
import pylab as pl
import reproject
import requests
import io

from astroquery.magpis import Magpis
from astroquery.herschel.higal import HiGal

from astropy import table
from astropy import units as u, coordinates
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy import wcs
from astropy.wcs import WCS
from astropy import visualization

from spectral_cube import SpectralCube

import matplotlib as mpl

# load up ALMA-IMF metadata
import sys
sys.path.append('/orange/adamginsburg/ALMA_IMF/reduction/analysis/')
from spectralindex import prefixes

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = False
mpl.rcParams['ytick.right'] = False

def getimg(*args, **kwargs):
    try:
        return Magpis.get_images(*args, **kwargs)
    except:
        return

wlmap = {'gps6': 6*u.cm,
         'gps6epoch2': 6*u.cm,
         'gps6epoch3': 6*u.cm,
         'gps6epoch4': 6*u.cm,
         'gps20': 20*u.cm,
         'gps20new': 20*u.cm,
         'gps90': 90*u.cm,
         'gpsmsx': 8*u.um,
         'gpsmsx2': 8*u.um,
         'gpsglimpse36': 3.6*u.um,
         'gpsglimpse45': 4.5*u.um,
         'gpsglimpse58': 5.8*u.um,
         'gpsglimpse80': 8.0*u.um,
         'mipsgal': 24*u.um,
         'atlasgal': 0.870*u.mm,
         'bolocam': 1.1*u.mm,
         'mgps': 3.*u.mm,
         'HiGal70': 70*u.um,
         'HiGal160': 160*u.um,
         'HiGal250': 250*u.um,
         'HiGal350': 350*u.um,
         'HiGal500': 500*u.um,
         'ALMA_finaliter_prefix_b3': 3*u.mm,
         'ALMA_finaliter_prefix_b6': 1*u.mm,
         }

survey_titles = {'gps6': 'MAGPIS Epoch 1',
                 'gps6epoch2': 'MAGPIS Epoch 2',
                 'gps6epoch3': 'MAGPIS Epoch 3',
                 'gps6epoch4': 'MAGPIS Epoch 4',
                 'gps20': "MAGPIS",
                 'gps20new': "MAGPIS New",
                 'gps90': "VLA",
                 'gpsmsx': "MSX",
                 'gpsmsx2': "MSX 2",
                 'gpsglimpse36': "IRAC",
                 'gpsglimpse45': "IRAC",
                 'gpsglimpse58': "IRAC",
                 'gpsglimpse80': "IRAC",
                 'mipsgal': "MIPSGAL",
                 'atlasgal': "ATLASGAL",
                 'bolocam': "BGPS",
                 'mgps': "$\mathbf{MGPS-90}$",
                 'HiGal70':  "Hi-Gal",
                 'HiGal160': "Hi-Gal",
                 'HiGal250': "Hi-Gal",
                 'HiGal350': "Hi-Gal",
                 'HiGal500': "Hi-Gal",
         'ALMA_finaliter_prefix_b3': 'ALMA',
         'ALMA_finaliter_prefix_b6': 'ALMA',
                 }

prefixes['W43MM1'] = dict(
    finaliter_prefix_b3="W43-MM1/B3/cleanest/W43-MM1_B3_uid___A001_X1296_X1af_continuum_merged_12M_robust0_selfcal4_finaliter",
    finaliter_prefix_b6="W43-MM2/B6/cleanest/W43-MM2_B6_uid___A001_X1296_X113_continuum_merged_12M_robust0_selfcal5_finaliter",)
alma_basepath='/orange/adamginsburg/ALMA_IMF/2017.1.01355.L/RestructuredImagingResults'

def get_wcs(cube):
    ww = cube.wcs.celestial
    ww._naxis = cube.shape[1:]
    return ww

alma_wcses = {key:
              {key1: get_wcs(SpectralCube.read(f'{alma_basepath}/{prefixes[key][key1]}.image.tt0.fits',
                                               format='fits', use_dask=False))
               for key1 in prefixes[key]}
              for key in prefixes}


glimpses=g_subsets=['glimpsei_0_6', 'glimpseii_0_6', 'glimpse3d_0_6',
                    'glimpse360_0_6', 'glimpse_cygx_0_6',
                    'glimpse_deepglimpse_0_6', 'glimpse_smog_0_6',
                    'glimpse_velacar_0_6', 'mipsgal_images']


def get_spitzer_data(crd, size):
    files = {}
    for spitzertbl in glimpses:
        if 'glimpse' in spitzertbl:
            url = f"https://irsa.ipac.caltech.edu/IBE?table=spitzer.{spitzertbl}&POS={crd.ra.deg},{crd.dec.deg}&ct=csv&mcen&where=fname+like+'%.fits'"
        else:
            url = f"https://irsa.ipac.caltech.edu/IBE?table=spitzer.{spitzertbl}&POS={crd.ra.deg},{crd.dec.deg}&ct=csv&where=fname+like+'%.fits'"
        response = requests.get(url)
        response.raise_for_status()
        tbl = table.Table.read(io.BytesIO(response.content), format='ascii.csv')

        if (len(tbl) >= 4) and 'I1' not in files:
            fnames = tbl['fname']

            for fname in fnames:
                irsa_url = f"https://irsa.ipac.caltech.edu/ibe/data/spitzer/{spitzertbl}/{fname}?center={crd.ra.deg},{crd.dec.deg}&size={size.to(u.arcmin).value}arcmin"

                key = re.search("I[1-4]", fname).group()

                fh = fits.open(irsa_url)
                files[key] = fh
        elif 'mipsgal' in spitzertbl:
            fnames = tbl['fname']
            for fname in fnames:
                irsa_url = f"https://irsa.ipac.caltech.edu/ibe/data/spitzer/{spitzertbl}/{fname}?center={crd.ra.deg},{crd.dec.deg}&size={size.to(u.arcmin).value}arcmin"
                if 'mosaics24' in irsa_url and 'covg' not in irsa_url and 'mask' not in irsa_url and 'std' not in irsa_url:
                    fh = fits.open(irsa_url)
                    files['MG'] = fh
    return files


def make_sed_plot(coordinate, width=10*u.arcsec, surveys=Magpis.list_surveys(), figure=None,
                  frame='icrs',
                  verbose=True,
                  images_to_use=['gpsglimpse36', 'gpsglimpse45', 'gpsglimpse58', 'gpsglimpse80', #'mipsgal',
                                 'ALMA_finaliter_prefix_b3', 'ALMA_finaliter_prefix_b6'],
                  skip_no_alma=True,
                  basepath='/blue/adamginsburg/adamginsburg/SPICY_ALMAIMF/'):

    coordname = "{0:06.3f}_{1:06.3f}".format(coordinate.galactic.l.deg,
                                             coordinate.galactic.b.deg)
    regname = f'GAL{int(coordinate.galactic.l.deg)}'

    #mgps_cutout = Cutout2D(mgps_fh.data, coordinate.transform_to(frame.name), size=width*2, wcs=wcs.WCS(mgps_fh.header))
    if verbose:
        print(f"Retrieving MAGPIS data for {coordname} ({coordinate.to_string()} {coordinate.frame.name})")
    # we're treating 'width' as a radius elsewhere, here it's a full width
    images = {survey:getimg(coordinate, image_size=width*3, survey=survey) for survey in surveys}
    images = {x:y for x,y in images.items() if y is not None}
    #images['mgps'] = [mgps_cutout]
    
    spz = get_spitzer_data(coordinate, width*2)
    for key1, key2 in zip(['I1', 'I2', 'I3', 'I4', 'MG'], 
                          ['gpsglimpse36',
                           'gpsglimpse45',
                           'gpsglimpse58',
                           'gpsglimpse80',
                           'mipsgal']):
        images[key2] = spz[key1]
                          

    regdir = os.path.join(basepath, regname)
    if not os.path.exists(regdir):
        os.mkdir(regdir)
    higaldir = os.path.join(basepath, regname, 'HiGalCutouts')
    if not os.path.exists(higaldir):
        os.mkdir(higaldir)
    if not any([os.path.exists(f"{higaldir}/{coordname}_{wavelength}.fits")
                for wavelength in map(int, HiGal.HIGAL_WAVELENGTHS.values())]):
        if verbose:
            print(f"Retrieving HiGal data for {coordname} ({coordinate.to_string()} {coordinate.frame.name})")
        higal_ims = HiGal.get_images(coordinate, radius=width*1.5)
        for hgim in higal_ims:
            images['HiGal{0}'.format(hgim[0].header['WAVELEN'])] = hgim
            hgim.writeto(f"{higaldir}/{coordname}_{hgim[0].header['WAVELEN']}.fits")
    else:
        if verbose:
            print(f"Loading HiGal data from disk for {coordname} ({coordinate.to_string()} {coordinate.frame.name})")
        for wavelength in map(int, HiGal.HIGAL_WAVELENGTHS.values()):
            hgfn = f"{higaldir}/{coordname}_{wavelength}.fits"
            if os.path.exists(hgfn):
                hgim = fits.open(hgfn)
                images['HiGal{0}'.format(hgim[0].header['WAVELEN'])] = hgim

    if 'gpsmsx2' in images:
        # redundant, save some space for a SED plot
        del images['gpsmsx2']
    if 'gps90' in images:
        # too low-res to be useful
        del images['gps90']
        
    # load ALMA-IMF images
    for key, bands in alma_wcses.items():
        for band, ww in bands.items():
            if ww.footprint_contains(coordinate):
                images[f'ALMA_{band}'] = [SpectralCube.read(f'{alma_basepath}/{prefixes[key][band]}.image.tt0.fits',
                                               format='fits', use_dask=False)[0].hdu]
                
    if skip_no_alma and not any('ALMA' in key for key in images):
        return

    if figure is None:
        figure = pl.figure(figsize=(15,7))


    # coordinate stuff so images can be reprojected to same frame
    if skip_no_alma:
        # guaranteed that ALMA will be present
        ww = WCS(images['ALMA_finaliter_prefix_b3'][0].header).celestial
    else:
        # ARBITRARY: just pick a random one
        for key in images:
            im = images[key][0]
        ww = WCS(im.header).celestial
        
    target_header = ww.to_header()
    del target_header['LONPOLE']
    del target_header['LATPOLE']
    #mgps_pixscale = (wcs.utils.proj_plane_pixel_area(ww)*u.deg**2)**0.5
    pixscale = 0.25*u.arcsec
    target_header['NAXES'] = 2
    target_header['NAXIS1'] = target_header['NAXIS2'] = (width / pixscale).decompose().value
    #shape = [int((width / mgps_pixscale).decompose().value)]*2
    outframe = wcs.utils.wcs_to_celestial_frame(ww)
    crd_outframe = coordinate.transform_to(outframe)

    figure.clf()

    imagelist = sorted(images.items(), key=lambda x: wlmap[x[0]])
    
    imagelist = [(x,y) for x,y in imagelist if x in images_to_use]

    #for ii, (survey,img) in enumerate(images.items()):
    if verbose:
        print(f"Found {len(imagelist)} images")
    for ii, (survey,img) in enumerate(imagelist):
        #if survey not in images_to_use:
        #    continue

        if hasattr(img[0], 'header'):
            inwcs = wcs.WCS(img[0].header).celestial
            pixscale_in = (wcs.utils.proj_plane_pixel_area(inwcs)*u.deg**2)**0.5

            target_header['CDELT1'] = -pixscale_in.value
            target_header['CDELT2'] = pixscale_in.value
            target_header['CRVAL1'] = crd_outframe.spherical.lon.deg
            target_header['CRVAL2'] = crd_outframe.spherical.lat.deg
            axsize = int((width*2.5 / pixscale_in).decompose().value)
            target_header['NAXIS1'] = target_header['NAXIS2'] = axsize
            target_header['CRPIX1'] = target_header['NAXIS1']/2
            target_header['CRPIX2'] = target_header['NAXIS2']/2
            shape_out = [axsize, axsize]

            if verbose:
                print(f"Reprojecting {survey} to scale {pixscale_in} with shape {shape_out} and center {crd_outframe.to_string()}")

            outwcs = wcs.WCS(target_header)

            new_img,_ = reproject.reproject_interp((img[0].data, inwcs), target_header, shape_out=shape_out)
        else:
            new_img = img[0].data
            outwcs = img[0].wcs
            pixscale_in = (wcs.utils.proj_plane_pixel_area(outwcs)*u.deg**2)**0.5

        ax = figure.add_subplot(1, 7, ii+1, projection=outwcs)
        ax.set_title("{0}: {1}".format(survey_titles[survey], wlmap[survey]))

        if not np.any(np.isfinite(new_img)):
            if verbose:
                print(f"SKIPPING {survey}")
            continue

        norm = visualization.ImageNormalize(new_img,
                                            interval=visualization.PercentileInterval(99.95),
                                            stretch=visualization.AsinhStretch(),
                                           )

        ax.imshow(new_img, origin='lower', interpolation='none', norm=norm, cmap='gray_r')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticklabels('')
        ax.yaxis.set_ticklabels('')
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticklabel_visible(False)

        if 'GLON' in outwcs.wcs.ctype[0]:
            xpix, ypix = outwcs.wcs_world2pix(coordinate.galactic.l, coordinate.galactic.b, 0)
        else:
            xpix, ypix = outwcs.wcs_world2pix(coordinate.fk5.ra, coordinate.fk5.dec, 0)
        ax.set_xlim(xpix - (width/pixscale_in), xpix + (width/pixscale_in))
        ax.set_ylim(ypix - (width/pixscale_in), ypix + (width/pixscale_in))

        # scalebar = 1 arcmin

        ax.plot([xpix - width/pixscale_in + 5*u.arcsec/pixscale_in,
                 xpix - width/pixscale_in + 65*u.arcsec/pixscale_in],
                [ypix - width/pixscale_in + 5*u.arcsec/pixscale_in,
                 ypix - width/pixscale_in + 5*u.arcsec/pixscale_in],
                linestyle='-', linewidth=1, color='w')
        ax.plot(crd_outframe.spherical.lon.deg, crd_outframe.spherical.lat.deg, marker=((0,-10), (0, -4)), color='w', linestyle='none',
                markersize=20, markeredgewidth=0.5,
                transform=ax.get_transform('world'))
        ax.plot(crd_outframe.spherical.lon.deg, crd_outframe.spherical.lat.deg, marker=((4, 0), (10, 0)), color='w', linestyle='none',
                markersize=20, markeredgewidth=0.5,
                transform=ax.get_transform('world'))

    pl.tight_layout()
    return figure
