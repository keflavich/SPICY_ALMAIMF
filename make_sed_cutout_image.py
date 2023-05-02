import numpy as np
import os
import pylab as pl
import reproject

from astroquery.magpis import Magpis
from astroquery.higal import HiGal
from astropy import units as u, coordinates
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy import wcs
from astropy import visualization

import paths

import matplotlib as mpl

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
                 'gpsglimpse36': "GLIMPSE",
                 'gpsglimpse45': "GLIMPSE",
                 'gpsglimpse58': "GLIMPSE",
                 'gpsglimpse80': "GLIMPSE",
                 'mipsgal': "MIPSGAL",
                 'atlasgal': "ATLASGAL",
                 'bolocam': "BGPS",
                 'mgps': "$\mathbf{MGPS-90}$",
                 'HiGal70':  "Hi-Gal",
                 'HiGal160': "Hi-Gal",
                 'HiGal250': "Hi-Gal",
                 'HiGal350': "Hi-Gal",
                 'HiGal500': "Hi-Gal",
                 }

def make_sed_plot(coordinate, mgpsfile, width=1*u.arcmin, surveys=Magpis.list_surveys(), figure=None,
                  regname='GAL_031'):

    mgps_fh = fits.open(mgpsfile)[0]
    frame = wcs.utils.wcs_to_celestial_frame(wcs.WCS(mgps_fh.header))

    coordname = "{0:06.3f}_{1:06.3f}".format(coordinate.galactic.l.deg,
                                             coordinate.galactic.b.deg)

    mgps_cutout = Cutout2D(mgps_fh.data, coordinate.transform_to(frame.name), size=width*2, wcs=wcs.WCS(mgps_fh.header))
    print(f"Retrieving MAGPIS data for {coordname} ({coordinate.to_string()} {coordinate.frame.name})")
    # we're treating 'width' as a radius elsewhere, here it's a full width
    images = {survey:getimg(coordinate, image_size=width*2.75, survey=survey) for survey in surveys}
    images = {x:y for x,y in images.items() if y is not None}
    images['mgps'] = [mgps_cutout]

    regdir = os.path.join(paths.basepath, regname)
    if not os.path.exists(regdir):
        os.mkdir(regdir)
    higaldir = os.path.join(paths.basepath, regname, 'HiGalCutouts')
    if not os.path.exists(higaldir):
        os.mkdir(higaldir)
    if not any([os.path.exists(f"{higaldir}/{coordname}_{wavelength}.fits")
                for wavelength in map(int, HiGal.HIGAL_WAVELENGTHS.values())]):
        print(f"Retrieving HiGal data for {coordname} ({coordinate.to_string()} {coordinate.frame.name})")
        higal_ims = HiGal.get_images(coordinate, radius=width*1.5)
        for hgim in higal_ims:
            images['HiGal{0}'.format(hgim[0].header['WAVELEN'])] = hgim
            hgim.writeto(f"{higaldir}/{coordname}_{hgim[0].header['WAVELEN']}.fits")
    else:
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

    if figure is None:
        figure = pl.figure(figsize=(15,12))


    # coordinate stuff so images can be reprojected to same frame
    ww = mgps_cutout.wcs.celestial
    target_header = ww.to_header()
    del target_header['LONPOLE']
    del target_header['LATPOLE']
    mgps_pixscale = (wcs.utils.proj_plane_pixel_area(ww)*u.deg**2)**0.5
    target_header['NAXES'] = 2
    target_header['NAXIS1'] = target_header['NAXIS2'] = (width / mgps_pixscale).decompose().value
    #shape = [int((width / mgps_pixscale).decompose().value)]*2
    outframe = wcs.utils.wcs_to_celestial_frame(ww)
    crd_outframe = coordinate.transform_to(outframe)

    figure.clf()

    imagelist = sorted(images.items(), key=lambda x: wlmap[x[0]])

    #for ii, (survey,img) in enumerate(images.items()):
    for ii, (survey,img) in enumerate(imagelist):

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

            print(f"Reprojecting {survey} to scale {pixscale_in} with shape {shape_out} and center {crd_outframe.to_string()}")

            outwcs = wcs.WCS(target_header)

            new_img,_ = reproject.reproject_interp((img[0].data, inwcs), target_header, shape_out=shape_out)
        else:
            new_img = img[0].data
            outwcs = img[0].wcs
            pixscale_in = (wcs.utils.proj_plane_pixel_area(outwcs)*u.deg**2)**0.5

        ax = figure.add_subplot(4, 5, ii+1, projection=outwcs)
        ax.set_title("{0}: {1}".format(survey_titles[survey], wlmap[survey]))

        if not np.any(np.isfinite(new_img)):
            print(f"SKIPPING {survey}")
            continue

        norm = visualization.ImageNormalize(new_img,
                                            interval=visualization.PercentileInterval(99.95),
                                            stretch=visualization.AsinhStretch(),
                                           )

        ax.imshow(new_img, origin='lower', interpolation='none', norm=norm)
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
