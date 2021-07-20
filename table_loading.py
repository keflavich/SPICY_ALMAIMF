import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy import table

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy import wcs
from astropy.wcs import WCS

from astroquery.vizier import Vizier
from astroquery.svo_fps import SvoFps

import photutils

from spectral_cube import SpectralCube
import os
import glob

# load up ALMA-IMF metadata
import sys
sys.path.append('/orange/adamginsburg/ALMA_IMF/reduction/analysis/')
from spectralindex import prefixes
import spitzer_plots
from spitzer_plots import show_fov_on_spitzer, contour_levels

from sedfitter.filter import Filter
from sedfitter.extinction import Extinction
from sedfitter.source import Source

from dust_extinction.parameter_averages import F19

geometries = ["s---s-i", "s---smi", "sp--s-i", "sp--h-i", "s---smi", "s-p-smi",
              "s-p-hmi", "s-pbsmi", "s-pbhmi", "s-u-smi", "s-u-hmi", "s-ubsmi",
              "spu-smi", "spu-hmi", "spubsmi", "spubhmi",]
              

def get_spicy_tbl():
    tbl = Table.read('/blue/adamginsburg/adamginsburg/ALMA_IMF/SPICY_ALMAIMF/table1.fits')
    #tbl = Table.read('https://sites.astro.caltech.edu/~mkuhn/SPICY/table1.fits')

    coords = SkyCoord(tbl['l'], tbl['b'], frame='galactic', unit=(u.deg, u.deg))

    return tbl,coords

def add_MIPS_matches(tbl):
    MIPS_IDs = tbl['MIPS']
    row_limit = len(tbl)
    MIPS_IDs_mask = np.array(['MG' in mid for mid in MIPS_IDs])
    if any(MIPS_IDs_mask):
        mips_match = Vizier(row_limit=row_limit,
                            columns=["MIPSGAL", "S24", "e_S24"]
                           ).query_constraints(MIPSGAL="=,"+",".join(map(str,
                                                                         MIPS_IDs[MIPS_IDs_mask])),
                                               catalog='J/AJ/149/64/catalog')[0]
        mips_match.rename_column('MIPSGAL','MIPS')
        return table.join(tbl, mips_match, join_type='left')
    else:
        tbl['MIPS'] = ''
        tbl['S24'] = np.nan
        tbl['e_S24'] = np.nan
        return tbl

def add_VVV_matches(tbl):
    virac_numbers = tbl['VIRAC']
    row_limit = len(tbl)
    # VIRAC uses numbers, not IDs, so we can just do comma-separated
    virac_match = Vizier(row_limit=row_limit).query_constraints(srcid=",".join(map(str, virac_numbers[~virac_numbers.mask])),
                                                           catalog='II/364/virac')[0]
    virac_match.rename_column('srcid','VIRAC')

    mskvirac = tbl['VIRAC'].mask
    tbl['VIRAC'].mask = False
    tbl['VIRAC'][mskvirac] = -99999
    rslt = table.join(tbl, virac_match, join_type='left', keys='VIRAC')
    rslt['VIRAC'].mask = mskvirac
    return rslt


def find_ALMAIMF_matches(tbl):
    # determine number of SPICY sources in each ALMA FOV
    os.chdir('/orange/adamginsburg/ALMA_IMF/May2021Release/')

    prefixes['W43MM1'] = dict(
        finaliter_prefix_b3="W43-MM1/B3/cleanest/W43-MM1_B3_uid___A001_X1296_X1af_continuum_merged_12M_robust0_selfcal4_finaliter",
        finaliter_prefix_b6="W43-MM2/B6/cleanest/W43-MM2_B6_uid___A001_X1296_X113_continuum_merged_12M_robust0_selfcal5_finaliter",)

    all_matches = np.zeros(len(tbl), dtype='bool')
    fieldids = np.empty(len(tbl), dtype='S8')

    for fieldid, pfxs in prefixes.items():
        cube = SpectralCube.read(pfxs['finaliter_prefix_b3']+".image.tt0.fits", format='fits', use_dask=False).minimal_subcube()
        ww = cube.wcs.celestial
        ww._naxis = cube.shape[1:]
        matches = ww.footprint_contains(coords)
        all_matches |= matches
        fieldids[matches] = fieldid

    tbl['in_ALMAIMF'] = all_matches
    tbl['ALMAIMF_FIELDID'] = fieldids
    return tbl


def show_source_on_spitzer(fieldid, coords,
                           basepath='/orange/adamginsburg/ALMA_IMF/2017.1.01355.L/RestructuredImagingResults',
                           mips=False):
    pfxs = prefixes[fieldid]
    fig = show_fov_on_spitzer(**{key: f'{basepath}/{val}' for key,val in pfxs.items()},
                              fieldid=fieldid, spitzerpath=f'{basepath}/spitzer_datapath',
                              contour_level=contour_levels[fieldid], mips=mips)

    cube = SpectralCube.read(basepath + '/' + pfxs['finaliter_prefix_b3']+".image.tt0.fits",
                             format='fits', use_dask=False).minimal_subcube()
    ww = cube.wcs.celestial
    ww._naxis = cube.shape[1:]
    matches = ww.footprint_contains(coords)

    cc = coords[matches]

    ax = fig.gca()
    ax.plot(cc.fk5.ra.deg, cc.fk5.dec.deg, 'wo', mfc='none', mec='w', markersize=10, transform=ax.get_transform('fk5'), )


# hacky function to extract the rows of an SED table as a plottable entry
def getrow(tb, rownum, keys=['Ymag', 'Zmag', 'Jmag', 'Hmag', 'Ksmag','mag3_6', 'mag4_5', 'mag5_8', 'mag8_0']):
    return np.array([tb[rownum][key] for key in keys])


magcols = ['Ymag', 'Zmag', 'Jmag', 'Hmag', 'Ksmag','mag3_6', 'mag4_5', 'mag5_8', 'mag8_0']
emagcols = ['Yell', 'Zell', 'Jell', 'Hell', 'KsEll','e_mag3_6', 'e_mag4_5', 'e_mag5_8', 'e_mag8_0']


def get_filters():
    # these are the official filternames on SVO_FPS
    filternames = ['Paranal/VISTA.Y', 'Paranal/VISTA.Z', 'Paranal/VISTA.J', 'Paranal/VISTA.H', 'Paranal/VISTA.Ks',
                   'Spitzer/IRAC.I1', 'Spitzer/IRAC.I2', 'Spitzer/IRAC.I3', 'Spitzer/IRAC.I4', 'Spitzer/MIPS.24mu',
                   'Herschel/Pacs.blue', 'Herschel/Pacs.red', 'Herschel/SPIRE.PSW', 'Herschel/SPIRE.PMW', 'Herschel/SPIRE.PLW'
                  ]
    # keep only the non "_ext" SPIRE filters (but we should look up which is more appropriate)
    spire_filters = SvoFps.get_filter_list(facility='Herschel', instrument='Spire')
    spire_filters = spire_filters[['_ext' not in fid for fid in spire_filters['filterID']]]

    filter_meta = table.vstack([SvoFps.get_filter_list(facility='Paranal', instrument='VIRCAM'),
                                SvoFps.get_filter_list(facility='Spitzer', instrument='IRAC'),
                                SvoFps.get_filter_list(facility='Spitzer', instrument='MIPS')[0],
                                SvoFps.get_filter_list(facility='Herschel', instrument='Pacs'),
                                spire_filters,
                               ])
    zpts = {filtername: filter_meta[filter_meta['filterID']==filtername]['ZeroPoint'] for filtername in filternames}

    filtercurves = {filtername: SvoFps.get_transmission_data(filtername) for filtername in filternames}
    wavelengths = [np.average(filtercurves[filtername]['Wavelength'],
                              weights=filtercurves[filtername]['Transmission'])
                  for filtername in filternames]
    wavelength_dict = {filtername: np.average(filtercurves[filtername]['Wavelength'],
                                              weights=filtercurves[filtername]['Transmission'])*u.AA
                       for filtername in filternames}

    filterfreqs = {filtername: u.Quantity(filtercurves[filtername]['Wavelength'], u.AA).to(u.Hz, u.spectral()) for filtername in filternames}
    filtertrans = {filtername: np.array(filtercurves[filtername]['Transmission'])[np.argsort(filterfreqs[filtername])]
                  for filtername in filternames}
    filterfreqs = {filtername: np.sort(filterfreqs[filtername]) for filtername in filternames}

    sed_filters = [Filter(name=filtername,
                          central_wavelength=wl*u.AA,
                          nu=filterfreqs[filtername],
                          response=filtertrans[filtername])
                   for filtername, wl in zip(filternames, wavelengths)]


    # Add in the custom ALMA-IMF filters
    almaimf_bandends_1mm = [[216.10085679, 216.36181569],
                            [217.05104378, 217.31175857],
                            [219.90488464, 220.04866835],
                            [218.13102322, 218.39222624],
                            [219.51976276, 219.66379059],
                            [230.31532951, 230.81137113],
                            [231.06503709, 231.56181105],
                            [231.52507012, 233.42623749]]*u.GHz
    nu_1mm = np.linspace(almaimf_bandends_1mm.min(), almaimf_bandends_1mm.max(), 5000)
    response_1mm = np.zeros(nu_1mm.size, dtype='bool')
    for start, stop in almaimf_bandends_1mm:
        response_1mm |= (nu_1mm > start) & (nu_1mm < stop)
    sed_filters.append(Filter(name='ALMA-IMF_1mm',
                              central_wavelength=(228.15802*u.GHz).to(u.mm, u.spectral()),
                              nu=nu_1mm,
                              response=response_1mm.astype(float),
                             ))

    for filterfunc in sed_filters:
        filterfunc.normalize()


    almaimf_bandends_3mm = [[ 93.13410936,  93.25141259],
                            [ 91.75059068,  92.68755174],
                            [102.15273354, 103.0896946 ],
                            [104.55323851, 105.49019957]]*u.GHz
    nu_3mm = np.linspace(almaimf_bandends_3mm.min(), almaimf_bandends_3mm.max(), 5000)
    response_3mm = np.zeros(nu_3mm.size, dtype='bool')
    for start, stop in almaimf_bandends_3mm:
        response_3mm |= (nu_3mm > start) & (nu_3mm < stop)
    sed_filters.append(Filter(name='ALMA-IMF_3mm',
                              central_wavelength=(99.68314596*u.GHz).to(u.mm, u.spectral()),
                              nu=nu_3mm,
                              response=response_3mm.astype(float),
                             ))

    wavelength_dict['ALMA-IMF_1mm'] = (228.15802*u.GHz).to(u.um, u.spectral())
    wavelength_dict['ALMA-IMF_3mm'] = (99.68314596*u.GHz).to(u.um, u.spectral())

    return sed_filters, wavelength_dict, filternames

def make_extinction():
    # make an extinction law
    ext = F19(3.1)

    # https://arxiv.org/abs/0903.2057
    # 1.34 is from memory
    guyver2009_avtocol = (2.21e21 * u.cm**-2 * (1.34*u.Da)).to(u.g/u.cm**2)
    ext_wav = np.sort((np.geomspace(0.301, 8.699, 1000)/u.um).to(u.um, u.spectral()))
    ext_vals = ext.evaluate(ext_wav, Rv=3.1)
    extinction = Extinction()
    extinction.wav = ext_wav
    extinction.chi = ext_vals / guyver2009_avtocol

    return extinction


sed_filters, wavelength_dict, filternames = get_filters()

def fit_a_source(data, error, valid,
    geometry='s-ubhmi', robitaille_modeldir='/blue/adamginsburg/richardson.t/research/flux/robitaille_models/',
                 extinction=make_extinction(),
                 filters=filternames,
                 aperture_size=3*u.arcsec,
                 distance_range=[1.8, 2.2]*u.kpc,
                 av_range=[4,40]
                ):

    # Define path to models
    model_dir = f'{robitaille_modeldir}/{geometry}'

    apertures = u.Quantity([aperture_size]*len(filternames))

    source = Source()

    source.valid = valid
    source.flux = data
    source.error =  error

    fitter = Fitter(filter_names=np.array(filters),
                    apertures=apertures,
                    model_dir=model_dir,
                    extinction_law=extinction,
                    distance_range=distance_range,
                    av_range=av_range,
                   )


    # Run the fitting
    fitinfo = fitter.fit(source)

    return fitinfo


def get_data_to_fit(rownumber, tbl, filters=filternames):

    flx = getrow(tbl, rownumber, keys=[key+"_flux" for key in filters])
    error = getrow(tbl, rownumber, keys=[key+"_eflux" for key in filters])

    valid = np.zeros(flx.size, dtype='int')

    # data to fit directly: both the flux and error are "valid"
    valid[(np.isfinite(flx) & np.isfinite(error))] = 1

    # data to ignore: neither the flux nor error are valid (nan or masked)
    valid[(~np.isfinite(flx) & ~np.isfinite(error))] = 0

    # data to treat as upper limits: the flux is not specified, but the error is
    valid[(~np.isfinite(flx) & np.isfinite(error))] = 3

    # set the "flux" to be the 3-sigma error wherever we're treating it as an upper limit
    flx[valid == 3] = error[valid == 3] * 3
    # then, set the confidence associated with that upper limit
    error[valid == 3] = 0.997 # 0.997 is (approximately) 3-sigma

    return flx, error, valid


def add_alma_photometry(tbl, aperture_radius=3*u.arcsec,
                        annulus_inner=3*u.arcsec, annulus_outer=5*u.arcsec,
                        basepath='/orange/adamginsburg/ALMA_IMF/2017.1.01355.L/RestructuredImagingResults',
                        band='b3', wlname='3mm'):

    tbl[f'ALMA-IMF_{wlname}_flux'] = np.zeros(len(tbl), dtype='float')
    tbl[f'ALMA-IMF_{wlname}_eflux'] = np.zeros(len(tbl), dtype='float')

    for fieldid in np.unique(tbl['ALMAIMF_FIELDID']):
        pfxs = prefixes[fieldid]
        cube = SpectralCube.read(basepath + '/' + pfxs[f'finaliter_prefix_{band}']+".image.tt0.fits",
                             format='fits', use_dask=False).minimal_subcube()
        alma_rms = cube.mad_std()

        ww = cube.wcs.celestial
        ww._naxis = cube.shape[1:]

        match = tbl['ALMAIMF_FIELDID'] == fieldid

        crds = SkyCoord(tbl['ra'], tbl['dec'])[match]
        sky_apertures = photutils.aperture.SkyCircularAperture(crds, aperture_radius)
        apertures = sky_apertures.to_pixel(ww)

        sky_annulus_aperture = photutils.aperture.SkyCircularAnnulus(crds, r_in=annulus_inner, r_out=annulus_outer)
        annulus_apertures = sky_annulus_aperture.to_pixel(ww)

        annulus_masks = annulus_apertures.to_mask(method='center')
        data = cube[0]

        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(data)
            if annulus_data is None:
                bkg_median.append(np.nan * data.unit)
                continue
            annulus_data_1d = annulus_data[mask.data != 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        bkg_median = u.Quantity(bkg_median)
        phot = photutils.aperture_photometry(data, apertures)
        phot['annulus_median'] = bkg_median
        phot['aper_bkg'] = bkg_median * apertures.area
        phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
        phot['flux'] = phot['aper_sum_bkgsub'] / cube.pixels_per_beam * u.beam
        phot['significant'] = phot['flux'] > 3 * alma_rms*u.beam

        tbl[f'ALMA-IMF_{wlname}_flux'][match] = np.where(phot['significant'], phot['flux'], np.nan)
        tbl[f'ALMA-IMF_{wlname}_eflux'][match] = alma_rms

    return tbl


def get_flx(crd, data, ww):
    crd = crd.transform_to(ww.wcs.radesys.lower())
    xpix, ypix = ww.all_world2pix(crd.spherical.lon, crd.spherical.lat, 0)
    xpix = int(np.round(xpix))
    ypix = int(np.round(ypix))
    return data[ypix, xpix]


def add_herschel_limits(tbl, coords, wls=[70,160,250,350,500], higalpath='/orange/adamginsburg/higal/'):
    rows = []
    for crd in coords.galactic:
        galrnd = int(crd.galactic.l.deg)
        # search +/- 2 deg:
        for gal in np.array([0,-1,1,-2,2]) + int(galrnd):
            files = glob.glob(f'{higalpath}/Field{gal}*.fits*')
            if any(files):
                fh = fits.open(files[0])[1]
                ww = wcs.WCS(fh.header)
                if ww.footprint_contains(crd):
                    flx = {fn.split("Parallel")[1].split("_")[0]:
                           get_flx(crd, fits.getdata(fn, ext=1), wcs.WCS(fits.getheader(fn, ext=1)))
                           for fn in files}
                    break
        rows.append(flx)
    columns = {wl: [row[wl] for row in rows] for wl in wls}
    tbl.add_columns(columns)
    return tbl
            

if __name__ == "__main__":

    fulltbl, coords = get_spicy_tbl()
    fulltbl = find_ALMAIMF_matches(fulltbl)
    tblmsk = fulltbl['in_ALMAIMF']
    tbl = fulltbl[tblmsk]
    coords = coords[tblmsk]

    tbl = add_herschel_limits(tbl, coords)
    tbl = add_MIPS_matches(tbl)
    tbl = add_VVV_matches(tbl)
    tbl = add_alma_photometry(tbl, band='b3', wlname='3mm')
    tbl = add_alma_photometry(tbl, band='b6', wlname='1mm')

    tbl.write('SPICY_withAddOns.fits', overwrite=True)
