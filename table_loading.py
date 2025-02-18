# basics
import os
import numpy as np
import pylab as pl
import glob
import time

# utility
from tqdm.auto import tqdm

# astropy
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
from astropy import table
from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy import wcs
from astropy.wcs import WCS
from astropy.table import QTable
from astropy.modeling.models import BlackBody

# astroquery
from astroquery.vizier import Vizier
from astroquery.ukidss import Ukidss
from astroquery.svo_fps import SvoFps

# photutils; load up ALMA-IMF metadata
import sys
import photutils
sys.path.append('/orange/adamginsburg/ALMA_IMF/reduction/analysis/')
from spectralindex import prefixes
from spectral_cube import SpectralCube

# spitzer plots
import spitzer_plots
from spitzer_plots import show_fov_on_spitzer, contour_levels, get_spitzer_data

# SED fitting
from sedfitter import fit, Fitter
from sedfitter.filter import Filter
from sedfitter.extinction import Extinction
from sedfitter.source import Source
from sedfitter.sed import SEDCube

# extinction
from dust_extinction.parameter_averages import F19
from dust_extinction.averages import CT06_MWLoc

# writing to file
import pickle

# convenience-sorting of fields based on which NIR observations they have
ukidss_fields = ['G10','G12','W43MM1','W43MM2','W43MM3','W51-E','W51IRS2']
virac_fields = ['G008','G327','G328','G333','G337','G338','G351','G353']

# define geometries, per Robitaille paper
geometries = ['s-pbhmi', 's-pbsmi',
              'sp--h-i', 's-p-hmi',
              'sp--hmi', 'sp--s-i',
              's-p-smi', 'sp--smi',
              'spubhmi', 'spubsmi',
              'spu-hmi', 'spu-smi',
              's---s-i', 's---smi',
              's-ubhmi', 's-ubsmi',
              's-u-hmi', 's-u-smi']

def get_spicy_tbl():
    # retrieve the SPICY catalog
    tbl = Table.read('/blue/adamginsburg/adamginsburg/ALMA_IMF/SPICY_ALMAIMF/table1.fits')
        #alternatively: tbl = Table.read('https://sites.astro.caltech.edu/~mkuhn/SPICY/table1.fits')
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
        tbl = table.join(tbl, mips_match, join_type='left')
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

    mskvirac = tbl['VIRAC'].mask.flatten().tolist()
    tbl['VIRAC'].mask = False
    tbl['VIRAC'][mskvirac] = -99999
    rslt = table.join(tbl, virac_match, join_type='left', keys='VIRAC')
    rslt.sort('SPICY')
    rslt['VIRAC'].mask = mskvirac

    return rslt

def add_VVV_limits(tbl, limits={"Y": 17,
                  "Z": 17.5,
                  "J": 16.5,
                  "H": 16,
                  "Ks": 15.5,}):

    for key in limits.keys():
        if f'{key}mag' and f'{key}ell' in tbl.keys():
            tbl[f'{key}ell'].fill_value = limits[key]
            tbl[f'{key}ell'][tbl['NIR data'] == "VIRAC"] = tbl[f'{key}ell'][tbl['NIR data'] == "VIRAC"].filled()
        else: print(f'{key} band not found.')

    return tbl

def add_UKIDSS_matches(tbl):
    mskukidss = tbl['NIR data'] == "UKIDSS"
    row_limit = len(tbl)

    ukidss_match = Vizier(row_limit=row_limit).query_constraints(UGPS=list(tbl['UKIDSS'][~mskukidss]),catalog='II/316/gps6')[0]
    print(len(ukidss_match))

    ukidss_match.rename_column('UGPS','UKIDSS')

    rslt = table.join(tbl, ukidss_match, join_type='left', keys='UKIDSS')

    return rslt

def add_UKIDSS_limits(tbl, limits={"J": 19.9,
                  "H": 19.0,
                  "K": 18.8,}):

    for key in limits.keys():
        if f'{key}mag' and f'{key}ell' in tbl.keys():
            tbl[f'{key}ell'].fill_value = limits[key]
            tbl[f'{key}ell'][tbl['NIR data'] == "UKIDSS"] = tbl[f'{key}ell'][tbl['NIR data'] == "UKIDSS"].filled()
        else: print(f'{key} band not found.')

    return tbl

def find_ALMAIMF_matches(tbl, coords):
    # determine number of SPICY sources in each ALMA FOV
    os.chdir('/orange/adamginsburg/web/secure/ALMA-IMF/May2021Release/')

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

def show_source_on_spitzer(fieldid, coords, source=None,
                           basepath='/orange/adamginsburg/ALMA_IMF/2017.1.01355.L/RestructuredImagingResults',
                           mips=False):

    tbl, coords = get_spicy_tbl()
    tbl = find_ALMAIMF_matches(tbl, coords)
    # tbl = Table.read('/blue/adamginsburg/adamginsburg/ALMA_IMF/SPICY_ALMAIMF/SPICY_withAddOns.fits')

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

    tbl = tbl[matches]
    #tbl = tbl[tbl['ALMAIMF_FIELDID'] == fieldid]

    try:
        if source == None:
            ax.plot(cc[0:len(cc)].fk5.ra.deg, cc[0:len(cc)].fk5.dec.deg, 'w*',
                    mfc='none', mec='w', markersize=17, transform=ax.get_transform('fk5'), )
        elif len(tbl) == 1:
            ax.plot(cc.fk5.ra.deg, cc.fk5.dec.deg, 'w*',
                    mfc='none', mec='w', markersize=17, transform=ax.get_transform('fk5'), )
        else:
            tbl.add_index('SPICY')
            tbl.sort('SPICY')
            rownum = tbl.loc_indices[source]
            ax.plot(cc[rownum:rownum+1].fk5.ra.deg, cc[rownum:rownum+1].fk5.dec.deg, 'w*',
                    mfc='none', mec='w', markersize=17, transform=ax.get_transform('fk5'), )

    except AttributeError:
        print("Failed")

    return fig

def get_filters(hemisphere='south'):
    # these are the official filternames on SVO_FPS
    if hemisphere == 'north':
        filternames = ['UKIRT/UKIDSS.J', 'UKIRT/UKIDSS.H', 'UKIRT/UKIDSS.K',
                   'Spitzer/IRAC.I1', 'Spitzer/IRAC.I2', 'Spitzer/IRAC.I3', 'Spitzer/IRAC.I4', 'Spitzer/MIPS.24mu',
                   'Herschel/Pacs.blue', 'Herschel/Pacs.red', 'Herschel/SPIRE.PSW', 'Herschel/SPIRE.PMW', 'Herschel/SPIRE.PLW'
                  ]
        # keep only the non "_ext" SPIRE filters (but we should look up which is more appropriate)
        spire_filters = SvoFps.get_filter_list(facility='Herschel', instrument='Spire')
        spire_filters = spire_filters[['_ext' not in fid for fid in spire_filters['filterID']]]

        filter_meta = table.vstack([SvoFps.get_filter_list(facility='UKIRT', instrument='WFCAM'),
                                SvoFps.get_filter_list(facility='Spitzer', instrument='IRAC'),
                                SvoFps.get_filter_list(facility='Spitzer', instrument='MIPS')[0],
                                SvoFps.get_filter_list(facility='Herschel', instrument='Pacs'),
                                spire_filters,
                               ])

    elif hemisphere == 'south':
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

    return sed_filters, wavelength_dict, filternames, zpts

def make_extinction():
    # make an extinction law
    ext = F19(3.1)
    ext2 = CT06_MWLoc()

    # https://arxiv.org/abs/0903.2057
    # 1.34 is from memory
    guyver2009_avtocol = (2.21e21 * u.cm**-2 * (1.34*u.Da)).to(u.g/u.cm**2)
    ext_wav = np.sort((np.geomspace(0.301, 8.699, 1000)/u.um).to(u.um, u.spectral()))
    ext_vals = ext.evaluate(ext_wav, Rv=3.1)

    # extend the extinction curve out
    ext_wav2 = np.geomspace(ext_wav.max(), 27*u.um, 100)
    ext_vals2 = ext2.evaluate(ext_wav2)

    extinction = Extinction()
    extinction.wav = np.hstack([ext_wav, ext_wav2])
    extinction.chi = np.hstack([ext_vals, ext_vals2]) / guyver2009_avtocol

    return extinction

def get_fitter(geometry, aperture_size,
               distance_range,
               robitaille_modeldir,
               filters, extinction,
               av_range):

    # Define path to models
    model_dir = f'{robitaille_modeldir}/{geometry}'

    if len(aperture_size) == 1:
        apertures = u.Quantity([aperture_size]*len(filters))
    else:
        apertures = u.Quantity(aperture_size, u.arcsec)

    if isinstance(filters, list):
        filters = np.array(filters)

    fitter = Fitter(filter_names=filters,
                    apertures=apertures,
                    model_dir=model_dir,
                    extinction_law=extinction,
                    distance_range=distance_range,
                    av_range=av_range,
                    use_memmap=True
                   )

    return fitter

def fit_a_source(data, error, valid, geometry='s-ubhmi',
        robitaille_modeldir='/blue/adamginsburg/richardson.t/research/flux/robitaille_models-1.2/',
        extinction=make_extinction(), filters=get_filters(),
        aperture_size=3*u.arcsec, distance_range=[1.8, 2.2]*u.kpc,
        av_range=[4,40], fitter=None, stash_to_mmap=False,):

    source = Source()
    source.valid = valid

    # https://sedfitter.readthedocs.io/en/stable/data.html
    # this site specifies that the fitter expects flux in mJy
    # if the data are given as a Jy-equivalent, convert them to mJy
    source.flux = u.Quantity(data, u.mJy).value
    source.error =  u.Quantity(error, u.mJy).value

    if fitter is None:
        fitter = get_fitter(geometry=geometry, aperture_size=aperture_size,
                            distance_range=distance_range, av_range=av_range,
                            robitaille_modeldir=robitaille_modeldir,
                            filters=filters, extinction=extinction)

    # Run the fitting
    fitinfo = fitter.fit(source)

    if stash_to_mmap:
        from tempfile import mkdtemp
        import os.path as path
        filename = path.join(mkdtemp(), f'{geometry}.dat')
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=fitinfo.model_fluxes.shape)
        fp[:] = fitinfo.model_fluxes[:]
        fp.flush()
        fitinfo.model_fluxes = fp
        print(f"Moved array with size {fitinfo.model_fluxes.shape} to {fp.filename}")

    return fitinfo

# nested function for convenience
def full_source_fit(tbl, rownum, filternames, apertures, robitaille_modeldir, extinction, distance_range, av_range):
    flx, error, valid = get_data_to_fit(rownum, tbl, filters=filternames+["ALMA-IMF_1mm", "ALMA-IMF_3mm"])
    ##optional: print out data points before fitting
    #datatable = Table([flx, error, valid])
    #print(datatable)

    fits = {geom:
            fit_a_source(data=flx, error=error, valid=valid,
                         geometry=geom, robitaille_modeldir=robitaille_modeldir,
                         extinction=extinction,
                         filters=filternames+["user_filters/ALMA-IMF_1mm", "user_filters/ALMA-IMF_3mm"],
                         aperture_size=apertures,
                         distance_range=distance_range,
                         av_range=av_range
                      )
            for geom in tqdm(geometries, desc = f'Fitting source {rownum+1}/{len(tbl)}')}
    return fits

def mag_to_flux(tbl, magcols, emagcols, zpts, filternames):
    # convert magnitudes to fluxes
    # (it's a pain to try to deal with a mix of magnitudes & fluxes)
    for colname, errcolname, zpn in zip(magcols, emagcols, filternames):
        zp = u.Quantity(zpts[zpn], u.Jy)
        # iterate through each colname
        if colname in tbl.keys() and errcolname in tbl.keys():
            #grab numerical value for the data; masked should be nan
            data = tbl[colname]
            error = tbl[errcolname]
            print(colname)
            if hasattr(tbl[colname], 'mask'):
                tbl[zpn+"_flux"] = flx = np.ma.masked_where(tbl[colname].mask, (zp * 10**(data.data/-2.5)).to(u.mJy))
            else:
                tbl[zpn+"_flux"] = flx = (zp * 10**(data.data/-2.5)).to(u.mJy)

            if hasattr(tbl[errcolname], 'mask') and hasattr(tbl[colname], 'mask'):
                tbl[zpn+"_eflux"] = err = np.ma.masked_where(tbl[errcolname].mask, np.where(tbl[colname].mask, (zp * 10**(error.data/-2.5)).to(u.mJy), error.quantity / (1.09*u.mag) * flx.data))
            elif not hasattr(tbl[errcolname], 'mask') and hasattr(tbl[colname], 'mask'):
                tbl[zpn+"_eflux"] = err = np.where(tbl[colname].mask, (zp * 10**(error.data/-2.5)).to(u.mJy), error.quantity / (1.09*u.mag) * flx.data)
            else:
                tbl[zpn+"_eflux"] = err = error.quantity / (1.09*u.mag) * flx.data
            tbl[zpn+"_flux"].unit = 'mJy'
            tbl[zpn+"_eflux"].unit = 'mJy'
            #err = tbl[errcolname] / (1.09*u.mag) * flx
            #tbl[zpn+"_eflux"] = err
        else: print(f'{colname} not found.')

    return tbl

# hacky function to extract the rows of an SED table as a plottable entry
def getrow(tbl, rownum, keys):
    return np.array([tbl[rownum][key] for key in keys])

apertures_VVV = {'Ymag': 1.415*u.arcsec,
                 'zmag': 1.415*u.arcsec,
                 'Jmag': 1.415*u.arcsec,
                 'Hmag': 1.415*u.arcsec,
                 'Kmag': 1.415*u.arcsec,}
apertures_UKIDSS = {'Ymag': 2*u.arcsec,
                    'zmag': 2*u.arcsec,
                    'Jmag': 2*u.arcsec,
                    'Hmag': 2*u.arcsec,
                    'Kmag': 2*u.arcsec,}
apertures_spitzer = {'mag3_6': 2.4*u.arcsec,
                     'mag4_5': 2.4*u.arcsec,
                     'mag5_8': 2.4*u.arcsec,
                     'mag8_0': 2.4*u.arcsec,
                     'S24': 6*u.arcsec,
                    }
apertures_herschel = {'70':  10*u.arcsec,
                      '160': 13.5*u.arcsec,
                      '250': 23*u.arcsec,
                      '350': 30*u.arcsec,
                      '500': 41*u.arcsec,
                     }
apertures_ALMA = {'3mm': 3*u.arcsec,
                  '1mm': 3*u.arcsec}



def get_data_to_fit(rownumber, tbl, filters):
    # remove all extraneous data from input table
    for key in filters:
        if key+"_flux" not in tbl.keys():
            tbl[key+"_flux"] = [np.nan for row in tbl]
            tbl[key+"_eflux"] = [np.nan for row in tbl]

    # extract fluxes and errors
    flx = getrow(tbl, rownumber, keys=[key+"_flux" for key in filters])
    error = getrow(tbl, rownumber, keys=[key+"_eflux" for key in filters])
    valid = np.zeros(flx.size, dtype='int')

    # set flags based on validity of data
    valid[(np.isfinite(flx) & np.isfinite(error))] = 1
        # both the flux and error are "valid": data is fitted directly
    valid[(~np.isfinite(flx) & ~np.isfinite(error))] = 0
        # neither the flux nor error are valid (nan or masked): data is discarded
    valid[(~np.isfinite(flx) & np.isfinite(error))] = 3
        # flux is not specified, but the error is: treated as upper limit

    # error-proofing: toss any data points which measure exactly 0
    valid[flx == 0] = 0
    valid[error == 0] = 0

    # set the "flux" to be the 3-sigma error wherever we're treating it as an upper limit
    flx[valid == 3] = error[valid == 3] * 3
    # then, set the confidence associated with that upper limit, AKA 3-sigma
    error[valid == 3] = 0.997

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
        tbl[f'ALMA-IMF_{wlname}_eflux'][match] = np.where(np.isfinite(phot['flux']), alma_rms, np.nan)

    return tbl

def get_flx(crd, data, ww):
    crd = crd.transform_to(ww.wcs.radesys.lower())
    xpix, ypix = ww.world_to_pixel(crd)
    xpix = int(np.round(xpix))
    ypix = int(np.round(ypix))
    return data[ypix, xpix]

def add_spicylimits(tbl, spicyupperlims = {"3_6": 14.9,
                  "4_5": 13.7,
                  "5_8": 12.9,
                  "8_0": 12.2,}):
    for key in spicyupperlims.keys():
        if f'mag{key}' and f'e_mag{key}' in tbl.keys():
            tbl[f'e_mag{key}'] = table.MaskedColumn(tbl[f'e_mag{key}'])
            tbl[f'e_mag{key}'].fill_value = spicyupperlims[key]
            try:
                tbl[f'e_mag{key}'] = tbl[f'e_mag{key}'].filled()
            except AttributeError:
                print(f"Column {key} has no masked values")
        else: print(f'{key} band not found.')
    return tbl

def add_herschel_limits(tbl, coords, wls=[70,160,250,350,500], higalpath='/orange/adamginsburg/higal/'):
    rows = []
    for crd in tqdm(coords.galactic):
        galrnd = int(crd.galactic.l.deg)
        flx = {wl: np.nan for wl in wls}
        # search +/- 2 deg:
        for gal in np.array([0,-1,1,-2,2]) + int(galrnd):
            files = glob.glob(f'{higalpath}/Field{gal}_*.fits*') + glob.glob(f"{higalpath}/l{gal}_*.fits*")
            if any(files):
                fh = fits.open(files[0])[1]
                ww = wcs.WCS(fh.header)
                if ww.footprint_contains(crd):
                    flx_ = {int(fn.split("Parallel")[1].split("_")[1]):
                           get_flx(crd, fits.getdata(fn, ext=1), wcs.WCS(fits.getheader(fn, ext=1)))
                           for fn in files
                           if wcs.WCS(fits.getheader(fn, ext=1)).footprint_contains(crd)
                          }
                    if flx_[70] != 0:
                        flx[70] = flx_[70]
                        flx[160] = flx_[160]
                    if 250 in flx_ and not np.isnan(flx_[250]):
                        flx[250] = flx_[250]
                        flx[350] = flx_[350]
                        flx[500] = flx_[500]
                    if flx[70] == 0 or np.isnan(flx[70]) or np.isnan(flx[250]):
                        # wrong field?
                        print(f"Failed match between {crd} and {files[0]}")
                        continue
                    else:
                        break
        rows.append(flx)

    # use the last successful one
    units = {int(fn.split("Parallel")[1].split("_")[1]): fits.getheader(fn,
                                                                        ext=1)['BUNIT']
             for fn in files if wcs.WCS(fits.getheader(fn,
                                                       ext=1)).footprint_contains(crd)
            }

    columns = {wl: [row[wl] for row in rows] for wl in wls}
    for name, data in columns.items():
        tbl.add_column(table.Column(name=name, data=data, unit=units[name]))
    return tbl

def add_mips_limits(tbl, coords, mipspath='/orange/adamginsburg/spitzer/mips/'):

    footprints = {fn: wcs.WCS(fits.getheader(fn)) for fn in glob.glob(f"{mipspath}/MG[0-9][0-9][0-9][0-9][pn][0-9][0-9][0-9]_024.fits")}

    debug_counter = 0

    rows = []
    for crd in tqdm(coords.galactic):
        match = False
        for fn, ww in footprints.items():
            if ww.footprint_contains(crd):
                flx = get_flx(crd, fits.getdata(fn), ww)
                rows.append(flx)
                match = True
                break
        if not match:
            rows.append(np.nan)

    # use the last successful one
    units = fits.getheader(fn)['BUNIT']

    tbl.add_column(table.Column(name='M24_flux_uplim', data=rows, unit=units))

    return tbl

Herschel_Beams = {'70': np.pi*9.7*10.7*u.arcsec**2 / (8*np.log(2)),
                  '160': np.pi*13.2*13.9*u.arcsec**2 / (8*np.log(2)),
                  '250': np.pi*22.8*23.9*u.arcsec**2 / (8*np.log(2)),
                  '350': np.pi*29.3*31.3*u.arcsec**2 / (8*np.log(2)),
                  '500': np.pi*41.1*43.8*u.arcsec**2 / (8*np.log(2)),
                 }


if __name__ == "__main__":
  # fetch SPICY catalogue
  tbl, coords = get_spicy_tbl()
  # find which SPICY sources are in each ALMA FOV
  tbl = find_ALMAIMF_matches(tbl, coords)
  # reduce table to only the shared sources
  tblmsk = tbl['in_ALMAIMF']
  tbl, coords = tbl[tblmsk], coords[tblmsk]

  # mark rows by what NIR data is available
  has_ukidss = [row['UKIDSS'] != '                   ' for row in tbl]
  has_virac = [row['VIRAC'] is not np.ma.masked for row in tbl]
  tbl.add_column("      ",name='NIR data')
  tbl['NIR data'][has_ukidss] = "UKIDSS"
  tbl['NIR data'][has_virac] = "VIRAC"

  # append SPICY upper limits
  print("Adding SPICY upper limits")
  tbl = add_spicylimits(tbl)
  # append ALMA-IMF photometry
  print("Adding ALMA-IMF photometry")
  tbl = add_alma_photometry(tbl, band='b3', wlname='3mm')
  tbl = add_alma_photometry(tbl, band='b6', wlname='1mm')

  # convert ALMA-IMF fluxes to mJy/beam
  print("Converting ALMA fluxes to mJy/beam")
  for colname in ['ALMA-IMF_3mm_flux', 'ALMA-IMF_3mm_eflux', 'ALMA-IMF_1mm_flux', 'ALMA-IMF_1mm_eflux']:
      tbl[colname] = tbl[colname] * u.Jy / u.beam
      tbl[colname] = tbl[colname].to(u.mJy / u.beam)

  # add MIPS data points and upper limits
  print("Adding MIPS match data")
  tbl = add_MIPS_matches(tbl)
  print("Adding MIPS limit data")
  tbl = add_mips_limits(tbl, coords)
  tbl.sort('SPICY') # previous function messes row order

  # populate MIPS error column with upper limits, rename MIPS columns
  tbl['e_S24'][tbl['e_S24'].mask] = tbl['M24_flux_uplim'][tbl['e_S24'].mask]
  tbl.rename_column('S24', 'Spitzer/MIPS.24mu_flux')
  tbl.rename_column('e_S24', 'Spitzer/MIPS.24mu_eflux')

  # add VVV data points, populate errors with upper limits
  print("Adding VVV data")
  tbl = add_VVV_matches(tbl)
  tbl.rename_column('KsEll', 'Ksell') # so that add_VVV_limits works right
  print("Adding VVV upper limits")
  tbl = add_VVV_limits(tbl)

  # append UKIDSS data points for matches
  print("Adding UKIDSS data")
  tbl = add_UKIDSS_matches(tbl)

  # table column housekeeping
  tbl['Jmag_1'][tbl['NIR data'] == "UKIDSS"] = tbl['Jmag_2'][tbl['NIR data'] == "UKIDSS"] # move J data
  tbl.rename_column('Jmag_1', 'Jmag')
  tbl['Jell'][tbl['NIR data'] == "UKIDSS"] = tbl['e_Jmag'][tbl['NIR data'] == "UKIDSS"]
  tbl['Hmag_1'][tbl['NIR data'] == "UKIDSS"] = tbl['Hmag_2'][tbl['NIR data'] == "UKIDSS"] # move H data
  tbl.rename_column('Hmag_1', 'Hmag')
  tbl['Hell'][tbl['NIR data'] == "UKIDSS"] = tbl['e_Hmag'][tbl['NIR data'] == "UKIDSS"]
  tbl.rename_column('Kmag1', 'Kmag') # tweak K column
  tbl.rename_column('e_Kmag1', 'Kell')

  # populate UKIDSS errors with upper limits
  print("Adding UKIDSS upper limits")
  tbl = add_UKIDSS_limits(tbl)

  # append Herschel data points
  print("Adding Herschel limits")
  tbl = add_herschel_limits(tbl, coords)

  # all Herschel values will be treated as upper limits
  print("Converting Herschel fluxes to upper limits")
  tbl["Herschel/Pacs.blue_eflux"] = (tbl['70' ].quantity * u.pixel).to(u.mJy)
  tbl["Herschel/Pacs.red_eflux"]  = (tbl['160'].quantity * u.pixel).to(u.mJy)
  tbl["Herschel/SPIRE.PSW_eflux"] = (tbl['250'].quantity * Herschel_Beams['250']).to(u.mJy)
  tbl["Herschel/SPIRE.PMW_eflux"] = (tbl['350'].quantity * Herschel_Beams['350']).to(u.mJy)
  tbl["Herschel/SPIRE.PLW_eflux"] = (tbl['500'].quantity * Herschel_Beams['500']).to(u.mJy)
  for x in ['Pacs.blue','Pacs.red','SPIRE.PSW','SPIRE.PMW','SPIRE.PLW']:
      tbl[f"Herschel/{x}_flux"] = np.nan

  # housekeeping
  for errcolname in ['Zell','Yell','Jell','Hell','Kell','Ksell']:
      tbl[errcolname].unit = 'mag'

  # VIRAC mag-to-flux conversion
  # acquire filternames and zero points
  sed_filters, wavelength_dict, filternames, zpts = get_filters("south")
  # define magcols and emagcols, for VIRAC fields
  magcols = ['Ymag', 'Zmag', 'Jmag', 'Hmag', 'Ksmag','mag3_6', 'mag4_5', 'mag5_8', 'mag8_0']
  emagcols = ['Yell', 'Zell', 'Jell', 'Hell', 'Ksell','e_mag3_6', 'e_mag4_5', 'e_mag5_8', 'e_mag8_0']
  # convert magnitudes to fluxes
  print("Converting magnitudes to fluxes")
  tbl_virac = tbl[[n in virac_fields for n in tbl['ALMAIMF_FIELDID']]]
  tbl_virac = mag_to_flux(tbl_virac, magcols, emagcols, zpts, filternames)

  # UKIDSS mag-to-flux conversion
  # acquire filternames and zero points
  sed_filters, wavelength_dict, filternames, zpts = get_filters("north")
  # define magcols and emagcols, for UKIDSS fields
  magcols = ['Jmag', 'Hmag', 'Kmag','mag3_6', 'mag4_5', 'mag5_8', 'mag8_0']
  emagcols = ['Jell', 'Hell', 'Kell','e_mag3_6', 'e_mag4_5', 'e_mag5_8', 'e_mag8_0']
  # convert magnitudes to fluxes
  print("Converting magnitudes to fluxes")
  tbl_ukidss = tbl[[n in ukidss_fields for n in tbl['ALMAIMF_FIELDID']]]
  tbl_ukidss = mag_to_flux(tbl_ukidss, magcols, emagcols, zpts, filternames)

  # VIRAC mag-to-flux conversion
  # acquire filternames and zero points
  sed_filters, wavelength_dict, filternames, zpts = get_filters("south")
  # define magcols and emagcols, for VIRAC fields
  magcols = ['Ymag', 'Zmag', 'Jmag', 'Hmag', 'Ksmag','mag3_6', 'mag4_5', 'mag5_8', 'mag8_0']
  emagcols = ['Yell', 'Zell', 'Jell', 'Hell', 'Ksell','e_mag3_6', 'e_mag4_5', 'e_mag5_8', 'e_mag8_0']
  # convert magnitudes to fluxes
  print("Converting magnitudes to fluxes")
  tbl_virac = tbl[[n in virac_fields for n in tbl['ALMAIMF_FIELDID']]]
  tbl_virac = mag_to_flux(tbl_virac, magcols, emagcols, zpts, filternames)

  tbl = vstack([tbl_ukidss, tbl_virac])
  tbl.sort('SPICY')

  # add distances to each field, based on values in the ALMA-IMF paper
  print("Adding distances")
  distances = {"G10": 4.95,"G12": 2.4,"W43MM1": 5.5,"W43MM2": 5.5,"W43MM3": 5.5,"W51-E": 5.4,"W51IRS2": 5.4,"G338": 3.9,
               "G008": 3.4,"G327": 2.5,"G328": 2.5,"G333": 4.2,"G337": 2.7,"G351": 2.0,"G353": 2.0,}
  tbl.add_column(0.00*u.kpc,name='Distance')
  for key in distances:
      tbl['Distance'][tbl['ALMAIMF_FIELDID'] == key] = distances[key]

  # cut table down to only necessary information
  tbl = tbl['SPICY','ra','dec','l','b','ALMAIMF_FIELDID','Distance','NIR data',
            'Spitzer/IRAC.I1_flux','Spitzer/IRAC.I1_eflux','Spitzer/IRAC.I2_flux','Spitzer/IRAC.I2_eflux',
            'Spitzer/IRAC.I3_flux','Spitzer/IRAC.I3_eflux','Spitzer/IRAC.I4_flux','Spitzer/IRAC.I4_eflux',
            'ALMA-IMF_3mm_flux','ALMA-IMF_3mm_eflux','ALMA-IMF_1mm_flux','ALMA-IMF_1mm_eflux',
            'Spitzer/MIPS.24mu_flux','Spitzer/MIPS.24mu_eflux',
            'UKIRT/UKIDSS.J_flux','UKIRT/UKIDSS.J_eflux','UKIRT/UKIDSS.H_flux','UKIRT/UKIDSS.H_eflux','UKIRT/UKIDSS.K_flux','UKIRT/UKIDSS.K_eflux',
            'Paranal/VISTA.Ks_flux','Paranal/VISTA.Ks_eflux','Paranal/VISTA.Z_flux','Paranal/VISTA.Z_eflux','Paranal/VISTA.Y_flux','Paranal/VISTA.Y_eflux',
            'Paranal/VISTA.J_flux','Paranal/VISTA.J_eflux','Paranal/VISTA.H_flux','Paranal/VISTA.H_eflux',
            'Herschel/Pacs.blue_flux','Herschel/Pacs.blue_eflux','Herschel/Pacs.red_flux','Herschel/Pacs.red_eflux',
            'Herschel/SPIRE.PMW_flux','Herschel/SPIRE.PMW_eflux','Herschel/SPIRE.PSW_flux','Herschel/SPIRE.PSW_eflux','Herschel/SPIRE.PLW_flux','Herschel/SPIRE.PLW_eflux']
  tbl.meta['description'] = None

  # save table as individual fits files per field
  for fieldid in np.unique(tbl['ALMAIMF_FIELDID']):
      tbl[tbl['ALMAIMF_FIELDID'] == fieldid].write(f'/blue/adamginsburg/adamginsburg/SPICY_ALMAIMF/BriceTingle/Region_tables/Unfitted/{fieldid}', format='fits', overwrite=True)
