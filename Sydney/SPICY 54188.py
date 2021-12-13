%matplotlib inline
import pylab as pl
pl.style.use('dark_background')
pl.rcParams['font.size'] = 16
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.io import fits
from astropy.table import Table

tbl = Table.read('/blue/adamginsburg/adamginsburg/ALMA_IMF/SPICY_ALMAIMF/SPICY_withAddOns.fits')

tbl['ALMA-IMF_1mm_eflux'] = tbl['ALMA-IMF_1mm_eflux'].quantity.to(u.mJy/u.beam)
tbl['ALMA-IMF_3mm_eflux'] = tbl['ALMA-IMF_3mm_eflux'].quantity.to(u.mJy/u.beam)
tbl['ALMA-IMF_1mm_flux'] = tbl['ALMA-IMF_1mm_flux'].quantity.to(u.mJy/u.beam)
tbl['ALMA-IMF_3mm_flux'] = tbl['ALMA-IMF_3mm_flux'].quantity.to(u.mJy/u.beam)

import sys
sys.path.append('/orange/adamginsburg/ALMA_IMF/reduction/analysis/')
from spectralindex import prefixes

from spectral_cube import SpectralCube

import os
os.chdir('/orange/adamginsburg/ALMA_IMF/May2021Release')

prefixes['W43MM1'] = dict(
    finaliter_prefix_b3="W43-MM1/B3/cleanest/W43-MM1_B3_uid___A001_X1296_X1af_continuum_merged_12M_robust0_selfcal4_finaliter",
    finaliter_prefix_b6="W43-MM2/B6/cleanest/W43-MM2_B6_uid___A001_X1296_X113_continuum_merged_12M_robust0_selfcal5_finaliter",)

for fieldid, pfxs in prefixes.items():
    cube = SpectralCube.read(pfxs['finaliter_prefix_b3']+".image.tt0.fits", format='fits', use_dask=False).minimal_subcube()
    ww = cube.wcs.celestial
    ww._naxis = cube.shape[1:]
    matches = ww.footprint_contains(coords)
    print(f"{fieldid}: {matches.sum()}")
    
for fieldid, pfxs in prefixes.items():
    cube = SpectralCube.read(pfxs['finaliter_prefix_b3']+".image.tt0.fits", format='fits', use_dask=False).minimal_subcube()
    ww = cube.wcs.celestial
    ww._naxis = cube.shape[1:]
    matches = ww.footprint_contains(coords)
    print(f"{fieldid}: {matches.sum()}")
    print(tbl[matches][['SPICY','p1','class','UKIDSS','VIRAC']])
    print()
    
import sys
sys.path.append('/orange/adamginsburg/ALMA_IMF/reduction/analysis/')

import spitzer_plots
from spitzer_plots import show_fov_on_spitzer, contour_levels
from spectralindex import prefixes
from spectral_cube import SpectralCube

# data from my region (G351...)

fieldid = 'G351'
basepath = '/orange/adamginsburg/ALMA_IMF/May2021Release'
pfxs = prefixes[fieldid]
fig = show_fov_on_spitzer(**{key: f'{basepath}/{val}' for key,val in pfxs.items()},
                          fieldid=fieldid, spitzerpath='/orange/adamginsburg/ALMA_IMF/RestructuredImagingResults/spitzer_datapath',
                          contour_level=contour_levels[fieldid])


cube = SpectralCube.read(basepath + '/' + pfxs['finaliter_prefix_b3']+".image.tt0.fits",
                         format='fits', use_dask=False).minimal_subcube()
ww = cube.wcs.celestial
ww._naxis = cube.shape[1:]
matches = ww.footprint_contains(coords)

cc = coords[matches]

ax = fig.gca()
ax.plot(cc.fk5[11].ra.deg, cc.fk5[11].dec.deg,'*', mfc='none', mec='w', markersize=17, transform=ax.get_transform('fk5'), )


ax.axis('off')
fig.savefig(f'/home/spetz/figures/source14', dpi = 300)

from astroquery.vizier import Vizier
from astroquery.svo_fps import SvoFps

def getrow(tb, rownum, keys=['Ymag', 'Zmag', 'Jmag', 'Hmag', 'Ksmag','mag3_6', 'mag4_5', 'mag5_8', 'mag8_0',]):
    return np.array([tb[rownum][key] for key in keys])
    
magcols = ['Ymag', 'Zmag', 'Jmag', 'Hmag', 'Ksmag','mag3_6', 'mag4_5', 'mag5_8', 'mag8_0',]
emagcols = ['Yell', 'Zell', 'Jell', 'Hell', 'KsEll','e_mag3_6', 'e_mag4_5', 'e_mag5_8', 'e_mag8_0',]

# these are the official filternames on SVO_FPS
filternames = ['Paranal/VISTA.Y', 'Paranal/VISTA.Z', 'Paranal/VISTA.J', 'Paranal/VISTA.H', 'Paranal/VISTA.Ks',
               'Spitzer/IRAC.I1', 'Spitzer/IRAC.I2', 'Spitzer/IRAC.I3', 'Spitzer/IRAC.I4']
filter_meta = table.vstack([SvoFps.get_filter_list(facility='Paranal', instrument='VIRCAM'),
                            SvoFps.get_filter_list(facility='Spitzer', instrument='IRAC')])
zpts = {filtername: filter_meta[filter_meta['filterID']==filtername]['ZeroPoint'] for filtername in filternames}
wavelengths = [np.average(SvoFps.get_transmission_data(filtername)['Wavelength'],
                          weights=SvoFps.get_transmission_data(filtername)['Transmission'])
              for filtername in filternames]
              
sed = sed_tbl[magcols]
esed = sed_tbl[emagcols]
zps = np.array([zpts[fn] for fn in filternames], dtype='float').squeeze()*u.Jy


for rownum in range(len(sed)):
    flx = zps.value * 10**(getrow(sed, rownum)/-2.5)
    err = getrow(esed, rownum, emagcols) / 1.09 * flx
    pl.errorbar(np.array(wavelengths)/1e4, flx, yerr=err, marker='x', linestyle='none')
_=pl.loglog()
_=pl.gca().set_xticks([1,2,3,4,5])
_=pl.gca().set_xticklabels([1,2,3,4,5])
_=pl.xlabel("Wavelelength (microns)")
_=pl.ylabel("Flux (mJy)")


robitaille_modeldir = '/blue/adamginsburg/richardson.t/research/flux/robitaille_models/'

from sedfitter import fit, Fitter
from sedfitter.filter import Filter
from astroquery.svo_fps import SvoFps

# these are the official filternames on SVO_FPS
filternames = ['Paranal/VISTA.Y', 'Paranal/VISTA.Z', 'Paranal/VISTA.J', 'Paranal/VISTA.H', 'Paranal/VISTA.Ks',
               'Spitzer/IRAC.I1', 'Spitzer/IRAC.I2', 'Spitzer/IRAC.I3', 'Spitzer/IRAC.I4', 'Herschel/Pacs.blue',
             'Herschel/Pacs.red', 'Herschel/SPIRE.PSW','Herschel/SPIRE.PMW','Herschel/SPIRE.PLW', 'Spitzer/MIPS.24mu']
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


if not os.path.exists(f'{robitaille_modeldir}/s-pbhmi/convolved/Spitzer/IRAC.I1.fits'):
    from sedfitter.convolve import convolve_model_dir

    for model_dir in glob.glob(f'{robitaille_modeldir}/s*'):
        convolve_model_dir(model_dir, sed_filters)
        
def getrow(tb, rownum, keys):
    return np.array([tb[rownum][key] for key in keys])
        
        
# convert magnitudes to fluxes now
# (it's a pain to try to deal with a mix of magnitudes & fluxes)
    
sed_tbl['Zmag'][11] = 19.81
sed_tbl['Zell'][11] = 0.997

sed_tbl['Ymag'][11] = 18.988
sed_tbl['Yell'][11] = 0.997

sed_tbl['Jmag'][11] = 18.023
sed_tbl['Jell'][11] = 0.997

sed_tbl['Hmag'][11] = 16.844
sed_tbl['Hell'][11] = 0.997


for colname, errcolname, zpn in zip(magcols, emagcols, filternames):
    print(colname, zpn)
    zp = u.Quantity(zpts[zpn], u.Jy)
    data = sed_tbl[colname].value
    if hasattr(sed_tbl[colname], 'mask'):
        sed_tbl[zpn+"_flux"] = flx = np.ma.masked_where(sed_tbl[colname].mask, (zp * 10**(data/-2.5)).to(u.mJy))
    else:
        sed_tbl[zpn+"_flux"] = flx = (zp * 10**(data/-2.5)).to(u.mJy)
    err = sed_tbl[errcolname] / 1.09 * flx
    sed_tbl[zpn+"_eflux"] = err
    


sed_tbl['Paranal/VISTA.H_flux'][11] = np.nan
sed_tbl['Paranal/VISTA.H_eflux'][11] = 0.1879800488357045/3

sed_tbl['Paranal/VISTA.J_flux'][11] = np.nan
sed_tbl['Paranal/VISTA.J_eflux'][11] = 0.09579069/3

sed_tbl['Paranal/VISTA.Z_flux'][11] = np.nan
sed_tbl['Paranal/VISTA.Z_eflux'][11] = 0.026974663181402018/3

sed_tbl['Paranal/VISTA.Y_flux'][11] = np.nan
sed_tbl['Paranal/VISTA.Y_eflux'][11] = 0.052943212767020784/3

from sedfitter.sed import SEDCube
sedcube = SEDCube.read(f"{model_dir}/flux.fits",)

def get_geometry_fits(rownumber, geometry="s----mi"):

    # Define path to models
    model_dir = f'{robitaille_modeldir}/{geometry}'

    


    # make an extinction law
    from dust_extinction.parameter_averages import F19
    ext = F19(3.1)
    from astropy import units as u
    from sedfitter.extinction import Extinction
    from sedfitter.source import Source


    # https://arxiv.org/abs/0903.2057
    # 1.34 is from memory
    guyver2009_avtocol = (2.21e21 * u.cm**-2 * (1.34*u.Da)).to(u.g/u.cm**2)
    ext_wav = np.sort((np.geomspace(0.301, 8.699, 1000)/u.um).to(u.um, u.spectral()))
    ext_vals = ext.evaluate(ext_wav, Rv=3.1)
    extinction = Extinction()
    extinction.wav = ext_wav
    extinction.chi = ext_vals / guyver2009_avtocol

    # Define filters and apertures
    #filters = ['2J', '2H', '2K', 'I1', 'I2', 'I3', 'I4']
    #apertures = [3., 3., 3., 3., 3., 3., 3.] * u.arcsec
    filters = filternames+["ALMA-IMF_1mm", "ALMA-IMF_3mm"]
    apertures = np.array([2.83, 2.83, 2.83, 2.83, 2.83, 2.4, 2.4, 2.4, 2.4, 3, 3, 3, 3, 3, 6.35, 3, 3])*u.arcsec
    apnum = np.argmin(np.abs((0.9*u.arcsec * 2*u.kpc).to(u.au, u.dimensionless_angles()) - sedcube.apertures))


    source = Source()
    # wavelengths, getrow(sed, rownum), yerr=getrow(esed, rownum, emagcols),
    rownum = 11

    flx = getrow(sed_tbl, rownum, keys=[key+"_flux" for key in filters])
    error = getrow(sed_tbl, rownum, keys=[key+"_eflux" for key in filters])
    valid = np.zeros(flx.size, dtype='int')
    valid[(np.isfinite(flx) & np.isfinite(error))] = 1
    valid[(~np.isfinite(flx) & ~np.isfinite(error))] = 0
    valid[(~np.isfinite(flx) & np.isfinite(error))] = 3

    flx[valid == 3] = error[valid == 3] * 3
    error[valid == 3] = 0.997

    source.valid = valid#[valid]
    source.flux = flx    
    source.error =  error

    fitter = Fitter(filter_names=np.array(filters),#[valid],
                    apertures=apertures,#[valid],
                    model_dir=model_dir,
                    extinction_law=extinction,
                    distance_range=[2.0,2.4]*u.kpc,
                    av_range=[4,20],
                   )
    

    # Run the fitting
    fitinfo = fitinfo = fitter.fit(source)
    
    return fitinfo
    
    
geometries = ['spubsmi','sp--smi', 'sp--hmi', 'sp--s-i', 's-ubsmi', 'spu-hmi']

bestfits = {geom: get_geometry_fits(11, geometry=geom) for geom in geometries}

wavelengths = u.Quantity([wavelength_dict[fn] for fn in filters], u.um)
pl.errorbar(wavelengths.value[valid==1], source.flux[valid==1], yerr=source.error[valid==1], linestyle='none', color='w', marker='o')
pl.plot(wavelengths.value[valid==3], source.flux[valid==3], linestyle='none', color='w', marker='v')

apnum = np.argmin(np.abs((0.9*u.arcsec * 2.2*u.kpc).to(u.au, u.dimensionless_angles()) - sedcube.apertures))

for geom in geometries:
    fitinfo=bestfits[geom]
    _=pl.plot(wavelengths.value,
          (10**fitinfo.model_fluxes[np.nanargmin(fitinfo.chi2), :].T),
          alpha=0.9,label=geom)
_=pl.loglog()
_=pl.xlabel("Wavelength (microns)")
_=pl.ylabel("Flux (mJy)")
_=pl.legend(bbox_to_anchor=(1.0,1.04))

{geom: np.nanmin(fitinfo.chi2) for geom, fitinfo in bestfits.items()}

pl.figure(figsize=(12,12))

for geom in geometries:
    fitinfo=bestfits[geom]
    
    model_dir = f'{robitaille_modeldir}/{geom}'
    sedcube = sedcube.read(f"{model_dir}/flux.fits",)
    
    index = np.nanargmin(fitinfo.chi2)
    modelname = fitinfo.model_name[index]
    sed = sedcube.get_sed(modelname)
    _=pl.plot(sedcube.wav, sed.flux[apnum] * fitinfo.sc[index] * 10 ** (fitinfo.av[index] * extinction.get_av(sed.wav)),label=geom, alpha=0.9)
   # _=pl.plot(wavelengths.value,



wavelengths = u.Quantity([wavelength_dict[fn] for fn in filters], u.um)
pl.errorbar(wavelengths.value[valid==1], source.flux[valid==1], yerr=source.error[valid==1], linestyle='none', color='w', marker='o')
pl.plot(wavelengths.value[valid==3], source.flux[valid==3], linestyle='none', color='w', marker='v')



apnum = np.argmin(np.abs((0.9*u.arcsec * 2.2*u.kpc).to(u.au, u.dimensionless_angles()) - sedcube.apertures))
distance = 2.2*u.kpc

_=pl.loglog()
_=pl.xlabel("Wavelength (microns)")
_=pl.ylabel("Flux (mJy)")
_=pl.ylim(5e-4,1e4)
_=pl.legend(bbox_to_anchor=(1.0,1.04))



from astropy import units as u
from sedfitter.extinction import Extinction
from sedfitter.source import Source
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

chi2lim = 100
selection = fitinfo.chi2 < np.nanmin(fitinfo.chi2) + chi2lim

def datafunction(geom, chi2lim, bestfits):
    pars = Table.read(f'/blue/adamginsburg/richardson.t/research/flux/pars/{geom}_augmented.fits')
    fitinfo = bestfits[geom]
    selection = fitinfo.chi2 < np.nanmin(fitinfo.chi2) + chi2lim
    data = pars[fitinfo.model_id[selection]]
    return pars, data
    
def binsfunction(param, kind, binsnum, chi2lim, geometries, bestfits, massnum=9):
    datamin = []
    datamax = []
    for geom in geometries:
        pars, data = datafunction(geom, chi2lim, bestfits)
        if param in pars.keys():
            if param == "Line-of-Sight Masses":
                dataparam = data[param]
                datamin.append(dataparam[massnum].min())
                datamax.append(dataparam[massnum].max())
            if param == "Sphere Masses":
                dataparam = data[param]
                datamin.append(dataparam[massnum].min())
                datamax.append(dataparam[massnum].max())
            else:
                datamin.append(data[param].min())
                datamax.append(data[param].max())

    for x in datamin:
        if x == 0:
            datamin.remove(x)
            
    for x in datamax:
        if x == 0:
            datamax.remove(x)
    
    if kind is 'log':
        binsmin = np.log10(min(datamin))
        binsmax = np.log10(max(datamax))
        bins = np.logspace(binsmin, binsmax, binsnum)
    
    if kind is 'lin':
        binsmin = min(datamin)
        binsmax = max(datamax)
        bins = np.linspace(binsmin, binsmax, binsnum)

    if kind is 'geom':
        binsmin = min(datamin)
        binsmax = max(datamax)
        bins = np.geomspace(binsmin, binsmax, binsnum)
        
    if kind is 'dist':
        binsmin = min(10**fitinfo.sc[selection])
        binsmax = max(10**fitinfo.sc[selection])
        bins = np.geomspace(binsmin, binsmax, binsnum)
        
    if kind is 'ext':
        binsmin = min(fitinfo.av[selection])
        binsmax = max(fitinfo.av[selection])
        bins = np.geomspace(binsmin, binsmax, binsnum)

    return bins
    
basefig = pl.figure(figsize=(20, 22))
gs = GridSpec(nrows=6, ncols=2, height_ratios=[4,1,1,1,1,1], hspace=0.25, wspace=0.1)

# --------------------------------

# Best fits plot
ax0 = basefig.add_subplot(gs[0, 1])
wavelengths = u.Quantity([wavelength_dict[fn] for fn in filters], u.um)
pl.errorbar(wavelengths.value[valid==1], source.flux[valid==1], yerr=source.error[valid==1], linestyle='none', color='w', marker='o')
pl.plot(wavelengths.value[valid==3], source.flux[valid==3], linestyle='none', color='w', marker='v')

distance = 2.2*u.kpc

for geom in geometries:
    global sedcube
    
    apnum = np.argmin(np.abs((0.9*u.arcsec * 2.2*u.kpc).to(u.au, u.dimensionless_angles()) - sedcube.apertures))
    fitinfo = bestfits[geom]
    
    model_dir = f'{robitaille_modeldir}/{geom}'
    sedcube = SEDCube.read(f"{model_dir}/flux.fits",)
    
    index = np.nanargmin(fitinfo.chi2)
    modelname = fitinfo.model_name[index]
    sed = sedcube.get_sed(modelname)
    _=ax0.plot(sedcube.wav,
             sed.flux[apnum] * fitinfo.sc[index] * 10**(fitinfo.av[index] * extinction.get_av(sed.wav)),
             label=geom, alpha=0.9)
    
ax0.loglog()
ax0.set_xlabel('Wavelength (microns)')
ax0.set_ylabel("Flux (mJy)")
ax0.set_xlim(0.5,1e4)
ax0.set_ylim(5e-4,3e4)

# --------------------------------

# ax1 = stellar temperature
# ax2 = model luminosity

# ax3 = stellar radius
# ax4 = line-of-sight mass

# ax5 = disk mass
# ax6 = sphere mass

ax1 = basefig.add_subplot(gs[1, 0])
ax2 = basefig.add_subplot(gs[1, 1])
ax3 = basefig.add_subplot(gs[2, 0])
ax4 = basefig.add_subplot(gs[2, 1])
ax5 = basefig.add_subplot(gs[3, 0])
ax6 = basefig.add_subplot(gs[3, 1])
ax7 = basefig.add_subplot(gs[4, 0])
ax8 = basefig.add_subplot(gs[4, 1])

histalpha = 0.8
lognum = 50
linnum = 50

tempbins = binsfunction('star.temperature', 'lin', linnum, chi2lim, geometries, bestfits)
lumbins = binsfunction('Model Luminosity', 'log', lognum, chi2lim, geometries, bestfits)
radbins = binsfunction('star.radius', 'log', lognum, chi2lim, geometries, bestfits)
losbins = binsfunction('Line-of-Sight Masses', 'log', 20, chi2lim, geometries, bestfits, 1)
dscbins = binsfunction('disk.mass', 'log', lognum, chi2lim, geometries, bestfits)
sphbins = binsfunction('Sphere Masses', 'log', lognum, chi2lim, geometries, bestfits)
distbins = binsfunction(10**fitinfo.sc[selection], 'dist', linnum, chi2lim, geometries, bestfits)
extbins = binsfunction(fitinfo.av[selection], 'ext', linnum, chi2lim, geometries, bestfits)


for geom in geometries:
    pars, data = datafunction(geom, chi2lim, bestfits)

    if 'star.temperature' in pars.keys():
        ax1.hist(data['star.temperature'], bins=tempbins, alpha=histalpha, label=geom)

    if 'Model Luminosity' in pars.keys():
        ax2.hist(data['Model Luminosity'], bins=lumbins, alpha=histalpha, label=geom)

    if 'star.radius' in pars.keys():
        ax3.hist(data['star.radius'], bins=radbins, alpha=histalpha, label=geom)

    if 'Line-of-Sight Masses' in pars.keys():
        ax4.hist(data['Line-of-Sight Masses'][:,1], bins=losbins, alpha=histalpha, label=geom)
        ax4.axvline(0.17334459, color='r', linestyle='dashed', linewidth=3)
        
    if 'disk.mass' in pars.keys():
        ax5.hist(data['disk.mass'], bins=dscbins, alpha=histalpha, label=geom)
        ax5.axvline(0.17334459, color='r', linestyle='dashed', linewidth=3)
        
    if 'Sphere Masses' in pars.keys():
        ax6.hist(data['Sphere Masses'][:,1], bins=sphbins, alpha=histalpha, label=geom)
        ax6.axvline(0.17334459, color='r', linestyle='dashed', linewidth=3)
        
ax7.hist(10**fitinfo.sc[selection], bins=distbins, alpha=histalpha, label=geom)
ax8.hist(fitinfo.av[selection], bins=extbins, alpha=histalpha, label=geom)

handles, labels = ax1.get_legend_handles_labels()
ax0.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.16,1.02))
ax1.set_xlabel("Stellar Temperature (K)")
ax2.set_xlabel("Stellar Luminosity (L$_\odot$)")
ax3.set_xlabel("Stellar Radius (R$_\odot$)")
ax4.set_xlabel("Line-of-Sight Masses (M$_\odot$)")
ax5.set_xlabel("Disk Mass (M$_\odot$)")
ax6.set_xlabel("Sphere Mass (M$_\odot$)")
ax7.set_xlabel("Distance (kpc)")
ax8.set_xlabel("Extinction (mag)")

_=ax2.semilogx()
_=ax3.semilogx()
_=ax4.semilogx()
_=ax5.semilogx()
_=ax6.semilogx()

# --------------------------------

# reading the saved image of the region with source location marked
locfig = mpimg.imread(f'/home/spetz/figures/source14.png')

# my image needs to be flipped
locfig = np.flipud(locfig)

ax9 = basefig.add_subplot(gs[0, 0])
ax9.imshow(locfig)
ttl = ax9.set_title(f'\n{fieldid}  |  SPICY {54188} | 0.17334459 M⊙\n', fontsize=25)
ttl.set_position([.5, 1])
#ax9.axis([90,630,90,630])
ax9.axis([450,2400,450,2400])
ax9.axis('off')



from astropy.stats import mad_std

star_temp_mean = np.nanmean(np.log10(data['star.temperature']))
model_lum_mean = np.nanmean(np.log10(data['Model Luminosity']))
star_rad_mean = np.nanmean(np.log10(data['star.radius']))
los_mass_mean = np.nanmean(np.log10(data['Line-of-Sight Masses'][:,1]))
disk_mass_mean = np.nanmean(np.log10(data['disk.mass']))
sphere_mass_mean = np.nanmean(np.log10(data['Sphere Masses'][:,1]))
distance_mean = np.nanmean(np.log10(10**fitinfo.sc[selection]))
ext_mean = np.nanmean(np.log10(fitinfo.av[selection]))

star_temp_std = np.nanstd(np.log10(data['star.temperature']))
model_lum_std = np.nanstd(np.log10(data['Model Luminosity']))
star_rad_std = np.nanstd(np.log10(data['star.radius']))
los_mass_std = np.nanstd(np.log10(data['Line-of-Sight Masses'][:,1]))
disk_mass_std = np.nanstd(np.log10(data['disk.mass']))
sphere_mass_std = np.nanstd(np.log10(data['Sphere Masses'][:,1]))
distance_std = np.nanstd(np.log10(10**fitinfo.sc[selection]))
ext_std = np.nanstd(np.log10(fitinfo.av[selection]))

star_temp_med = np.nanmedian(np.log10(data['star.temperature']))
model_lum_med = np.nanmedian(np.log10(data['Model Luminosity']))
star_rad_med = np.nanmedian(np.log10(data['star.radius']))
los_mass_med = np.nanmedian(np.log10(data['Line-of-Sight Masses'][:,1]))
disk_mass_med = np.nanmedian(np.log10(data['disk.mass']))
sphere_mass_med = np.nanmedian(np.log10(data['Sphere Masses'][:,1]))
distance_med = np.nanmedian(np.log10(10**fitinfo.sc[selection]))
ext_med = np.nanmedian(np.log10(fitinfo.av[selection]))

star_temp_mad = mad_std(np.log10(data['star.temperature']))
model_lum_mad = mad_std(np.log10(data['Model Luminosity']))
star_rad_mad = mad_std(np.log10(data['star.radius']))
los_mass_mad = mad_std(np.log10(data['Line-of-Sight Masses'][:,1]))
disk_mass_mad = mad_std(np.log10(data['disk.mass']))
sphere_mass_mad = mad_std(np.log10(data['Sphere Masses'][:,1]))
distance_mad = mad_std(np.log10(10**fitinfo.sc[selection]))
ext_mad = mad_std(np.log10(fitinfo.av[selection]))

PARAMETERS = ['star.temperature', 'Model Luminosity', 'star.radius', 'Line-of-Sight Masses', 'disk.mass', 'Sphere Masses',
             'distance', 'extinction']
STD = [star_temp_std, model_lum_std, star_rad_std, los_mass_std, disk_mass_std, sphere_mass_std, distance_std, ext_std]
MEAN = [star_temp_mean, model_lum_mean, star_rad_mean, los_mass_mean, disk_mass_mean, sphere_mass_mean, distance_mean,
       ext_mean]
Median = [star_temp_med, model_lum_med, star_rad_med, los_mass_med, disk_mass_med, sphere_mass_med, distance_med,
       ext_med]
MAD = [star_temp_mad, model_lum_mad, star_rad_mad, los_mass_mad, disk_mass_mad, sphere_mass_mad, distance_mad,
       ext_mad]

parameter_table = Table([PARAMETERS, MEAN, STD, Median, MAD],
             names=('Parameters', 'Mean', 'Standard Deviation','Median','Median Absolute Deviation'),
             meta={'name': 'SPICY 54188'})

parameter_table


from astropy.modeling.models import BlackBody
sed_tbl[11]

(((0.4857932886640101*3)*u.mJy * (2.2*u.kpc)**2) / (0.008*u.cm**2/u.g * BlackBody(20*u.K)(230*u.GHz) * u.sr)).to(u.M_sun)

