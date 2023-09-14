import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

from astropy import units as u
import sedfitter
from sedfitter.sed import SEDCube
from astropy.table import Table
from astropy.modeling.models import BlackBody

import table_loading

geometries = ['s-pbhmi', 's-pbsmi',
              'sp--h-i', 's-p-hmi', 
              'sp--hmi', 'sp--s-i', 
              's-p-smi', 'sp--smi', 
              'spubhmi', 'spubsmi', 
              'spu-hmi', 'spu-smi', 
              's---s-i', 's---smi', 
              's-ubhmi', 's-ubsmi', 
              's-u-hmi', 's-u-smi']

distances = {
    "W51-E": 5.4,
    "W43MM1": 5.5,
    "G333": 4.2,
    "W51IRS2": 5.4,
    "G338": 3.9,
    "G10": 4.95,
    "W43MM2": 5.5,
    "G008": 3.4,
    "G12": 2.4,
    "G327": 2.5,
    "W43MM3": 5.5,
    "G351": 2.0,
    "G353": 2.0,
    "G337": 2.7,
    "G328": 2.5,
}

def find_mass_ul(tbl, row_num, regiondistance):
<<<<<<< HEAD
    if not np.isnan(tbl[row_num]['ALMA-IMF_1mm_flux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_1mm_flux']):
        alma_detect = tbl[row_num]['ALMA-IMF_1mm_flux']
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.008*u.cm**2/u.g * BlackBody(20*u.K)(230*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(tbl[row_num]['ALMA-IMF_3mm_flux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_3mm_flux']):
        alma_detect = tbl[row_num]['ALMA-IMF_3mm_flux']
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.002*u.cm**2/u.g * BlackBody(20*u.K)(100*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(tbl[row_num]['ALMA-IMF_1mm_eflux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_1mm_eflux']):
        alma_detect = tbl[row_num]['ALMA-IMF_1mm_eflux']
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.008*u.cm**2/u.g * BlackBody(20*u.K)(230*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(tbl[row_num]['ALMA-IMF_3mm_eflux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_3mm_eflux']):
        alma_detect = tbl[row_num]['ALMA-IMF_3mm_eflux']
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.002*u.cm**2/u.g * BlackBody(20*u.K)(100*u.GHz) * u.sr)).to(u.M_sun)

=======
    if not np.isnan(tbl[row_num]['ALMA-IMF_1mm_flux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_1mm_flux']): 
        alma_detect = tbl[row_num]['ALMA-IMF_1mm_flux'].quantity * u.beam
        mass_ul = (((alma_detect) * (regiondistance*u.kpc)**2) / (0.008*u.cm**2/u.g * BlackBody(20*u.K)(230*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(tbl[row_num]['ALMA-IMF_3mm_flux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_3mm_flux']): 
        alma_detect = tbl[row_num]['ALMA-IMF_3mm_flux'].quantity * u.beam
        mass_ul = (((alma_detect) * (regiondistance*u.kpc)**2) / (0.002*u.cm**2/u.g * BlackBody(20*u.K)(100*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(tbl[row_num]['ALMA-IMF_1mm_eflux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_1mm_eflux']): 
        alma_detect = tbl[row_num]['ALMA-IMF_1mm_eflux'].quantity * u.beam
        mass_ul = (((alma_detect) * (regiondistance*u.kpc)**2) / (0.008*u.cm**2/u.g * BlackBody(20*u.K)(230*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(tbl[row_num]['ALMA-IMF_3mm_eflux']) and not np.ma.isMA(tbl[row_num]['ALMA-IMF_3mm_eflux']): 
        alma_detect = tbl[row_num]['ALMA-IMF_3mm_eflux'].quantity * u.beam
        mass_ul = (((alma_detect) * (regiondistance*u.kpc)**2) / (0.002*u.cm**2/u.g * BlackBody(20*u.K)(100*u.GHz) * u.sr)).to(u.M_sun)
    else:
        mass_ul = np.nan
        
>>>>>>> 71b75f64a174bd03d60b786c882cae7baefa3c77
    #230 for 1mm, 100 for 3mm

    return(mass_ul)

def get_okgeo(fits,chi2limit=3,show=True):
    okgeo = []

    for geom in geometries:
        # we impose an _absolute_ chi^2 limit (the fitter uses a _relative_, delta chi2 limit)
        if show:
            print(f"{geom}: {np.nanmin(fits[geom].chi2):12.1f}")
        if np.nanmin(fits[geom].chi2) < chi2limit:
            okgeo.append(geom)
    if show:
        print(okgeo)
        
    return okgeo

def get_modelcount(fits,okgeo,chi2limit=3):
    modelcount = 0
    for geom in okgeo:
        modelcount = (fits[geom].chi2 < chi2limit).sum()
    return modelcount

def get_chi2limit(fits):
    chi2min = np.nanmin([np.nanmin(fits[geom].chi2) for geom in geometries])
    chi2limit = chi2min*3
    if chi2limit < 3:
        chi2limit = 3
    return chi2limit, chi2min


def datafunction(geom, chi2limit, bestfits, min_chi2=None):
    pars = Table.read(f'/blue/adamginsburg/richardson.t/research/flux/pars/{geom}_augmented.fits')
    fitinfo = bestfits[geom]
    if min_chi2 is None:
        min_chi2 = np.nanmin(fitinfo.chi2)
    selection = fitinfo.chi2 < chi2limit
    data = pars[fitinfo.model_id[selection]]
    return pars, data, selection

def binsfunction(param, kind, binsnum, chi2limit, geometries, bestfits, massnum=9, min_chi2=None):
    # note: the massnum indicates an index for aperture size, and is used in the
    # parameters which involve multiple aperture sizes to select just one. you'll
    # need to find out what your massnum= is if you use this.

    datamin = []
    datamax = []
    for geom in geometries:
        pars, data, selection = datafunction(geom, chi2limit, bestfits, min_chi2=min_chi2)
        if param in pars.keys():
            if param == "Line-of-Sight Masses":
                dataparam = data[param]
                datamin.append(dataparam[massnum].min())
                datamax.append(dataparam[massnum].max())
            elif param == "Sphere Masses":
                dataparam = data[param]
                datamin.append(dataparam[massnum].min())
                datamax.append(dataparam[massnum].max())
            else:
                datamin.append(data[param].min())
                datamax.append(data[param].max())

    # just some idiot-proofing because i ran into a problem with this
    datamin = np.array(datamin)
    datamin = datamin[datamin>0]
    datamax = np.array(datamax)
    datamax = datamax[datamax>0]

    if len(datamin) == 0:
        return

    if kind == 'log':
        binsmin = np.log10(min(datamin))
        binsmax = np.log10(max(datamax))
        bins = np.logspace(binsmin, binsmax, binsnum)

    if kind == 'lin':
        binsmin = min(datamin)
        binsmax = max(datamax)
        bins = np.linspace(binsmin, binsmax, binsnum)

    if kind == 'geom':
        binsmin = min(datamin)
        binsmax = max(datamax)
        bins = np.geomspace(binsmin, binsmax, binsnum)


    if np.any(np.isnan(bins)):
        raise ValueError('found a nan')

    return bins

def plot_fit(fieldid, spicyid, fits, okgeo=geometries, chi2limit=3, min_chi2=None,
             modelcount=None, show_all_models=False, alpha_allmodels=None,
             default_aperture=3000*u.au, show_per_aperture=True,
             extinction=make_extinction(), extinction_range=[0,60],
             robitaille_modeldir='/blue/adamginsburg/richardson.t/research/flux/robitaille_models-1.2',
             loc_imagedir='/blue/adamginsburg/adamginsburg/SPICY_ALMAIMF/BriceTingle/Location_figures'):

    """
    Parameters
    ----------
    fieldid : string
        'G328' (ex. - whatever your region is)
    spicyid : number
        31415 (ex. - whatever source you're looking at)
    fits : dict
        contains 18 sedfitter.fit_info.FitInfo objects, labeled per geometry
    okgeo : list
        contains strings (the labels of the best-fit geometries)
    wavelength_dict : dict
        entry ex. "'UKIRT/UKIDSS.J': <Quantity 12510.1752769 Angstrom>"
    chi2limit : number
        chi2 value to serve as upper bound for limiting models shown
    min_chi2 : number
        chi2 value to serve as lower bound for limiting models shown. 
        if None, min_chi2 will be recalculated for each geometry
    modelcount : number
        3525 (ex. - the number of 'good fit' models being incorporated into the plot)
    show_all_models : bool
        whether or not to show every model on the SED plot,
        instead of only the best fit from each geom
    alpha_allmodels : number
        override the transparency of the SED models shown
    default_aperture : Quantity
        3000*u.au (ex. - default aperture size)
    show_per_aperture : bool
        whether or not to show per aperture
    extinction : sedfitter.extinction.extinction.Extinction
        created with make_extinction()
    extinction_range : array containing two numbers
        [0,40] (ex. - the presumed lower and upper bounds on extinction)
    robitaille_modeldir : string
        filepath to the Robitaille models 
    loc_imagedir : string
        filepath to the location images (This should be a single folder containing
        the location images for all sources to be fit, with the naming scheme of
        "[fieldid]_[spicyid].png". plot_fit won't break if the image is missing.
    """
    
    # --------------------------------
    # Set up plot surface
    # --------------------------------

    basefig = plt.figure(figsize=(20, 22))
    gs = GridSpec(nrows=6, ncols=2, height_ratios=[4,1,1,1,1,1], hspace=0.35, wspace=0.1)
    plt.rcParams.update({'font.size': 20})

    # --------------------------------
    # Top-right: Best fits plot
    # --------------------------------
    
    ax0 = basefig.add_subplot(gs[0, 1])
    
    # gather some information consistent across all geoms
    fitinfo = fits[okgeo[0]]
    source = fitinfo.source
    valid = source.valid
    
    if fieldid in ['G10','G12','W43MM1','W43MM2','W43MM3','W51-E','W51IRS2']:
        sed_filters, wavelength_dict, filternames, zpts = get_filters("north")
    elif fieldid in ['G008','G327','G328','G333','G337','G338','G351','G353']:
        sed_filters, wavelength_dict, filternames, zpts = get_filters("south")
    
    filters=filternames+["ALMA-IMF_1mm", "ALMA-IMF_3mm"]
    wavelengths = u.Quantity([wavelength_dict[fn] for fn in filters], u.um)
    apertures = u.Quantity([x['aperture_arcsec'] for x in fitinfo.meta.filters], u.arcsec)
    
    #distance = (10**fitinfo.sc * u.kpc).mean()

    # preserve this parameter before loop
    recalc_min_chi2 = min_chi2 is None
    
    # store colors per geometry
    colors = {}
    
    if show_all_models and alpha_allmodels is None:
        if modelcount <= 50:
            alpha_allmodels = 0.5
        if 50 < modelcount <= 100:
            alpha_allmodels = 0.4
        if 100 < modelcount <= 1000:
            alpha_allmodels = 0.3
        if 1000 < modelcount <= 2000:
            alpha_allmodels = 0.1
        if 2000 < modelcount:
            alpha_allmodels = 0.05

    # loop over all 'good' geometries to display SED models:
    for geom in okgeo:

        fitinfo = fits[geom]
        model_dir = f'{robitaille_modeldir}/{geom}'
        sedcube = SEDCube.read(f"{model_dir}/flux.fits",)
        index = np.nanargmin(fitinfo.chi2)
        distance = (10**fitinfo.sc[index] * u.kpc)
        modelname = fitinfo.model_name[index]
        sed = sedcube.get_sed(modelname)
        apnum = np.argmin(np.abs(default_aperture - sedcube.apertures))
        # https://github.com/astrofrog/sedfitter/blob/41dee15bdd069132b7c2fc0f71c4e2741194c83e/sedfitter/sed/sed.py#L64
        distance_scale = (1*u.kpc/distance)**2
        av_scale = 10**((fitinfo.av[index] * extinction.get_av(sed.wav)))

        line, = ax0.plot(sedcube.wav,
                 sed.flux[apnum] * distance_scale * av_scale,
                 label=geom, alpha=0.9)
        
        colors[geom] = line.get_color()

        if recalc_min_chi2:
            min_chi2 = np.nanmin(fitinfo.chi2)
        indices = fitinfo.chi2 < chi2limit

        if show_all_models and any(indices):
            dist_scs = ((1*u.kpc)/(10**fitinfo.sc[indices] * u.kpc))**2
            mods = np.array([sedcube.get_sed(modelname).flux[apnum] for modelname in fitinfo.model_name[indices]])
            av_scales = 10**((fitinfo.av[indices][:,None] * extinction.get_av(sed.wav)[None,:]))

            lines = ax0.plot(sedcube.wav,
                             (mods * dist_scs[:,None] * av_scales).T,
                             alpha=alpha_allmodels,
                             c=line.get_color())

        if show_per_aperture:
            apnums = np.array([
                np.argmin(np.abs((apsize * distance).to(u.au, u.dimensionless_angles()) - sedcube.apertures))
                for apsize in apertures])
            wlids = np.array([
                np.argmin(np.abs(ww - sedcube.wav)) for ww in wavelengths])
            flux = np.array([sed.flux[apn, wavid].value for apn, wavid in zip(apnums, wlids)])

            av_scale_conv = 10**((fitinfo.av[index] * extinction.get_av(wavelengths)))
            flux = flux * distance_scale * av_scale_conv
            ax0.scatter(wavelengths, flux, marker='s', s=apertures.value, c=line.get_color())
    
    ax0.errorbar(wavelengths.value[valid==1], source.flux[valid==1], yerr=source.error[valid==1], linestyle='none', color='black', marker='o', markersize=10)
    ax0.plot(wavelengths.value[valid==3], source.flux[valid==3], linestyle='none', color='black', marker='v', markersize=10)
    
    if recalc_min_chi2:
        min_chi2 = None
            
    ax0.loglog()
    ax0.set_xlabel('Wavelength (microns)')
    ax0.set_ylabel("Flux (mJy)")
    ax0.set_xlim(0.5,1e4)
    ax0.set_ylim(5e-4,3e6)
    
    # --------------------------------
    # Bottom: Parameter histograms
    # --------------------------------
    
    histogram_alpha = 0.8
    
    # stellar temperature
    ax1 = basefig.add_subplot(gs[1, 0])
    ax1.set_xlabel("Stellar Temperature (K)")
    temperature_bins = np.linspace(2000, 30000, 50)
    
    # model luminosity
    ax2 = basefig.add_subplot(gs[1, 1])
    ax2.set_xlabel("Stellar Luminosity (L$_\odot$)")
    luminosity_bins = np.logspace(-4,7,100)
    _=ax2.semilogx()
    
    # stellar radius
    ax3 = basefig.add_subplot(gs[2, 0])
    ax3.set_xlabel("Stellar Radius (R$_\odot$)")
    radius_bins = np.logspace(-1, 3, 50)
    _=ax3.semilogx()

    # line-of-sight mass
    ax4 = basefig.add_subplot(gs[2, 1])
    ax4.set_xlabel("Line-of-Sight Masses (M$_\odot$)")
    los_bins = np.logspace(-4,10,100)
    _=ax4.semilogx()
    
    # disk mass
    ax5 = basefig.add_subplot(gs[3, 0])
    ax5.set_xlabel("Disk Mass (M$_\odot$)")
    disk_bins = np.logspace(-4,10,100)
    _=ax5.semilogx()
    
    # sphere mass
    ax6 = basefig.add_subplot(gs[3, 1])
    ax6.set_xlabel("Sphere Mass (M$_\odot$)")
    sphere_bins = np.logspace(-4,10,100)
    _=ax6.semilogx()
    
    ax7 = basefig.add_subplot(gs[4, 0])
    
    ax8 = basefig.add_subplot(gs[4, 1])
    for geom in okgeo:
        pars, data, selection = datafunction(geom, chi2limit, fits, min_chi2=min_chi2, avdatarange=extinction_range)

        if 'star.temperature' in pars.keys():
            ax1.hist(data['star.temperature'], bins=temperature_bins, alpha=histogram_alpha, label=geom, color=colors[geom])
        if 'Model Luminosity' in pars.keys():
            ax2.hist(data['Model Luminosity'], bins=luminosity_bins, alpha=histogram_alpha, label=geom, color=colors[geom])
        if 'star.radius' in pars.keys():
            ax3.hist(data['star.radius'], bins=radius_bins, alpha=histogram_alpha, label=geom, color=colors[geom])
        if 'Line-of-Sight Masses' in pars.keys():
            try:
                ax4.hist(data['Line-of-Sight Masses'][:,apnum], bins=los_bins, alpha=histogram_alpha, label=geom, color=colors[geom])
            except ValueError:
                print("ValueError while plotting LOS mass for ",geom)
        if 'disk.mass' in pars.keys():
            try:
                ax5.hist(data['disk.mass'], bins=disk_bins, alpha=histogram_alpha, label=geom, color=colors[geom])
            except ValueError:
                print("ValueError while plotting disk mass for ",geom)
        if 'Sphere Masses' in pars.keys():
            try:
                ax6.hist(data['Sphere Masses'][:,apnum], bins=sphere_bins, alpha=histogram_alpha, label=geom, color=colors[geom])
            except ValueError:
                print("ValueError while plotting sphere mass for ",geom)
        if not np.isnan(mass_ul):
            for axis in [ax4,ax5,ax6]:
                axis.axvline(mass_ul/u.M_sun, color='r', linestyle='dashed', linewidth=3)
                
        distances = 10**fits[geom].sc
        ax7.hist(distances[selection], bins=np.linspace(distances[selection].min(), distances[selection].max()), color=colors[geom])
        ax8.hist(fits[geom].av[selection], bins=np.linspace(extinction_range[0], extinction_range[1]), color=colors[geom])
    
        loc_imagepath = f'{loc_imagedir}/{fieldid}_{spicyid}.png'
        if os.path.exists(loc_imagepath):
            loc_image = mpimg.imread(loc_imagepath)
            loc_image = np.flipud(loc_image)
            ax9 = basefig.add_subplot(gs[0, 0])
            ax9.imshow(loc_image)
            ttl = ax9.set_title(f'\n{fieldid}  |  SPICY {spicyid} | {modelcount} models\n', fontsize=25)
            #ttl.set_position([.5, 1])
            #ax9.axis([90,630,90,630])
            ax9.axis([170,550,170,550])
            ax9.axis('off')
        else:
            print(f"Figure {loc_imagepath} doesn't exist")
    
    return basefig
