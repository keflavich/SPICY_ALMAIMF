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

 """
    Necessary object
    ----------
    fit_results is a dict whose structure is as follows:
    
    for each source in the sample (spicyid):
    {spicyid:
        {'flux' : array
        'error' : array
        'valid : array
        for each well-fitting model geometry (geom):
        'geom':
            {'model': list
            'chi2' : list
            'av' : list
            'sc' : list}
            }
        }
 """

geometries = ['s-pbhmi', 's-pbsmi',
              'sp--h-i', 's-p-hmi', 
              'sp--hmi', 'sp--s-i', 
              's-p-smi', 'sp--smi', 
              'spubhmi', 'spubsmi', 
              'spu-hmi', 'spu-smi', 
              's---s-i', 's---smi', 
              's-ubhmi', 's-ubsmi', 
              's-u-hmi', 's-u-smi']

def construct_fitinfo_tbl(fit_rslt):
    fits = {}
    for spicyid in tqdm(set(fit_rslt['SPICY'])):
        per_source_tbl = fit_rslt[fit_rslt['SPICY'] == spicyid]
        
        fits[spicyid] = {}
            
        fits[spicyid]['flux'] = np.array([float(x) for x in per_source_tbl[0]['source.flux'].split(", ")])
        fits[spicyid]['error'] = np.array([float(x) for x in per_source_tbl[0]['source.error'].split(", ")])
        fits[spicyid]['valid'] = np.array([int(x) for x in per_source_tbl[0]['source.valid'].split(", ")])
        
        for geom in set(per_source_tbl['geometry']):
            per_geom_tbl = per_source_tbl[per_source_tbl['geometry'] == geom]
            
            fits[spicyid][geom] = {}
            
            fits[spicyid][geom]['model'] = list(per_geom_tbl['MODEL_NAME'])
            fits[spicyid][geom]['chi2'] = list(per_geom_tbl['chi2'])
            fits[spicyid][geom]['av'] = list(per_geom_tbl['av'])
            fits[spicyid][geom]['sc'] = list(per_geom_tbl['sc'])
    return fits

def field_lookup(spicyid):
    field_lookup_dict = {'W43MM1':[92122, 92074],
                         'W51IRS2':[102000, 102002, 102007],
                         'G328':[31362, 31395, 31366, 31367, 31431, 31432, 31453, 31463, 31405, 31438, 31441, 31444, 31415, 31420, 31389, 31390, 31423],
                         'G337':[40344, 40328, 40362, 40365, 40367, 40343, 40312, 40380, 40382, 40311],
                         'W43MM3':[92076],
                         'G327':[30375, 30411, 30414, 30416, 30423, 30425],
                         'G353':[55873, 55876, 55881, 55853, 55858, 55859, 55924, 55862, 55896, 55932, 55901],
                         'G12':[77504, 77443, 77507, 77447, 77416, 77452, 77454, 77428, 77462, 77465, 77498],
                         'W43MM2':[92039, 92043, 92015, 91989, 92055, 92056],
                         'G10':[75717, 75752, 75756, 75725, 75788, 75767, 75735, 75743],
                         'G333':[36252, 36263],
                         'G351':[54167, 54182, 54188, 54189, 54192, 54197, 54200, 54207, 54212, 54213, 54221, 54222, 54224, 54228, 54233, 54235, 54251, 54254, 54255, 54265, 54268],
                         'G008':[73698, 73668, 73673, 73642, 73675, 73676, 73678, 73650, 73683, 73653, 73659, 73662, 73695],
                         'W51-E':[102062, 102038],
                         'G338':[40135, 40136, 40141, 40114, 40158]}
    for fieldid in field_lookup_dict:
        if spicyid in field_lookup_dict[fieldid]:
            return fieldid
    print("ID not valid or not in sample.")

def distance_lookup(spicyid):
    fieldid = field_lookup(spicyid)
    if fieldid != None:
        distance_lookup_dict = {"W51-E": 5.4,"W43MM1": 5.5,"G333": 4.2,
                             "W51IRS2": 5.4,"G338": 3.9,"G10": 4.95,
                             "W43MM2": 5.5,"G008": 3.4,"G12": 2.4,
                             "G327": 2.5,"W43MM3": 5.5,"G351": 2.0,
                             "G353": 2.0,"G337": 2.7,"G328": 2.5,}
        return distance_lookup_dict[fieldid]
    else:
        print("ID not valid or not in sample.")

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

def find_mass_ul(spicyid, fit_results):
    regiondistance = distance_lookup(spicyid)
    alma_1mm_flx = fit_results[spicyid]['flux'][len(fit_results[spicyid]['flux'])-2]
    alma_3mm_flx = fit_results[spicyid]['flux'][len(fit_results[spicyid]['flux'])-1]
    alma_1mm_err = fit_results[spicyid]['error'][len(fit_results[spicyid]['error'])-2]
    alma_3mm_err = fit_results[spicyid]['error'][len(fit_results[spicyid]['error'])-1]
    
    if not np.isnan(alma_1mm_flx) and not np.ma.isMA(alma_1mm_flx): 
        alma_detect = alma_1mm_flx
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.008*u.cm**2/u.g * BlackBody(20*u.K)(230*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(alma_3mm_flx) and not np.ma.isMA(alma_3mm_flx): 
        alma_detect = alma_3mm_flx
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.002*u.cm**2/u.g * BlackBody(20*u.K)(100*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(alma_1mm_err) and not np.ma.isMA(alma_1mm_err): 
        alma_detect = alma_1mm_err
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.008*u.cm**2/u.g * BlackBody(20*u.K)(230*u.GHz) * u.sr)).to(u.M_sun)
    elif not np.isnan(alma_3mm_err) and not np.ma.isMA(alma_3mm_err): 
        alma_detect = alma_3mm_err        
        mass_ul = (((alma_detect)*u.Jy * (regiondistance*u.kpc)**2) / (0.002*u.cm**2/u.g * BlackBody(20*u.K)(100*u.GHz) * u.sr)).to(u.M_sun)
    else:
        mass_ul = np.nan  
    #230 for 1mm, 100 for 3mm
    return(mass_ul)

def get_approx_avrange(spicyid, fit_results):
    okgeo = list(fit_results[spicyid].keys())[3:len(fit_results[spicyid])]
    avmin = round(np.nanmin([np.nanmin(fit_results[spicyid][geom]['av']) for geom in okgeo]))
    avmax = round(np.nanmax([np.nanmax(fit_results[spicyid][geom]['av']) for geom in okgeo]))
    avrange = [avmin,avmax]
    return avrange

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

def get_modelcount(spicyid,fit_results):
    modelcount = 0
    for geom in geometries:
        if geom in fit_results[spicyid].keys():
            modelcount = modelcount+len(fit_results[spicyid][geom]['model'])
    return modelcount

def get_approx_chi2limit(spicyid,fit_results):
    # because the data in these tables has already been cropped to a chi2limit,
    # this function can just retrieve that chi2limit, or something close to it,
    # by finding the max chi2 that was allowed for a source.
    okgeo = list(fit_results[spicyid].keys())[3:len(fit_results[spicyid])]
    chi2limit = np.ceil((np.nanmax([np.nanmax(fit_results[spicyid][geom]['chi2']) for geom in okgeo])))
    return chi2limit

def datafunction(geom, chi2limit, bestfits, min_chi2=None):
    pars = Table.read(f'/blue/adamginsburg/richardson.t/research/flux/pars/{geom}_augmented.fits')
    fitinfo = bestfits[geom]
    if min_chi2 is None:
        min_chi2 = np.nanmin(fitinfo.chi2)
    selection = fitinfo.chi2 < chi2limit
    data = pars[fitinfo.model_id[selection]]
    return pars, data, selection

def binsfunction(param, kind, steps, spicyid, fit_results, robitaille_modeldir):
    chi2limit=get_approx_chi2limit(spicyid,fit_results)   
    default_aperture=3000*u.au
    
    datamin = []
    datamax = []
    for geom in list(fit_results[spicyid].keys())[3:len(fit_results[spicyid])]:
        fitinfo = fit_results[spicyid][geom]
        model_dir = f'{robitaille_modeldir}/{geom}'
        sedcube = SEDCube.read(f"{model_dir}/flux.fits",)
        apnum = np.argmin(np.abs(default_aperture - sedcube.apertures))
        
        pars = Table.read(f'/blue/adamginsburg/richardson.t/research/flux/pars/{geom}_augmented.fits')
        pars.add_index('MODEL_NAME')
        fitinfo = fit_results[spicyid][geom]
        indices = [x < chi2limit for x in fitinfo['chi2']]
        data = pars[[pars.loc_indices[x] for x in np.array(fitinfo['model'])[indices]]]
        
        if param in pars.keys():
            if param == "Line-of-Sight Masses":
                dataparam = data[param]
                datamin.append(dataparam[:,apnum].min())
                datamax.append(dataparam[:,apnum].max())
            elif param == "Sphere Masses":
                dataparam = data[param]
                datamin.append(dataparam[:,apnum].min())
                datamax.append(dataparam[:,apnum].max())
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
        bins = np.logspace(binsmin, binsmax, steps)

    if kind == 'lin':
        binsmin = min(datamin)
        binsmax = max(datamax)
        bins = np.linspace(binsmin, binsmax, steps)

    if kind == 'geom':
        binsmin = min(datamin)
        binsmax = max(datamax)
        bins = np.geomspace(binsmin, binsmax, steps)

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
