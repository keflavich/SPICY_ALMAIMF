import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

from astropy import units as u
import sedfitter
from sedfitter.sed import SEDCube
from astropy.table import Table

import table_loading

def datafunction(geom, deltachi2lim, bestfits):
    pars = Table.read(f'/blue/adamginsburg/richardson.t/research/flux/pars/{geom}_augmented.fits')
    fitinfo = bestfits[geom]
    selection = fitinfo.chi2 < (np.nanmin(fitinfo.chi2) + deltachi2lim)
    data = pars[fitinfo.model_id[selection]]
    return pars, data

def binsfunction(param, kind, binsnum, deltachi2lim, geometries, bestfits, massnum=9):
    # note: the massnum indicates an index for aperture size, and is used in the
    # parameters which involve multiple aperture sizes to select just one. you'll
    # need to find out what your massnum= is if you use this.

    datamin = []
    datamax = []
    for geom in geometries:
        pars, data = datafunction(geom, deltachi2lim, bestfits)
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
    for x in datamin:
        if x == 0:
            datamin.remove(x)

    for x in datamax:
        if x == 0:
            datamax.remove(x)

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

    return bins

def plot_fit(bestfits_source, geometries_selection,
        deltachi2limit, chi2limit, fieldid, spicyid, ylimplot=3e7,
        extinction=table_loading.make_extinction(),
        show_per_aperture=True,
        default_aperture=3*u.arcsec,
        robitaille_modeldir='/blue/adamginsburg/richardson.t/research/flux/robitaille_models/',
        show_all_models=False,
             alpha_allmodels=0.1,
            ):

    # Setting up the plot surface
    basefig = plt.figure(figsize=(20, 22))
    gs = GridSpec(nrows=6, ncols=2, height_ratios=[4,1,1,1,1,1], hspace=0.25, wspace=0.1)

    # --------------------------------

    # Best fits plot
    ax0 = basefig.add_subplot(gs[0, 1])
    #wavelengths = u.Quantity([wavelength_dict[fn] for fn in filters], u.um)
    fitinfo = bestfits_source[geometries_selection[0]]
    source = fitinfo.source
    valid = source.valid
    wavelengths = u.Quantity([x['wav'] for x in fitinfo.meta.filters], u.um)
    apertures = u.Quantity([x['aperture_arcsec'] for x in fitinfo.meta.filters], u.arcsec)
    ax0.errorbar(wavelengths.value[valid==1], source.flux[valid==1], yerr=source.error[valid==1], linestyle='none', color='w', marker='o', markersize=10)
    ax0.plot(wavelengths.value[valid==3], source.flux[valid==3], linestyle='none', color='w', marker='v', markersize=10)

    distance = (10**fitinfo.sc * u.kpc).mean()
    
    for geom in geometries_selection:

        fitinfo = bestfits_source[geom]

        model_dir = f'{robitaille_modeldir}/{geom}'
        sedcube = SEDCube.read(f"{model_dir}/flux.fits",)

           
        
        index = np.nanargmin(fitinfo.chi2)
        
        distance = (10**fitinfo.sc[index] * u.kpc)

        modelname = fitinfo.model_name[index]
        sed = sedcube.get_sed(modelname)

        apnum = np.argmin(np.abs((default_aperture * distance).to(u.au, u.dimensionless_angles()) - sedcube.apertures))

        # https://github.com/astrofrog/sedfitter/blob/41dee15bdd069132b7c2fc0f71c4e2741194c83e/sedfitter/sed/sed.py#L64
        distance_scale = (1*u.kpc/distance)**2
        
        # https://github.com/astrofrog/sedfitter/blob/41dee15bdd069132b7c2fc0f71c4e2741194c83e/sedfitter/sed/sed.py#L84
        av_scale = 10**((fitinfo.av[index] * extinction.get_av(sed.wav)))
        
        line, = ax0.plot(sedcube.wav,
                 sed.flux[apnum] * distance_scale * av_scale,
                 label=geom, alpha=0.9)

        
        indices = fitinfo.chi2 < (deltachi2limit + np.nanmin(fitinfo.chi2))
        
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

    #tempbins = binsfunction('star.temperature', 'lin', linnum, chi2limit, geometries_selection, bestfits_source)
    tempbins = np.linspace(2000, 30000, 50)
    #lumbins = binsfunction('Model Luminosity', 'log', lognum, chi2limit, geometries_selection, bestfits_source)
    lumbins = np.logspace(-4,7,100)
    #radbins = binsfunction('star.radius', 'log', lognum, chi2limit, geometries_selection, bestfits_source)
    radbins = np.geomspace(0.1, 100, 50)
    try:
        losbins = binsfunction('Line-of-Sight Masses', 'log', 20, deltachi2limit, geometries_selection, bestfits_source, 0)
    except ValueError:
        losbins = np.geomspace(1e-4,10)
    try:
        dscbins = binsfunction('disk.mass', 'log', lognum, deltachi2limit, geometries_selection, bestfits_source)
    except ValueError:
        # this is OK; some models don't have disks
        pass
    sphbins = binsfunction('Sphere Masses', 'log', 50, deltachi2limit, geometries_selection, bestfits_source, 0)

    # index values used above and below for mass-related parameters should, i think, be the same as your
    # massnum index, which again has to do with aperture sizes

    for geom in geometries_selection:
        pars, data = datafunction(geom, deltachi2limit, bestfits_source)

        if 'star.temperature' in pars.keys():
            ax1.hist(data['star.temperature'], bins=tempbins, alpha=histalpha, label=geom)

        if 'Model Luminosity' in pars.keys():
            ax2.hist(data['Model Luminosity'], bins=lumbins, alpha=histalpha, label=geom)

        if 'star.radius' in pars.keys():
            ax3.hist(data['star.radius'], bins=radbins, alpha=histalpha, label=geom)

        if 'Line-of-Sight Masses' in pars.keys():
            ax4.hist(data['Line-of-Sight Masses'][:,apnum], bins=losbins, alpha=histalpha, label=geom)

        if 'disk.mass' in pars.keys():
            ax5.hist(data['disk.mass'], bins=dscbins, alpha=histalpha, label=geom)

        if 'Sphere Masses' in pars.keys():
            ax6.hist(data['Sphere Masses'][:,apnum], bins=sphbins, alpha=histalpha, label=geom)
   
    for geom in geometries_selection:

        fitinfo = bestfits_source[geom]
        
        distances = 10**fitinfo.sc
        ax7.hist(distances, bins=np.linspace(distances.min(), distances.max()))

        ax8.hist(fitinfo.av, bins=np.linspace(np.nanmin(fitinfo.av), np.nanmax(fitinfo.av)))
    
            
    handles, labels = ax1.get_legend_handles_labels()
    ax0.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.16,1.02))
    ax1.set_xlabel("Stellar Temperature (K)")
    ax2.set_xlabel("Stellar Luminosity (L$_\odot$)")
    ax3.set_xlabel("Stellar Radius (R$_\odot$)")
    ax4.set_xlabel("Line-of-Sight Masses (M$_\odot$)")
    ax5.set_xlabel("Disk Mass (M$_\odot$)")
    ax6.set_xlabel("Sphere Mass (M$_\odot$)")
    ax7.set_xlabel("Distance (kpc)")
    ax8.set_xlabel("Extinction [$A_V$]")

    _=ax2.semilogx()
    _=ax3.semilogx()
    _=ax4.semilogx()
    _=ax5.semilogx()
    _=ax6.semilogx()

    # --------------------------------

    # reading the saved image of the region with source location marked
    locfig = mpimg.imread(f'/home/btingle/figures/{fieldid}_{spicyid}.png')

    # my image needs to be flipped
    locfig = np.flipud(locfig)

    ax9 = basefig.add_subplot(gs[0, 0])
    ax9.imshow(locfig)
    ttl = ax9.set_title(f'\n{fieldid}  |  SPICY {spicyid}  | [{chi2limit}]\n', fontsize=25)
    ttl.set_position([.5, 1])
    #ax9.axis([90,630,90,630])
    ax9.axis([170,550,170,550])
    ax9.axis('off')