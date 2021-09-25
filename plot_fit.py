import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

from astropy import units as u
import sedfitter
from sedfitter.sed import SEDCube
from astropy.table import Table

import table_loading

def datafunction(geom, chi2lim, bestfits):
    pars = Table.read(f'/blue/adamginsburg/richardson.t/research/flux/pars/{geom}_augmented.fits')
    fitinfo = bestfits[geom]
    selection = fitinfo.chi2 < np.nanmin(fitinfo.chi2) + chi2lim
    data = pars[fitinfo.model_id[selection]]
    return pars, data

def binsfunction(param, kind, binsnum, chi2lim, geometries, bestfits, massnum=9):
    # note: the massnum indicates an index for aperture size, and is used in the
    # parameters which involve multiple aperture sizes to select just one. you'll
    # need to find out what your massnum= is if you use this.

    datamin = []
    datamax = []
    for geom in geometries:
        pars, data = datafunction(geom, chi2lim, bestfits)
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
        chi2limit,
        distance=2*u.kpc,
        extinction=table_loading.make_extinction(),
        show_per_aperture=True,
        default_aperture=3*u.arcsec,
        robitaille_modeldir='/blue/adamginsburg/richardson.t/research/flux/robitaille_models/',):

    # Setting up the plot surface
    basefig = plt.figure(figsize=(20, 22))
    gs = GridSpec(nrows=5, ncols=2, height_ratios=[4,1,1,1,1], hspace=0.25, wspace=0.1)

    # --------------------------------

    # Best fits plot
    ax0 = basefig.add_subplot(gs[0, 1])
    #wavelengths = u.Quantity([wavelength_dict[fn] for fn in filters], u.um)
    fitinfo = bestfits_source[geometries_selection[0]]
    source = fitinfo.source
    valid = source.valid
    wavelengths = u.Quantity([x['wav'] for x in fitinfo.meta.filters], u.um)
    apertures = u.Quantity([x['aperture_arcsec'] for x in fitinfo.meta.filters], u.arcsec)
    ax0.errorbar(wavelengths.value[valid==1], source.flux[valid==1]/1e3, yerr=source.error[valid==1]/1e3, linestyle='none', color='w', marker='o', markersize=10)
    ax0.plot(wavelengths.value[valid==3], source.flux[valid==3], linestyle='none', color='w', marker='v', markersize=10)

    for geom in geometries_selection:

        fitinfo = bestfits_source[geom]

        model_dir = f'{robitaille_modeldir}/{geom}'
        sedcube = SEDCube.read(f"{model_dir}/flux.fits",)

        index = np.nanargmin(fitinfo.chi2)
        modelname = fitinfo.model_name[index]
        sed = sedcube.get_sed(modelname)

        apnum = np.argmin(np.abs((default_aperture * distance).to(u.au, u.dimensionless_angles()) - sedcube.apertures))

        line, = ax0.plot(sedcube.wav,
                 sed.flux[apnum] * 10**fitinfo.sc[index] * 10**(fitinfo.av[index] * extinction.get_av(sed.wav)),
                 label=geom, alpha=0.9)

        if show_per_aperture:
            apnums = np.array([
                np.argmin(np.abs((apsize * distance).to(u.au, u.dimensionless_angles()) - sedcube.apertures))
                for apsize in apertures])
            wlids = np.array([
                np.argmin(np.abs(ww - sedcube.wav)) for ww in wavelengths])
            flux = np.array([sed.flux[apn, wavid].value for apn, wavid in zip(apnums, wlids)])
            flux = flux * 10**fitinfo.sc[index] * 10**(fitinfo.av[index] * extinction.get_av(wavelengths)),
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

    histalpha = 0.8
    lognum = 50
    linnum = 50

    tempbins = binsfunction('star.temperature', 'lin', linnum, chi2limit, geometries_selection, bestfits_source)
    lumbins = binsfunction('Model Luminosity', 'log', lognum, chi2limit, geometries_selection, bestfits_source)
    radbins = binsfunction('star.radius', 'log', lognum, chi2limit, geometries_selection, bestfits_source)
    losbins = binsfunction('Line-of-Sight Masses', 'log', 20, chi2limit, geometries_selection, bestfits_source, 0)
    try:
        dscbins = binsfunction('disk.mass', 'log', lognum, chi2limit, geometries_selection, bestfits_source)
    except ValueError:
        # this is OK; some models don't have disks
        pass
    sphbins = binsfunction('Sphere Masses', 'log', 20, chi2limit, geometries_selection, bestfits_source, 0)

    # index values used above and below for mass-related parameters should, i think, be the same as your
    # massnum index, which again has to do with aperture sizes

    for geom in geometries_selection:
        pars, data = datafunction(geom, chi2limit, bestfits_source)

        if 'star.temperature' in pars.keys():
            ax1.hist(data['star.temperature'], bins=tempbins, alpha=histalpha, label=geom)

        if 'Model Luminosity' in pars.keys():
            ax2.hist(data['Model Luminosity'], bins=lumbins, alpha=histalpha, label=geom)

        if 'star.radius' in pars.keys():
            ax3.hist(data['star.radius'], bins=radbins, alpha=histalpha, label=geom)

        if 'Line-of-Sight Masses' in pars.keys():
            ax4.hist(data['Line-of-Sight Masses'][0], bins=losbins, alpha=histalpha, label=geom)

        if 'disk.mass' in pars.keys():
            ax5.hist(data['disk.mass'], bins=dscbins, alpha=histalpha, label=geom)

        if 'Sphere Masses' in pars.keys():
            ax6.hist(data['Sphere Masses'][0], bins=sphbins, alpha=histalpha, label=geom)

    handles, labels = ax1.get_legend_handles_labels()
    ax0.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.16,1.02))
    ax1.set_xlabel("Stellar Temperature (K)")
    ax2.set_xlabel("Stellar Luminosity (L$_\odot$)")
    ax3.set_xlabel("Stellar Radius (R$_\odot$)")
    ax4.set_xlabel("Line-of-Sight Masses (M$_\odot$)")
    ax5.set_xlabel("Disk Mass (M$_\odot$)")
    ax6.set_xlabel("Sphere Mass (M$_\odot$)")

    _=ax2.semilogx()
    _=ax3.semilogx()
    _=ax4.semilogx()
    _=ax5.semilogx()
    _=ax6.semilogx()

    # --------------------------------

    # reading the saved image of the region with source location marked
    # locfig = mpimg.imread(f'/home/btingle/figures/{fieldid}_{spicyid}.png')

    # my image needs to be flipped
    # locfig = np.flipud(locfig)

    # ax9 = basefig.add_subplot(gs[0, 0])
    # ax9.imshow(locfig)
    # ttl = ax9.set_title(f'\n{fieldid}  |  SPICY {spicyid}\n', fontsize=25)
    # ttl.set_position([.5, 1])
    # #ax9.axis([90,630,90,630])
    # ax9.axis([170,550,170,550])
    # ax9.axis('off')
