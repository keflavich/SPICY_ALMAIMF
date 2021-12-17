from astropy.stats import mad_std

# setup
default_aperture=3*u.arcsec
distance = (regiondistance * u.kpc).mean()
# use the distance to your region (from literature) in the above line, mine is 2.5

sedcube = SEDCube.read(f"/blue/adamginsburg/richardson.t/research/flux/robitaille_models/s---s-i/flux.fits")
apnum = np.argmin(np.abs((default_aperture * distance).to(u.au, u.dimensionless_angles()) - sedcube.apertures))

all_data = []
extinction_data, distance_data = [], []
for geom in geometries:
    pars, data, selection = datafunction(geom, chi2limit, fits)
    all_data.append(data)
    distance_data.extend(10**fits[geom].sc[selection])
    extinction_data.extend(fits[geom].av[selection])
data = table.vstack(all_data)

# assign all parameter variables
star_temp_mean = np.nanmean(np.log10(data['star.temperature']))
model_lum_mean = np.nanmean(np.log10(data['Model Luminosity']))
star_rad_mean = np.nanmean(np.log10(data['star.radius']))
los_mass_mean = np.nanmean(np.log10(data['Line-of-Sight Masses'][:,apnum]))
#disk_mass_mean = np.nanmean(np.log10(data['disk.mass']))
sphere_mass_mean = np.nanmean(np.log10(data['Sphere Masses'][:,apnum]))
distance_mean = np.nanmean(np.log10(distance_data))
ext_mean = np.nanmean(np.log10(extinction_data))

star_temp_std = np.nanstd(np.log10(data['star.temperature']))
model_lum_std = np.nanstd(np.log10(data['Model Luminosity']))
star_rad_std = np.nanstd(np.log10(data['star.radius']))
los_mass_std = np.nanstd(np.log10(data['Line-of-Sight Masses'][:,apnum]))
#disk_mass_std = np.nanstd(np.log10(data['disk.mass']))
sphere_mass_std = np.nanstd(np.log10(data['Sphere Masses'][:,apnum]))
distance_std = np.nanstd(np.log10(distance_data))
ext_std = np.nanstd(np.log10(extinction_data))

star_temp_med = np.nanmedian(np.log10(data['star.temperature']))
model_lum_med = np.nanmedian(np.log10(data['Model Luminosity']))
star_rad_med = np.nanmedian(np.log10(data['star.radius']))
los_mass_med = np.nanmedian(np.log10(data['Line-of-Sight Masses'][:,apnum]))
#disk_mass_med = np.nanmedian(np.log10(data['disk.mass']))
sphere_mass_med = np.nanmedian(np.log10(data['Sphere Masses'][:,apnum]))
distance_med = np.nanmedian(np.log10(distance_data))
ext_med = np.nanmedian(np.log10(extinction_data))

star_temp_mad = mad_std(np.log10(data['star.temperature']))
model_lum_mad = mad_std(np.log10(data['Model Luminosity']))
star_rad_mad = mad_std(np.log10(data['star.radius']))
los_mass_mad = mad_std(np.log10(data['Line-of-Sight Masses'][:,apnum]))
#disk_mass_mad = mad_std(np.log10(data['disk.mass']))
sphere_mass_mad = mad_std(np.log10(data['Sphere Masses'][:,apnum]))
distance_mad = mad_std(np.log10(distance_data))
ext_mad = mad_std(np.log10(extinction_data))

# create table columns if necessary
for parameter in ['Temperature','Luminosity','Radius',
                  'LOS mass','Sphere mass','Distance','Extinction']:
    for new_column_name in (f'{parameter} mean',
                        f'{parameter} standard deviation',
                        f'{parameter} Median',
                        f'{parameter} MAD'):
        if new_column_name not in tbl.colnames:
            tbl[new_column_name] = np.nan

if 'Included Geometries' not in tbl.colnames:
    tbl['Included Geometries'] = ",".join(geometries)

if 'Chi2 threshold' not in tbl.colnames:
    tbl['Chi2 threshold'] = np.nan

# put the data in the source's tbl row
tbl['Temperature mean'][rownum] = star_temp_mean
tbl['Temperature standard deviation'][rownum] = star_temp_std
tbl['Temperature Median'][rownum] = star_temp_med
tbl['Temperature MAD'][rownum] = star_temp_mad

tbl['Luminosity mean'][rownum] = model_lum_mean
tbl['Luminosity standard deviation'][rownum] = model_lum_std
tbl['Luminosity Median'][rownum] = model_lum_med
tbl['Luminosity MAD'][rownum] = model_lum_mad

tbl['Radius mean'][rownum] = star_rad_mean
tbl['Radius standard deviation'][rownum] = star_rad_std
tbl['Radius Median'][rownum] = star_rad_med
tbl['Radius MAD'][rownum] = star_rad_mad

tbl['LOS mass mean'][rownum] = los_mass_mean
tbl['LOS mass standard deviation'][rownum] = los_mass_std
tbl['LOS mass Median'][rownum] = los_mass_med
tbl['LOS mass MAD'][rownum] = los_mass_mad

tbl['Sphere mass mean'][rownum] = sphere_mass_mean
tbl['Sphere mass standard deviation'][rownum] = sphere_mass_std
tbl['Sphere mass Median'][rownum] = sphere_mass_med
tbl['Sphere mass MAD'][rownum] = sphere_mass_mad

tbl['Distance mean'][rownum] = distance_mean
tbl['Distance standard deviation'][rownum] = distance_std
tbl['Distance Median'][rownum] = distance_med
tbl['Distance MAD'][rownum] = distance_mad

tbl['Extinction mean'][rownum] = ext_mean
tbl['Extinction standard deviation'][rownum] = ext_std
tbl['Extinction Median'][rownum] = ext_med
tbl['Extinction MAD'][rownum] = ext_mad

tbl['Included Geometries'][rownum] = ",".join(okgeo)
tbl['Chi2 threshold'][rownum] = chi2limit
# chi2limit is the threshold number you used to select okgeo AKA the "good fit" geometries

tbl.write(f'/home/btingle/tbl_G328', format='fits', overwrite=True)
# alter above line to point to the fits table containing your region's rows, with correct data, errors, conversions, etc.
# run this code (or your version of it as long as columns match) for each source in your region to assemble a complete table
