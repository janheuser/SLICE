##############################################################################################################################
# Date: 2022-5-9
# Name: SLICE_1D.py
# Author: James Anheuser
# Description: Applies SLICE thermodynamic sea ice thickness growth retrieval to a 1D profile along ice mass balance buoy tracks 
# Inputs: Buoy data, snow--ice interface temperature
# Outputs: NetCDF file saved to output directory containing buoy profiles and SLICE profiles

import numpy as np
import pandas as pd
import glob
from datetime import timedelta
from datetime import datetime
from scipy.stats import circmean
import calctsi as ct
import xarray as xr
from pyproj import Transformer, CRS, Geod
from scipy.interpolate import griddata


buoys = ['2003C', '2005F', '2006C', '2012G', '2012H', '2012L', '2013F', '2013G', '2015F'] #buoys to analyze
tsifp = '/ships19/cryo/janheuser/tsi/corrected/' #filepath for snow--ice interface data
outfp = '/home/janheuser/projects/thk/SLICE_paperrevs/' #output directory

transformer = Transformer.from_crs(4326, 6931)

def latmean(lons):
    return circmean(lons, high=360, low=0, nan_policy='omit')


def cond_eff(T, S):
    """Calculate effective sea ice conductivity based on Feltham et al. (2006)

    Arguments:
    T: sea ice temperature [C]
    S: sea ice salinity [ppt]

    Returns:
    effective sea ice conductivity [W m^-1 K^-1]
    """

    k_a = 0.03
    V_a = 0.025
    k_i = 1.16 * (1.91 - 8.66e-3 * T + 2.97e-5 * T ** 2)
    k_b = 1.16 * (0.45 + 1.08e-2 * T + 5.04e-5 * T ** 2)
    k_bi = k_i * (2 * k_i + k_a - 2 * V_a * (k_i - k_a)) / (2 * k_i + k_a + 2 * V_a * (k_i - k_a))
    T_l = -.0592 * S - 9.37e-6 * S ** 2 - 5.33e-7 * S ** 3

    return k_bi - (k_bi - k_b) * T_l / T


def stefpred(H_i, T_si, F_w, dur_seconds):
    """Calculate sea ice thickness using SLICE

    Arguments:
    U_i: initial sea ice thickness [m]
    T_si: snow--ice interface temperature [K]
    F_w: basal flux [W m^-2]
    dur_seconds: time step [s]

    Returns:
    sea ice thickness [m]
    """

    S = 33
    T_f = -.0592 * S - 9.37e-6 * S ** 2 - 5.33e-7 * S ** 3
    L = 333700 + 762.7 * T_f - 7.929 * T_f ** 2
    rho = 917

    k_eff = cond_eff(T_si - 273, 0)

    return np.sqrt(H_i ** 2 + (2 * k_eff / rho / L) * dur_seconds * (T_f + 273 - T_si)) - dur_seconds * F_w / rho / L

alldata = []

fp = glob.glob('/data/users/janheuser/crrel_imb/2*')

for folder in buoys:

    buoy = folder[-5:]
    print(buoy)

    tsifn = '/data/users/janheuser/crrel_imb/' + buoy + '/' + buoy + '_Temperature_Data.csv'
    thkfn = '/data/users/janheuser/crrel_imb/' + buoy + '/' + buoy + '_Mass_Balance_Data.csv'
    posfn = '/data/users/janheuser/crrel_imb/' + buoy + '/' + buoy + '_Position_Data.csv'

    tsi = pd.read_csv(tsifn, skiprows=1, index_col=0, parse_dates=True).dropna(how='all')
    thk = pd.read_csv(thkfn, skiprows=1, index_col=0, parse_dates=True).dropna(how='all')
    pos = pd.read_csv(posfn, skiprows=1, index_col=0, parse_dates=True).dropna(how='all')

    thk = thk.drop(['(mm/dd/yy hh:mm)'])
    thk = thk.astype(float)
    pos = pos.drop(['(mm/dd/yy hh:mm)'])
    pos = pos.where(pos.Longitude != ' ')
    pos = pos.where(pos.Latitude != '0')
    pos = pos.astype(float)
    pos.index = pd.to_datetime(pos.index)
    pos.Longitude[pos.Longitude < 0] = pos.Longitude[pos.Longitude < 0] + 360
    pos.Latitude = pos.Latitude.where(np.abs(pos.Latitude) <= 90)

    data = tsi['0.0'].to_frame().join(thk['Ice Thickness'], how='outer')
    data = data.join(pos.Longitude.resample('D').apply(latmean), how='outer')
    data = data.join(pos.Latitude, how='outer')
    data = data.loc[~data.index.duplicated(keep='first')]
    data = data.resample('D').mean()

    if buoy == '2013F':
        data = pd.concat([data[data.index.get_loc(datetime(data.index[0].year, 11, 1)):data.index.get_loc(
            datetime(data.index[0].year + 1, 4, 2))],
                          data[data.index.get_loc(datetime(data.index[0].year + 1, 11, 1)):data.index.get_loc(
                              datetime(data.index[0].year + 2, 4, 2))]])
    else:
        data = data[data.index.get_loc(datetime(data.index[0].year, 11, 1)):data.index.get_loc(
            datetime(data.index[0].year + 1, 4, 2))]

    data = data.dropna(how='any')

    data['dur'] = data.index.to_series().diff()

    data['atsi'] = np.nan

    data['H_predicted'] = np.nan
    data['H_predicted_2'] = np.nan
    data['H_predicted_5'] = np.nan
    data['H_predicted_10'] = np.nan
    data['H_predicted_-.25'] = np.nan
    data['H_predicted_+.25'] = np.nan
    data['H_predicted_-.5'] = np.nan
    data['H_predicted_+.5'] = np.nan

    for i in range(0, len(data)):
        print(f'{data.index[i]}\r', end="")

        if data.loc[data.index[i], 'dur'] > timedelta(minutes=0):
#             data.loc[data.index[i], 'atsi'] = ct.calcTsi_amsr_loc(data.index[i].date(),
#                                                                   data.loc[data.index[i], 'Latitude'],
#                                                                   data.loc[data.index[i], 'Longitude'])
            
            tsi = xr.open_dataset(tsifp + str(data.index[i].year) + str(data.index[i].dayofyear) + '.nc').dat
            tsi_x, tsi_y = transformer.transform(tsi.lat.values, tsi.lon.values)
            b_x, b_y = transformer.transform(data.Latitude[i], data.Longitude[i])
            data.loc[data.index[i], 'atsi'] = griddata((tsi_y.flatten(),tsi_x.flatten()),tsi.values.flatten(),(b_y,b_x), method='linear')
    
    data['atsi']=data['atsi'].fillna(method='bfill')
    
    i = 0

    data.loc[data.index[i], 'H_predicted'] = data.loc[data.index[i], 'Ice Thickness']
    data.loc[data.index[i], 'H_predicted_2'] = data.loc[data.index[i], 'Ice Thickness']
    data.loc[data.index[i], 'H_predicted_5'] = data.loc[data.index[i], 'Ice Thickness']
    data.loc[data.index[i], 'H_predicted_10'] = data.loc[data.index[i], 'Ice Thickness']
    data.loc[data.index[i], 'H_predicted_-.25'] = data.loc[data.index[i], 'Ice Thickness'] - 0.25
    data.loc[data.index[i], 'H_predicted_+.25'] = data.loc[data.index[i], 'Ice Thickness'] + 0.25
    data.loc[data.index[i], 'H_predicted_-.5'] = data.loc[data.index[i], 'Ice Thickness'] - 0.5
    data.loc[data.index[i], 'H_predicted_+.5'] = data.loc[data.index[i], 'Ice Thickness'] + 0.5

    i = i + 1

    while i < len(data):
        print(f'{data.index[i]}\r', end="")

        if data.index[i] - data.index[i - 1] > pd.Timedelta('60 days') and data.index[i].month > 7:

            i = data.index.get_loc(datetime(data.index[i].year, 11, 1))

            data.loc[data.index[i], 'H_predicted'] = data.loc[data.index[i], 'Ice Thickness']
            data.loc[data.index[i], 'H_predicted_2'] = data.loc[data.index[i], 'Ice Thickness']
            data.loc[data.index[i], 'H_predicted_5'] = data.loc[data.index[i], 'Ice Thickness']
            data.loc[data.index[i], 'H_predicted_10'] = data.loc[data.index[i], 'Ice Thickness']
            data.loc[data.index[i], 'H_predicted_-.25'] = data.loc[data.index[i], 'Ice Thickness'] - 0.25
            data.loc[data.index[i], 'H_predicted_+.25'] = data.loc[data.index[i], 'Ice Thickness'] + 0.25
            data.loc[data.index[i], 'H_predicted_-.5'] = data.loc[data.index[i], 'Ice Thickness'] - 0.5
            data.loc[data.index[i], 'H_predicted_+.5'] = data.loc[data.index[i], 'Ice Thickness'] + 0.5

            i += 1

        if data.loc[data.index[i], 'dur'] > timedelta(minutes=0):

            if np.isnan(data.loc[data.index[i], 'atsi']):
                data.loc[data.index[i], 'H_predicted'] = data.loc[data.index[i - 1], 'H_predicted']
                data.loc[data.index[i], 'H_predicted_2'] = data.loc[data.index[i - 1], 'H_predicted_2']
                data.loc[data.index[i], 'H_predicted_5'] = data.loc[data.index[i - 1], 'H_predicted_5']
                data.loc[data.index[i], 'H_predicted_10'] = data.loc[data.index[i - 1], 'H_predicted_10']
                data.loc[data.index[i], 'H_predicted_-.25'] = data.loc[data.index[i - 1], 'H_predicted_-.25']
                data.loc[data.index[i], 'H_predicted_+.25'] = data.loc[data.index[i - 1], 'H_predicted_+.25']
                data.loc[data.index[i], 'H_predicted_-.5'] = data.loc[data.index[i - 1], 'H_predicted_-.5']
                data.loc[data.index[i], 'H_predicted_+.5'] = data.loc[data.index[i - 1], 'H_predicted_+.5']

            else:
                data.loc[data.index[i], 'H_predicted'] = stefpred(data.loc[data.index[i - 1], 'H_predicted'],
                                                                  data.loc[data.index[i], 'atsi'], 0,
                                                                  data.loc[data.index[i], 'dur'].total_seconds())

                data.loc[data.index[i], 'H_predicted_2'] = stefpred(data.loc[data.index[i - 1], 'H_predicted_2'],
                                                                    data.loc[data.index[i], 'atsi'], 2,
                                                                    data.loc[data.index[i], 'dur'].total_seconds())

                data.loc[data.index[i], 'H_predicted_5'] = stefpred(data.loc[data.index[i - 1], 'H_predicted_5'],
                                                                    data.loc[data.index[i], 'atsi'], 5,
                                                                    data.loc[data.index[i], 'dur'].total_seconds())

                data.loc[data.index[i], 'H_predicted_10'] = stefpred(data.loc[data.index[i - 1], 'H_predicted_10'],
                                                                     data.loc[data.index[i], 'atsi'], 10,
                                                                     data.loc[data.index[i], 'dur'].total_seconds())

                data.loc[data.index[i], 'H_predicted_-.25'] = stefpred(data.loc[data.index[i - 1], 'H_predicted_-.25'],
                                                                       data.loc[data.index[i], 'atsi'], 2,
                                                                       data.loc[data.index[i], 'dur'].total_seconds())

                data.loc[data.index[i], 'H_predicted_+.25'] = stefpred(data.loc[data.index[i - 1], 'H_predicted_+.25'],
                                                                       data.loc[data.index[i], 'atsi'], 2,
                                                                       data.loc[data.index[i], 'dur'].total_seconds())

                data.loc[data.index[i], 'H_predicted_-.5'] = stefpred(data.loc[data.index[i - 1], 'H_predicted_-.5'],
                                                                      data.loc[data.index[i], 'atsi'], 2,
                                                                      data.loc[data.index[i], 'dur'].total_seconds())

                data.loc[data.index[i], 'H_predicted_+.5'] = stefpred(data.loc[data.index[i - 1], 'H_predicted_+.5'],
                                                                      data.loc[data.index[i], 'atsi'], 2,
                                                                      data.loc[data.index[i], 'dur'].total_seconds())

            i += 1

    data = data.reset_index().to_xarray()
    data['buoy'] = buoy
    alldata.append(data)

alldata = xr.concat(alldata, 'buoy')

snip = alldata.sel(buoy='2013F').dropna(dim='index', how='all').sel(index=slice(152, None))
snip['buoy'] = '2013Fb'
snip['index'] = range(0, 152)
alldata = xr.concat([alldata, snip], 'buoy')
snip = alldata.sel(buoy='2013F').dropna(dim='index', how='all').sel(index=slice(None, 151))
alldata = alldata.drop_sel(buoy='2013F')
alldata = xr.concat([alldata, snip], 'buoy')
alldata = alldata.dropna(dim='index', how='all')

alldata.to_netcdf(outfp + 'SLICE_1D.nc')
