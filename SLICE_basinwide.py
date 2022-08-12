##############################################################################################################################
# Date: 2022-8-10
# Name: SLICE_basinwide.py
# Author: James Anheuser
# Description: Applies SLICE thermodynamic sea ice thickness growth retrieval to a basin-wide scale sea ice model
# Inputs: Years, Polar Pathfinder sea ice motion vectors, AWI CS2SMOS sea ice thickness data, snow--ice interface temperature data, sea ice concentration data
# Outputs: NetCDF file saved to output directory containing basinwide daily winter time sea ice thickness for the years in question


import xarray as xr
from datetime import date
import numpy as np
from pyproj import Transformer
from scipy.interpolate import griddata
import calctsi as ct
from scipy.stats import binned_statistic_2d
import glob



years = range(2012,2020) #years to apply 
motvecfp = '/ships19/cryo/janheuser/mot_vec/' #filepath for motion vector data
cs2fp = '/ships19/cryo/janheuser/cs2smos/ftp.awi.de/sea_ice/product/cryosat2_smos/v203/nh/' #filepath for CryoSat-2 data
tsifp = '/ships19/cryo/janheuser/tsi/corrected/' #filepath for snow--ice interface data
outfp = '/home/janheuser/projects/thk/SLICE_paperrevs/release_2/SLICE_basinwide/' #output directory


def cond_eff(T, S):
    """Calculate effective sea ice conductivity based on Feltham et al. (2006)
    
    Arguments:
    T: sea ice temperature [C]
    S: sea ice salinity [ppt]
    
    Returns:
    effective sea ice conductivity [W m^-1 K^-1]
    """
    
    k_a=0.03
    V_a=0.025
    k_i=1.16*(1.91-8.66e-3*T+2.97e-5*T**2)
    k_b=1.16*(0.45+1.08e-2*T+5.04e-5*T**2)
    k_bi=k_i*(2*k_i+k_a-2*V_a*(k_i-k_a))/(2*k_i+k_a+2*V_a*(k_i-k_a))
    T_l = -.0592*S-9.37e-6*S**2-5.33e-7*S**3
    
    return k_bi-(k_bi-k_b)*T_l/T


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
    T_f = -.0592*S-9.37e-6*S**2-5.33e-7*S**3
    L = 333700+762.7*T_f-7.929*T_f**2
    rho = 917
    
    T_si=T_si.fillna(T_f+273)
    
    k_eff=cond_eff(T_si-273, 0)
    
    return np.sqrt(H_i**2 + (2*k_eff/rho/L)*dur_seconds*(T_f+273-T_si))-dur_seconds*F_w/rho/L


ease=xr.open_dataset('/home/janheuser/projects/thk/EASE2_N25km.geolocation.v0.9.nc')
ease['y']=-ease.y

y_bins=0.5*ease.y.values[:-1]+0.5*ease.y.values[1:]
y_bins=np.concatenate([[y_bins[0]-np.diff(y_bins)[0]],y_bins,[y_bins[-1]+np.diff(y_bins)[0]]])

x_bins=0.5*ease.x.values[:-1]+0.5*ease.x.values[1:]
x_bins=np.concatenate([[x_bins[0]-np.diff(x_bins)[0]],x_bins,[x_bins[-1]+np.diff(x_bins)[0]]])


for year in years:

    mot1 = xr.open_dataset(motvecfp + 'icemotion_daily_nh_25km_'+str(year)+'0101_'+str(year)+'1231_v4.1.nc')
    mot2 = xr.open_dataset(motvecfp + 'icemotion_daily_nh_25km_'+str(year+1)+'0101_'+str(year+1)+'1231_v4.1.nc')

    mot = xr.merge([mot1, mot2])
    mot = mot.sel(time=slice(str(year)+'-11-01',str(year+1)+'-04-01'))
    
    bw = xr.DataArray(np.zeros((len(mot.time), len(ease.y), len(ease.x))), coords=[mot.time, ease.y, ease.x], dims=["time", "y", "x"])
    bw[:,:,:]=np.nan

    #transform motion from EASE 1 to EASE 2
    transformer = Transformer.from_crs(3408, 6931)
    new_x, new_y = transformer.transform(mot.x.values, mot.y.values)
    mot['x']=new_x
    mot['y']=new_y

    #replace nans with 0
    mot['u'] = mot['u'].fillna(0)
    mot['v'] = mot['v'].fillna(0)

    # intialize parcels with cs2smos data
    cs2fn = cs2fp + 'W_XX-ESA,SMOS_CS2,NH_25KM_EASE2_' + str(mot.time[0].dt.year.values) + '1028*.nc'
    cs2fn = glob.glob(cs2fn)[0]

    thk_0=xr.open_dataset(cs2fn).analysis_sea_ice_thickness[0,:,:]
    thk_0['xc']=thk_0.xc*1000
    thk_0['yc']=thk_0.yc*1000
    thk_0=thk_0.rename({'xc':'x','yc':'y'})

#     thk_0 = thk_0.analysis_sea_ice_thickness.sel(xc=ease.x, yc=ease.y,method="nearest")[0,:,:]

    bw[0,:,:].loc[dict(x=thk_0.x, y=thk_0.y)]=thk_0.values


    # interpolate initial thickness to 5km grid and initialize parcel variable
    xi = np.linspace(ease.x[0], ease.x[-1], (ease.dims["x"]-1) * 5+1)
    yi = np.linspace(ease.y[0], ease.y[-1], (ease.dims["y"]-1) * 5+1)

    parcels=thk_0.interp(x=xi,y=yi).stack(z=('y','x')).drop(['lat','lon'])
    parcels=parcels.where(~np.isnan(parcels), drop=True)
    parcels=parcels.expand_dims(time=len(mot.time)).copy().assign_coords(time=mot.time)
    parcels[1:,:]=np.nan
    x=np.zeros((len(parcels.time), len(parcels.x)))
    y=np.zeros((len(parcels.time), len(parcels.y)))
    id25=np.zeros((len(parcels.time), len(parcels.y)))
    x[:]=np.nan
    y[:]=np.nan
    x[0,:]=parcels.x.values
    y[0,:]=parcels.y.values
    parcels['z']=range(0,len(parcels.z))
    parcels['x']=xr.DataArray(x, coords=[parcels.time, parcels.z], dims=['time', 'z'])
    parcels['y']=xr.DataArray(y, coords=[parcels.time, parcels.z], dims=['time', 'z'])


    for t in range(0,len(mot.time)-1):
        print(str(mot.time[t].dt.year.values) + '/' + str(mot.time[t].dt.month.values) + '/' +
              str(mot.time[t].dt.day.values), end="\r") #print date 

        #load in Tsi and SIC
        sic = ct.readamsr_sic(date(mot.time[t+1].dt.year.values, mot.time[t+1].dt.month.values, mot.time[t+1].dt.day.values))
        sic = sic.where(sic.dat!=120)
        sic = sic.where(sic.dat>0)
        tsi=ct.calcTsi_amsr(date(mot.time[t+1].dt.year.values, mot.time[t+1].dt.month.values, mot.time[t+1].dt.day.values)).dat

        #regrid Tsi/ SIC from stereo to EASE
        transformer = Transformer.from_crs(4326, 6931)
        new_x, new_y = transformer.transform(tsi.lat.values, tsi.lon.values)
        tsi = griddata((new_y.flatten(),new_x.flatten()),tsi.values.flatten(),
                    (mot.y.values[:,None],mot.x.values[None,:]), method='linear')
        tsi = xr.DataArray(tsi, coords=[mot.y, mot.x], dims=['y','x'])

        transformer = Transformer.from_crs(4326, 6931)
        new_x, new_y = transformer.transform(sic.lat.values, sic.lon.values)
        sic = griddata((new_y.flatten(),new_x.flatten()),sic.dat.values.flatten(),
                    (mot.y.values[:,None],mot.x.values[None,:]), method='linear')
        sic = xr.DataArray(sic, coords=[mot.y, mot.x], dims=['y','x'])
#         sic_interp=sic.interp(x=(xi[1:] + xi[:-1]) / 2, y=(yi[1:] + yi[:-1]) / 2, method='nearest')  

        tsi=tsi.where(sic>95)

        #move ice parcels per ice motion product velocity interpolated to parcel position
        parcels['x'][t+1,:]=parcels.x[t,:]+86400e-2*mot.u[t,:].interp(x=parcels.x[t,:],y=parcels.y[t,:]).fillna(0)
        parcels['y'][t+1,:]=parcels.y[t,:]+86400e-2*mot.v[t,:].interp(x=parcels.x[t,:],y=parcels.y[t,:]).fillna(0)

        # caluclate new thickness and add thickness where no ice exists but SIC says it does
        tsi.loc[dict(x=slice(-200000,200000),y=slice(-200000,200000))]=(.5*tsi.loc[dict(
            x=slice(-200000,200000),y=slice(-200000,200000))].interpolate_na(dim='x')+.5*tsi.loc[dict(
            x=slice(-200000,200000),y=slice(-200000,200000))].interpolate_na(dim='y'))
        newH = stefpred(parcels[t,:], tsi.sel(x=parcels.x[t,:],y=parcels.y[t,:],method='nearest'), 2, 8.64e4)
        parcels[t+1,:] = newH.where(newH>0,0)
        parcels[t+1,:] = parcels[t+1,:].where(sic.sel(x=parcels.x[t,:],y=parcels.y[t,:],method='nearest')>0)

        counts = binned_statistic_2d(parcels.y[t+1,:].where(~np.isnan(parcels[t+1,:]), drop=True).values, 
                                 parcels.x[t+1,:].where(~np.isnan(parcels[t+1,:]), drop=True).values, 
                                 parcels[t+1,:].where(~np.isnan(parcels[t+1,:]), drop=True).values, 'count', 
                                 bins=[mot.y.values, mot.x.values]).statistic
        counts = xr.DataArray(counts, coords=[0.5*(mot.y.values[1:]+mot.y.values[:-1]),0.5*(mot.x.values[1:]+mot.x.values[:-1])], dims=['y','x'])
        counts = counts.sel(y=yi,x=xi,method='nearest')
        counts['y']=yi
        counts['x']=xi

        sic = sic.sel(y=yi,x=xi,method='nearest')
        sic['y']=yi
        sic['x']=xi

        newice = xr.DataArray(0.05+np.zeros((len(yi),len(xi))), coords=[yi, xi], 
                          dims=["y", "x"]).where((sic>95) & (counts<1),drop=True)
        
        parcels_append=newice.stack(z=('y','x'))

        parcels_append=parcels_append.where(~np.isnan(parcels_append), drop=True)

        parcels_append=parcels_append.expand_dims(time=len(mot.time)).copy().assign_coords(time=mot.time)
        parcels_append[:t+1,:]=np.nan
        parcels_append[t+2:,:]=np.nan
        x=np.zeros((len(parcels_append.time),len(parcels_append.x)))
        y=np.zeros((len(parcels_append.time),len(parcels_append.y)))
        x[:]=np.nan
        y[:]=np.nan
        x[t+1,:]=parcels_append.x.values
        y[t+1,:]=parcels_append.y.values
        parcels_append['z']=range(len(parcels.z),len(parcels_append.z)+len(parcels.z))
        parcels_append['x']=xr.DataArray(x, coords=[parcels_append.time, parcels_append.z], dims=['time', 'z'])
        parcels_append['y']=xr.DataArray(y, coords=[parcels_append.time, parcels_append.z], dims=['time', 'z'])
        parcels=xr.concat([parcels, parcels_append], 'z')
        
        bw[t+1,:,:] = binned_statistic_2d(parcels.y[t+1,:].where(~np.isnan(parcels[t+1,:]), drop=True).values, 
                        parcels.x[t+1,:].where(~np.isnan(parcels[t+1,:]), drop=True).values, 
                        parcels[t+1,:].where(~np.isnan(parcels[t+1,:]), drop=True).values, 'mean', 
                        bins=[y_bins, x_bins]).statistic
    

    bw.to_dataset(name='sea_ice_thickness').to_netcdf(outfp + str(year) + '_slice_basinwide.nc')

