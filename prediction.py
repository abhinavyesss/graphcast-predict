#pyright: reportMissingImports=false

import cdsapi
import datetime
import functools
from google.cloud import storage
from graphcast import autoregressive, casting, checkpoint, data_utils as du, graphcast, normalization, rollout
import haiku as hk
import isodate
import jax
import math
import netCDF4
import numpy as np
import os
import pandas as pd
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude
import pytz
from typing import Dict
import warnings
import xarray
import zipfile
warnings.filterwarnings('ignore')

from constants import Constants

client = cdsapi.Client()

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket('dm_graphcast')

singlelevelfields = {
                        'u10': '10m_u_component_of_wind',
                        'v10': '10m_v_component_of_wind',
                        't2m': '2m_temperature',
                        'z': 'geopotential',
                        'lsm': 'land_sea_mask',
                        'msl': 'mean_sea_level_pressure',
                        'tisr': 'toa_incident_solar_radiation',
                        'tp': 'total_precipitation'
                    }
pressurelevelfields = {
                        'u': 'u_component_of_wind',
                        'v': 'v_component_of_wind',
                        'z': 'geopotential',
                        'q': 'specific_humidity',
                        't': 'temperature',
                        'w': 'vertical_velocity'
                    }
predictionFields = [
                        'u_component_of_wind',
                        'v_component_of_wind',
                        'geopotential',
                        'specific_humidity',
                        'temperature',
                        'vertical_velocity',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '2m_temperature',
                        'mean_sea_level_pressure',
                        'total_precipitation_6hr'
                    ]
pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
pi = math.pi
gap = 6
predictions_steps = 4
watts_to_joules = 3600
first_prediction = datetime.datetime(2025, 1, 1, 18, 0)
lat_range = range(-90, 91, 1)
lon_range = range(0, 360, 1)

class AssignCoordinates:
    
    coordinates = {
                    '2m_temperature': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'time'],
                    'mean_sea_level_pressure': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'time'],
                    '10m_v_component_of_wind': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'time'],
                    '10m_u_component_of_wind': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'time'],
                    'total_precipitation_6hr': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'time'],
                    'temperature': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'level', 'time'],
                    'geopotential': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'level', 'time'],
                    'u_component_of_wind': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'level', 'time'],
                    'v_component_of_wind': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'level', 'time'],
                    'vertical_velocity': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'level', 'time'],
                    'specific_humidity': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'level', 'time'],
                    'toa_incident_solar_radiation': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'lat', 'time'],
                    'year_progress_cos': [Constants.Graphcast.BATCH_FIELD.value, 'time'],
                    'year_progress_sin': [Constants.Graphcast.BATCH_FIELD.value, 'time'],
                    'day_progress_cos': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'time'],
                    'day_progress_sin': [Constants.Graphcast.BATCH_FIELD.value, 'lon', 'time'],
                    'geopotential_at_surface': ['lon', 'lat'],
                    'land_sea_mask': ['lon', 'lat'],
                }

print('Connecting to dm_graphcast bucket...')
with gcs_bucket.blob(f'params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz').open('rb') as model:
    ckpt = checkpoint.load(model, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

print('Loading the diffs_stddev_by_level.nc file...')
with open(r'model/stats/diffs_stddev_by_level.nc', 'rb') as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()

print('Loading the mean_by_level.nc file...')
with open(r'model/stats/mean_by_level.nc', 'rb') as f:
    mean_by_level = xarray.load_dataset(f).compute()

print('Loading the stddev_by_level.nc file...')
with open(r'model/stats/stddev_by_level.nc', 'rb') as f:
    stddev_by_level = xarray.load_dataset(f).compute()
    
def construct_wrapped_graphcast(model_config:graphcast.ModelConfig, task_config:graphcast.TaskConfig):
    
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(predictor, diffs_stddev_by_level = diffs_stddev_by_level, mean_by_level = mean_by_level, stddev_by_level = stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing = True)
    
    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    
    predictor = construct_wrapped_graphcast(model_config, task_config)
    
    return predictor(inputs, targets_template = targets_template, forcings = forcings)

def with_configs(fn):

    return functools.partial(fn, model_config = model_config, task_config = task_config)

def with_params(fn):

    return functools.partial(fn, params = params, state = state)

def drop_state(fn):

    return lambda **kw: fn(**kw)[0]

run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

class Predictor:

    @classmethod
    def predict(cls, inputs, targets, forcings) -> xarray.Dataset:
        
        predictions = rollout.chunked_prediction(run_forward_jitted, rng = jax.random.PRNGKey(0), inputs = inputs, targets_template = targets, forcings = forcings)

        return predictions

# Converting the variable to a datetime object.
def toDatetime(dt) -> datetime.datetime:

    if isinstance(dt, datetime.date) and isinstance(dt, datetime.datetime):

        return dt
    
    elif isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
        
        return datetime.datetime.combine(dt, datetime.datetime.min.time())
    
    elif isinstance(dt, str):

        if 'T' in dt:
            return isodate.parse_datetime(dt)
        else:
            return datetime.datetime.combine(isodate.parse_date(dt), datetime.datetime.min.time())

def nans(*args) -> list:

    return np.full((args), np.nan)

def deltaTime(dt, **delta) -> datetime.datetime:

    return dt + datetime.timedelta(**delta)

def addTimezone(dt, tz = pytz.UTC) -> datetime.datetime:

    dt = toDatetime(dt)
    if dt.tzinfo == None:
        return pytz.UTC.localize(dt).astimezone(tz)
    else:
        return dt.astimezone(tz)

def remove_junk_columns(df:pd.DataFrame):

    for col in ['number', 'expver']:
        if col in df.columns.values.tolist():
            df.pop(col)
    
    return df

def getSingleLevelValues(filename):

    extract_to = filename.split('.')[0]
    with zipfile.ZipFile(filename, 'r') as f:
        f.extractall(extract_to)

    dfs = []
    for i in os.listdir(extract_to):
        extension = i.split('.')[-1]
        if extension == 'nc':
            df = xarray.open_dataset('{}/{}'.format(extract_to, i), engine = netCDF4.__name__.lower()).to_dataframe()
            df = remove_junk_columns(df)
            dfs.append(df)

    single_level_df = pd.concat(dfs, axis = 1)

    return single_level_df

# Getting the single and pressure level values.
def getSingleAndPressureValues():
    
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': list(singlelevelfields.values()),
            'grid': '1.0/1.0',
            'year': [2025],
            'month': [1],
            'day': [1],
            'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00'],
            'data_format': 'netcdf',
            'download_format': 'zip'
        }
    ).download('single-level.zip')
    singlelevel = getSingleLevelValues('single-level.zip')
    singlelevel = singlelevel.rename(columns = {col:singlelevelfields[col] for col in singlelevel.columns.values.tolist() if col in singlelevelfields})
    singlelevel = singlelevel.rename(columns = {'geopotential': 'geopotential_at_surface'})

    # Calculating the sum of the last 6 hours of rainfall.
    singlelevel = singlelevel.sort_index()
    singlelevel['total_precipitation_6hr'] = singlelevel.groupby(level=[0, 1])['total_precipitation'].rolling(window = 6, min_periods = 1).sum().reset_index(level=[0, 1], drop=True)
    singlelevel.pop('total_precipitation')
    
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': list(pressurelevelfields.values()),
            'grid': '1.0/1.0',
            'year': [2025],
            'month': [1],
            'day': [1],
            'time': ['06:00', '12:00'],
            'pressure_level': pressure_levels,
            'data_format': 'netcdf',
            'download_format': 'unarchived'
        }
    ).download('pressure-level.nc')
    pressurelevel = xarray.open_dataset('pressure-level.nc', engine = netCDF4.__name__.lower()).to_dataframe()
    pressurelevel = remove_junk_columns(pressurelevel)
    pressurelevel = pressurelevel.rename(columns = {col:pressurelevelfields[col] for col in pressurelevel.columns.values.tolist() if col in pressurelevelfields})

    return singlelevel, pressurelevel

# Adding sin and cos of the year progress.
def addYearProgress(secs, data):

    progress = du.get_year_progress(secs)
    data['year_progress_sin'] = math.sin(2 * pi * progress)
    data['year_progress_cos'] = math.cos(2 * pi * progress)

    return data

# Adding sin and cos of the day progress.
def addDayProgress(secs, lon:str, data:pd.DataFrame):

    lons = data.index.get_level_values(lon).unique()
    progress:np.ndarray = du.get_day_progress(secs, np.array(lons))
    prxlon = {lon:prog for lon, prog in list(zip(list(lons), progress.tolist()))}
    data['day_progress_sin'] = data.index.get_level_values(lon).map(lambda x: math.sin(2 * pi * prxlon[x]))
    data['day_progress_cos'] = data.index.get_level_values(lon).map(lambda x: math.cos(2 * pi * prxlon[x]))
    
    return data

def integrateProgress(data:pd.DataFrame):
        
    for dt in data.index.get_level_values(Constants.CDSConstants.TIME_FIELD.value).unique():
        seconds_since_epoch = toDatetime(dt).timestamp()
        data = addYearProgress(seconds_since_epoch, data)
        data = addDayProgress(seconds_since_epoch, 'longitude' if 'longitude' in data.index.names else 'lon', data)

    return data

def getSolarRadiation(longitude, latitude, dt):
        
    altitude_degrees = get_altitude(latitude, longitude, addTimezone(dt))
    solar_radiation = get_radiation_direct(dt, altitude_degrees) if altitude_degrees > 0 else 0

    return solar_radiation * watts_to_joules

def integrateSolarRadiation(data:pd.DataFrame):

    dates = list(data.index.get_level_values(Constants.CDSConstants.TIME_FIELD.value).unique())
    coords = [[lat, lon] for lat in lat_range for lon in lon_range]
    values = []

    for dt in dates:
        values.extend(list(map(lambda coord:{Constants.CDSConstants.TIME_FIELD.value: dt, Constants.CDSConstants.LON_FIELD.value: coord[1], Constants.CDSConstants.LAT_FIELD.value: coord[0], 'toa_incident_solar_radiation': getSolarRadiation(coord[1], coord[0], dt)}, coords)))

    values = pd.DataFrame(values).set_index(keys = [Constants.CDSConstants.LAT_FIELD.value, Constants.CDSConstants.LON_FIELD.value, Constants.CDSConstants.TIME_FIELD.value])
    
    return pd.merge(data, values, left_index = True, right_index = True, how = 'inner')

def modifyCoordinates(data:xarray.Dataset):
        
    for var in list(data.data_vars):
        varArray:xarray.DataArray = data[var]
        nonIndices = list(set(list(varArray.coords)).difference(set(AssignCoordinates.coordinates[var])))
        data[var] = varArray.isel(**{coord: 0 for coord in nonIndices})
    data = data.drop_vars(Constants.Graphcast.BATCH_FIELD.value)

    return data

def makeXarray(data:pd.DataFrame) -> xarray.Dataset:

    data = data.rename_axis(index={
                Constants.CDSConstants.TIME_FIELD.value: Constants.Graphcast.TIME_FIELD.value,
                Constants.CDSConstants.PRESSURE_FIELD.value: Constants.Graphcast.PRESSURE_FIELD.value
            })
    data = data.to_xarray()
    data = modifyCoordinates(data)

    return data

def formatData(data:pd.DataFrame) -> pd.DataFrame:
        
    data = data.rename_axis(index = {Constants.CDSConstants.LAT_FIELD.value: 'lat', Constants.CDSConstants.LON_FIELD.value: 'lon'})
    if Constants.Graphcast.BATCH_FIELD.value not in data.index.names:
        data[Constants.Graphcast.BATCH_FIELD.value] = 0
        data = data.set_index(Constants.Graphcast.BATCH_FIELD.value, append = True)
    
    return data

def getTargets(dt, data:pd.DataFrame):

    lat, lon, levels, batch = sorted(data.index.get_level_values('lat').unique().tolist()), sorted(data.index.get_level_values('lon').unique().tolist()), sorted(data.index.get_level_values('pressure_level').unique().tolist()), data.index.get_level_values(Constants.Graphcast.BATCH_FIELD.value).unique().tolist()
    time = [deltaTime(dt, hours = days * gap) for days in range(predictions_steps)]
    target = xarray.Dataset({field: (['lat', 'lon', 'level', Constants.CDSConstants.TIME_FIELD.value], nans(len(lat), len(lon), len(levels), len(time))) for field in predictionFields}, coords = {'lat': lat, 'lon': lon, 'level': levels, Constants.CDSConstants.TIME_FIELD.value: time, Constants.Graphcast.BATCH_FIELD.value: batch})

    return target.to_dataframe()

def getForcings(data:pd.DataFrame):

    forcingdf = data.reset_index(level = 'level', drop = True).drop(labels = predictionFields, axis = 1)
    forcingdf = pd.DataFrame(index = forcingdf.index.drop_duplicates(keep = 'first'))
    forcingdf = integrateProgress(forcingdf)
    forcingdf = integrateSolarRadiation(forcingdf)

    return forcingdf

if __name__ == '__main__':

    values:Dict[str, xarray.Dataset] = {}
    
    single, pressure = getSingleAndPressureValues()
    values['inputs'] = pd.merge(pressure, single, left_index = True, right_index = True, how = 'inner')
    values['inputs'] = integrateProgress(values['inputs'])
    values['inputs'] = formatData(values['inputs'])
    values['targets'] = getTargets(first_prediction, values['inputs'])
    values['forcings'] = getForcings(values['targets'])
    values = {value:makeXarray(values[value]) for value in values}

    predictions = Predictor.predict(values['inputs'], values['targets'], values['forcings'])
    predictions.to_dataframe().to_csv('predictions.csv', sep = ',')
