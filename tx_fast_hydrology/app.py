import time
import asyncio
import numpy as np
import pandas as pd
import requests
from tx_fast_hydrology.muskingum import ModelCollection
from tx_fast_hydrology.simulation import AsyncSimulation, CheckPoint
from tx_fast_hydrology.download import get_forcing_directories, get_forecast_path, download_nwm_forcings, download_nwm_streamflow
from sanic import Sanic, response, json, text, empty
from sanic.log import logger
from sanic.worker.manager import WorkerManager

# Create server app
APP_NAME = 'muskingum'
app = Sanic(APP_NAME)
# Set timeout threshold for app startup
WorkerManager.THRESHOLD = 1200

# Response codes
OK = 200
CREATED = 201
BAD_REQUEST = 400

# Load COMIDS
# TODO: Clean this up
comids = pd.read_csv('/Users/mdbartos/Git/tx-fast-hydrology/notebooks/COMIDS.csv',
                        index_col=0)['0'].values


async def tick(app):
    tick_dt = app.ctx.tick_dt
    simulation = app.ctx.simulation
    last_timestamp = app.ctx.current_timestamp
    logger.info(f'Sleeping for {tick_dt} seconds...')
    await asyncio.sleep(tick_dt)
    urls = get_forcing_directories()
    nwm_dir, forecast_hour, timestamp = get_forecast_path(urls)
    if timestamp > last_timestamp:
        logger.info(f'New forcings available at timestamp {timestamp}')
        inputs = download_nwm_forcings(nwm_dir, forecast_hour=forecast_hour, comids=comids)
        logger.info(f'Forcings downloaded')
        simulation.load_states()
        simulation.inputs = simulation.load_inputs(inputs)
        logger.info('Beginning simulation...')
        # Step model forward in time
        outputs = await simulation.simulate()
        outputs = pd.concat([series for series in outputs.values()], axis=1)
        logger.info('Simulation finished')
        app.ctx.simulation = simulation
        app.ctx.outputs = outputs
        app.ctx.current_timestamp = timestamp
    # Queue next loop
    app.add_task(tick, name='tick')

@app.before_server_start
async def start_model(app, loop):
    # Create model
    input_path = '/Users/mdbartos/Git/tx-fast-hydrology/notebooks/tmp/huc6.cfg'
    # Set app parameters
    app.ctx.tick_dt = 600.0
    # Create model collection
    model_collection = ModelCollection.from_file(input_path)
    logger.info('Model collection loaded')
    # Set checkpoint callbacks
    for model in model_collection.models.values():
        checkpoint = CheckPoint(model, timedelta=3600.)
        model.bind_callback(checkpoint, key='checkpoint')
    # Download latest forcings and streamflows
    logger.info('Downloading initial NWM forcings and streamflows...')
    urls = get_forcing_directories()
    nwm_dir, forecast_hour, timestamp = get_forecast_path(urls)
    streamflow = download_nwm_streamflow(nwm_dir, forecast_hour=forecast_hour, comids=comids)
    logger.info('Initial streamflow downloaded')
    inputs = download_nwm_forcings(nwm_dir, forecast_hour=forecast_hour, comids=comids)
    logger.info('Initial forcings downloaded')
    # Create simulation class instance
    simulation = AsyncSimulation(model_collection, inputs)
    logger.info('Simulation instance created')
    timestamp = streamflow.index.item()
    streamflow_values = streamflow.loc[timestamp]
    # Set timestamp and initial states for all models
    simulation.set_datetime(timestamp)
    simulation.init_states(streamflow_values)
    simulation.save_states()
    logger.info('Model states initialized')
    # Run initial simulation
    logger.info('Beginning initial simulation...')
    outputs = await simulation.simulate()
    outputs = pd.concat([series for series in outputs.values()], axis=1)
    logger.info('Initial simulation finished')
    # Add persistent objects to global app context
    app.ctx.simulation = simulation
    app.ctx.outputs = outputs
    app.ctx.current_timestamp = timestamp
    # Start loop
    app.add_task(tick)

@app.get("/forecast/<reach_id>")
async def reach_forecast(request, reach_id):
    outputs = app.ctx.outputs
    reach_id = str(reach_id)
    timestamp_utc = [index.isoformat() for index in outputs.index]
    streamflow_cms = [value for value in outputs[reach_id].values]
    json_output = {'timestamp__utc' : timestamp_utc, 
                   'streamflow__cms' : streamflow_cms}
    return json(json_output)

if __name__ == "__main__":
    app.run()