import time
import asyncio
import logging
import numpy as np
import pandas as pd
import requests
from tx_fast_hydrology.muskingum import Muskingum, ModelCollection
from tx_fast_hydrology.simulation import AsyncSimulation
from tx_fast_hydrology.download import get_forcing_directories, get_forecast_path, download_forcings, download_streamflow
from sanic import Sanic, response, json, text, empty

# Start server app
APP_NAME = 'muskingum'
app = Sanic(APP_NAME)
logger = logging.getLogger(APP_NAME)

# Response codes
OK = 200
CREATED = 201
BAD_REQUEST = 400

async def tick(app):
    tick_dt = app.ctx.tick_dt
    await asyncio.sleep(tick_dt)
    # Step model forward in time
    # Queue next loop
    app.add_task(tick, name='tick')

@app.before_server_start
async def start_model(app, loop):
    # Create model
    input_path = '/Users/mdbartos/Git/tx-fast-hydrology/notebooks/tmp/huc6.cfg'
    # Load COMIDS
    # TODO: Clean this up
    comids = pd.read_csv('/Users/mdbartos/Git/tx-fast-hydrology/notebooks/COMIDS.csv',
                         index_col=0)['0'].values
    # Set app parameters
    app.ctx.tick_dt = 60.0
    # Create model collection
    model_collection = ModelCollection.from_file(input_path)
    logger.info('Model collection loaded')
    # Download latest forcings
    urls = get_forcing_directories()
    nwm_dir, forecast_hour = get_forecast_path(urls)
    streamflow = download_streamflow(nwm_dir, forecast_hour=forecast_hour, comids=comids)
    inputs = download_forcings(nwm_dir, forecast_hour=forecast_hour, comids=comids)
    logger.info('Initial forcings loaded')
    # Create simulation class instance
    simulation = AsyncSimulation(model_collection, inputs)
    timestamp = streamflow.index.item()
    streamflow_values = streamflow.loc[timestamp]
    simulation.init_states(streamflow_values)
    outputs = await simulation.simulate()
    # Add persistent objects to global app context
    app.ctx.simulation = simulation
    app.ctx.outputs = outputs

app.add_task(tick)
