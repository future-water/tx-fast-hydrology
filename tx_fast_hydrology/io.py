import re
import json
import uuid
import numpy as np
import pandas as pd

DEFAULT_START_TIME = pd.to_datetime(0., utc=True)
DEFAULT_TIMEDELTA = pd.to_timedelta(3600, unit='s')

class ModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return super().default(obj)

class ModelDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        iso_timestamp = '^(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))$'
        iso_timedelta = '^P(\d+Y)?(\d+M)?(\d+D)?T(\d+H)?(\d+M)?([0-9.]S)?$'
        self.timestamp_regex = re.compile(iso_timestamp)
        self.timedelta_regex = re.compile(iso_timedelta)
        return json.JSONDecoder.__init__(self, object_hook=self.object_hook,
                                         *args, **kwargs)

    def parse_datetimes(self, value):
        if isinstance(value, str):
            if self.timestamp_regex.match(value):
                return pd.to_datetime(value)
            elif self.timedelta_regex.match(value):
                return pd.to_timedelta(value)
        return value
        
    def object_hook(self, dct):
        return {k : self.parse_datetimes(v) for k, v, in dct.items()}

def load_model_file(file_path, load_optional=True):
    required_fields = {'name', 'datetime', 'timedelta', 'reach_ids',
     'startnodes', 'endnodes', 'K', 'X', 'o_t'}
    with open(file_path, 'r') as f:
        obj = json.load(f, cls=ModelDecoder)
    try:
        assert required_fields.issubset(set(obj.keys()))
    except:
        raise ValueError(f'Model field must contain fields {required_fields}')
    obj['startnodes'] = np.asarray(obj['startnodes'], dtype=np.int64)
    obj['endnodes'] = np.asarray(obj['endnodes'], dtype=np.int64)
    obj['K'] = np.asarray(obj['K'], dtype=np.float64)
    obj['X'] = np.asarray(obj['X'], dtype=np.float64)
    obj['o_t'] = np.asarray(obj['o_t'], dtype=np.float64)
    try:
        assert (obj['startnodes'].size == obj['endnodes'].size 
                == obj['K'].size == obj['X'].size == obj['o_t'].size)
        obj['n'] = obj['startnodes'].size
    except:
        raise ValueError('Arrays are not the same length')
    if load_optional:
        return obj
    else:
        obj = {k : v for k, v in obj.items()
               if not k in required_fields}
        return obj

def dump_model_file(obj, file_path, dump_optional=True):
    required_fields = {'name', 'datetime', 'timedelta', 'reach_ids',
     'startnodes', 'endnodes', 'K', 'X', 'o_t'}
    with open(file_path, 'w') as f:
        if dump_optional:
            to_dump = obj
        else:
            to_dump = {k : v for k, v in obj.items()
                        if k in required_fields}
        json.dump(to_dump, f, cls=ModelEncoder)

def load_nhd_geojson(d):
    node_ids = [i['attributes']['COMID'] for i in d['features']]
    link_ids = [i['attributes']['COMID'] for i in d['features']]
    paths = [np.asarray(i['geometry']['paths']) for i in d['features']]
    reach_ids = link_ids
    reach_ids = [str(x) for x in reach_ids]
    source_node_ids = [i['attributes']['COMID'] for i in d['features']]
    target_node_ids = [i['attributes']['toCOMID'] for i in d['features']]
    dx = np.asarray([i['attributes']['Shape_Length'] for i in d['features']]) #* 1000 # km to m
    # NOTE: node_ids and source_node_ids should always be the same for NHD
    # But not necessarily so for original json files
    node_ids = node_ids
    source_node_ids = source_node_ids
    target_node_ids = target_node_ids
    self_loops = []
    node_index_map = pd.Series(np.arange(len(node_ids)), index=node_ids)
    startnodes = node_index_map.reindex(source_node_ids, fill_value=-1).values
    endnodes = node_index_map.reindex(target_node_ids, fill_value=-1).values
    for i in range(len(startnodes)):
        if endnodes[i] == -1:
            self_loops.append(i)
            endnodes[i] = startnodes[i]
    n = startnodes.size
    obj = {
        'name' : str(uuid.uuid4()),
        'datetime' : DEFAULT_START_TIME,
        'timedelta' : DEFAULT_TIMEDELTA,
        'reach_ids' : reach_ids,
        'startnodes' : startnodes,
        'endnodes' : endnodes,
        'K' : 3600 * np.ones(n, dtype=np.float64),
        'X' : 0.29 * np.ones(n, dtype=np.float64),
        'o_t' : 1e-3 * np.ones(n, dtype=np.float64),
        'dx' : dx,
        'paths' : paths
    }
    return obj
