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
        return json.JSONDecoder.__init__(self, object_hook=self.object_hook,
                                         *args, **kwargs)

    def parse_fields(self, key, value):
        if key == 'startnodes':
            return np.asarray(value, dtype=np.int64)
        elif key == 'endnodes':
            return np.asarray(value, dtype=np.int64)
        elif key == 'K':
            return np.asarray(value, dtype=np.float64)
        elif key == 'X':
            return np.asarray(value, dtype=np.float64)
        elif key == 'o_t':
            return np.asarray(value, dtype=np.float64)
        elif key == 'datetime':
            return pd.to_datetime(value)
        elif key == 'timedelta':
            return pd.to_timedelta(value)
        return value
        
    def object_hook(self, obj):
        if isinstance(obj, dict):
            return {k : self.parse_fields(k, v) for k, v, in obj.items()}
        else:
            return obj

