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
        elif isinstance(obj, np.int_):
            return int(obj)
        elif isinstance(obj, np.float_):
            return float(obj)
        # Let the base class default method raise the TypeError
        return super().default(obj)

class ModelDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        return json.JSONDecoder.__init__(self, object_hook=self.object_hook,
                                         *args, **kwargs)

    def parse_fields(self, key, value):
        match key:
            case 'datetime':
                return pd.to_datetime(value)
            case 'timedelta':
                return pd.to_timedelta(value)
            case 'startnodes':
                return np.asarray(value, dtype=np.int64)
            case 'endnodes':
                return np.asarray(value, dtype=np.int64)
            case 'K':
                return np.asarray(value, dtype=np.float64)
            case 'X':
                return np.asarray(value, dtype=np.float64)
            case 'o_t':
                return np.asarray(value, dtype=np.float64)
            case 'A_s':
                return np.asarray(value, dtype=np.float64)
            case 'C_w':
                return np.asarray(value, dtype=np.float64)
            case 'L':
                return np.asarray(value, dtype=np.float64)
            case 'L_d':
                return np.asarray(value, dtype=np.float64)
            case 'h_max':
                return np.asarray(value, dtype=np.float64)
            case 'h_w':
                return np.asarray(value, dtype=np.float64)
            case 'h_o':
                return np.asarray(value, dtype=np.float64)
            case 'C_o':
                return np.asarray(value, dtype=np.float64)
            case 'O_a':
                return np.asarray(value, dtype=np.float64)
            case 'h_t':
                return np.asarray(value, dtype=np.float64)
        return value
        
    def object_hook(self, obj):
        if isinstance(obj, dict):
            return {k : self.parse_fields(k, v) for k, v, in obj.items()}
        else:
            return obj

