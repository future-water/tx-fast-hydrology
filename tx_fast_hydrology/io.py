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

