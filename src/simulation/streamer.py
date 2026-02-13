from config.paths import TRAIN_FILE, SETTINGS_FILE
import pandas as pd
#import time
import json

class SensorStreamer:
    def __init__(self, engine_id=1):
        # NASA train data column names
        self.columns = json.load(open(SETTINGS_FILE)).get('data_col_names')
        self.data = pd.read_csv(TRAIN_FILE, sep='\s+', header=None, names=self.columns)

        # Gets the desired engine with provided engine id
        self.engine_data = self.data[self.data['unit_number'] == engine_id]
        self.max_cycles = self.engine_data['cycle'].max()
        self.current_cycle = 0

    def stream(self):
        """
        Yields the engine data at each call.
        """
        for _, row in self.engine_data.iterrows():
            data_packet = row.to_dict()
            yield data_packet

            # time.sleep(0.1)