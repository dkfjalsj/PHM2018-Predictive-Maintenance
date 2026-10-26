"""Read only small headers; do not analyze sensor data or train models."""
from pathlib import Path
import csv, json, re, sys
import numpy as np
import pandas as pd
from src.config import data_root

def check():
    root = data_root() / 'train'
    files = sorted(p for p in root.glob('*.csv') if re.fullmatch(r'\d{2}_M\d{2}_DC_train\.csv', p.name))
    if len(files) != 20:
        raise ValueError('Expected exactly 20 equipment sensor files')
    for p in files:
        fault = root/'train_faults'/('_'.join(p.stem.split('_')[:2])+'_train_fault_data.csv')
        ttf = root/'train_ttf'/p.name
        for q, count in [(p,24),(ttf,4),(fault,3)]:
            with q.open(encoding='utf-8-sig',newline='') as f:
                header = next(csv.reader(f))
            if len(header) != count:
                raise ValueError('Unexpected source schema')
    return {'python':sys.version.split()[0], 'numpy':np.__version__, 'pandas':pd.__version__, 'equipment_triplets':len(files), 'schema_headers_only':True, 'model_training':False}

if __name__ == '__main__':
    print(json.dumps(check(),indent=2))
