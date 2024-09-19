import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

import numpy as np # type: ignore
import pandas as pd # type: ignore
import dill # type: ignore

def save_obj(file_path, data_frame):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(data_frame, f)
    except Exception as e:
        raise CustomException(e,sys)