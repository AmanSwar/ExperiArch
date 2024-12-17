import pandas as pd


DATA_DIR = ''
TARGET_VAR = ''
FEATURES = []
VARS = []


if len(FEATURES) == 1:
    VARS.append(FEATURES[0])

else:
    for feat in FEATURES:
        VARS.append(feat)






