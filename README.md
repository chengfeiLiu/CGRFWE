## All code in "Coal-gas risk fusion early warning based on explainable deep learning: A case study in mining energy systems"
## Usage: 
The training interface is implemented in trainInter.py. Modify the following settings in default_args_dict:
    "data_root": "./data/UEA_multivariate",
    "model": "SBM",
    "dnn_type": "FCN",
    "seq_len": 128, 
    "folder_path": "ourtrain/testdata", 
    "data_columns": ['T0', 'dust', 'temp', 'wind speed'],
    "label_column": "label"
These parameters can be customized to train different models, use different datasets, or test various parameters. After training, a checkpoints folder will be generated.
During the testing phase, use the 'visual-changeoutputinfo_dataoutput.py' script to generate SHAPELETS matching diagrams, MACN weight attention distribution cloud plots, and data files.
