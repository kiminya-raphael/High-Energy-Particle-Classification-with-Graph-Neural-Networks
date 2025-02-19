# [High-Energy Particle Classification with Graph Neural Networks](https://www.biendata.xyz/competition/jet/)

# Minimum recommended environment specs
OS: Ubuntu 18.04 
RAM: 32GB
GPU: Tesla P100 16GB

# Deployment 
Install [Anaconda](https://www.anaconda.com)

### From Anaconda Prompt, cd to the code directory and create 'baai' conda environment:
`conda env create -f enviroment.yaml`

### Activate environment
`conda activate baai`

## Process train datasets (optional)
#### Run this only if you want to retrain the model on new data

`python process_dataset.py ./input/jet_complex_data ./train_processed`

### Train models (optional)

#### Run this only if you want to retrain the model on new data. Trained models will be saved to ./model directory

`python train.py ./train_processed 1 850 20 model_1_1`
`python train.py ./train_processed 2 850 20 model_1_2`
`python train.py ./train_processed 1 1500 15 model_2_1`
`python train.py ./train_processed 2 1500 30 model_2_2`


### Evaluate model on test set. Predictions will be saved to ./submission.csv

`python test.py ./input/jet_complex_data/ ./ ./model`

