# Reproducing our radio interferometry results

This .md file has all the basic information required to reproduce the results from the [RIGAN paper](https://arxiv.org/abs/2507.21270).

## Setup Instructions
Install the required modules via anaconda:
```
conda create --name <env> --file requirements.txt
```
Note that you may need to change the PyTorch version, depending on your CUDA distribution.

## Update Config
Update ```configs/radio_meerkat_macro.yml``` or ```configs/radio_meerkat_macro_gradient.yml``` with the path to your data, where you want to store checkpoints, and with the path to additional data such as masks.

# Logging
By default, our model is tracked by Weights and Biases platform. See [their documentation](https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

## Weight and biases

Parameters and environmental variables
WANDB_CACHE_DIR
WANDB_DATA_DIR
will need to be updated to your desired directory in the .sh files in ```mass_map_utils/jobs``` using the following command:
``` bash
export WANDB_DIR=<directory_name>/wandb/logs
export WANDB_CACHE_DIR=<directory_name>/wandb/.cache/wandb
export WANDB_CONFIG_DIR=<directory_name>/wandb/.config/wandb
```
where <directory_name> is the name of the directory you wish to save your logs to.

logs -> `./wandb` -> `WANDB_DIR`
artifacts -> `~/.cache/wandb` -> `WANDB_CACHE_DIR`
configs -> `~/.config/wandb` -> `WANDB_CONFIG_DIR`

## Data
Examples on how the data is created can be found in the ```radio-interferometry/scripts/create_radio_data.py``` and ```radio-interferometry/scripts/create_30dor_data.py``` files.

## Training
Training is as simple as running the following command:
```python
python train.py --config ./configs/radio_meerkat_macro.yml --exp-name training_name --num-gpus X
```
where training_name will be used to access checkpoints for validation/testing/plotting, and for tracking weights and biases via wandb. ```X``` is the number of GPUs you plan to use. 

See wandb documentation (https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

If you need to resume training at any point, use the following command:
```python
python train.py --config ./configs/radio_meerkat_macro.yml --exp-name training_name --num-gpus X --resume --resume-epoch Y
```
where ```Y``` is the epoch to resume from.

By default, we save the previous 50 epochs. Ensure that your checkpoint path points to a location with sufficient disk space.
If disk space is a concern, 50 can be reduced to 25.

For details specific to multi-GPU runs and batch size tuning please refer to ```docs/comments.md```.

## Validation
During training, validation is necessary in order to update the weight applied to
the standard deviation reward. There are a variety of metrics that can be assessed during validation, it is up to you what you'd rather use to choose the best training epoch. By default the model will select the model which the best RMSE results. Once completed, all other checkpoints will be automatically deleted this - to toggle this edit the end of the  ```validation.py``` file.

To validate, run the following command:
```python
python ./scripts/radio/validate.py --config ./configs/radio_meerkat_macro.yml --exp-name RIGAN_training
```

## Generating Posterior Samples
To generate figures similar to those found in our paper, execute the following command:
```python
python ./scripts/radio/plot.py --config ./configs/radio_meerkat_macro.yml --exp-name RIGAN_training
```
This script will generate posterior samples, then save the associated reconstruction and uncertainty estimates.

## Running the models of the paper
In order to run the trained models from the paper, you will need to download the supplementary files from [zenodo](https://zenodo.org/records/16529320). They contain the correct checkpoints, configs and normalisation parameters to run both the regular and gradient-based RIGAN models. Using those files, you can run the script that predicts the 30 Doradus region.

For the U-Net RIGAN model, run the following command:
```python
python ./scripts/radio/plot_30dor.py --config ./radio_30dor.yaml --exp-name radio_30dor
```

For the GU-Net RIGAN model, run the following command:
```python
python ./scripts/radio/plot_30dor.py --config ./radio_30dor_gradient.yaml --exp-name radio_30dor_gradient
```