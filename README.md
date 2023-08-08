_This code is a supplementary material to the paper "TLIO: Tight Learned Inertial Odometry". To use the code here requires the user to generate its own dataset and retrain. For more information about the paper and the video materials, please refer to our [website](https://cathias.github.io/TLIO/)._


# Installation
All dependencies can be installed using conda via
```shell script
conda env create -f environment.yaml
```
Then the virtual environment is accessible with:
```shell script
conda activate tlio
```

Next commands should be run from this environment.

# Dataset
A dataset is needed in numpy format to run with this code. 
We have released the dataset used in the paper.
The data can be downloaded [here](https://drive.google.com/file/d/10Bc6R-s0ZLy9OEK_1mfpmtDg3jIu8X6g/view?usp=share_link) or with the following command (with the conda env activated) at the root of the repo:
```shell script
gdown 14YKW7PsozjHo_EdxivKvumsQB7JMw1eg
mkdir -p local_data/ # or ln -s /path/to/data_drive/ local_data/
unzip golden-new-format-cc-by-nc-with-imus-v1.5.zip -d local_data/
rm golden-new-format-cc-by-nc-with-imus-v1.5.zip
```
https://drive.google.com/file/d/14YKW7PsozjHo_EdxivKvumsQB7JMw1eg/view?usp=share_link
The dataset tree structure looks like this.
Assume for the examples we have extracted the data under root directory `local_data/tlio_golden`:
```
local_data/tlio_golden
├── 1008221029329889
│   ├── calibration.json
│   ├── imu0_resampled_description.json
│   ├── imu0_resampled.npy
│   └── imu_samples_0.csv
├── 1014753008676428
│   ├── calibration.json
│   ├── imu0_resampled_description.json
│   ├── imu0_resampled.npy
│   └── imu_samples_0.csv
...
├── test_list.txt
├── train_list.txt
└── val_list.txt
```

`imu0_resampled.npy` contains calibrated IMU data and processed VIO ground truth data.
`imu0_resampled_description.json` describes what the different columns in the data are.
The test sequences contain `imu_samples_0.csv` which is the raw IMU data for running the filter. 
`calibration.json` contains the offline calibration. 
Attitude filter data is not included with the release.

# Network training and evaluation

## For training or evaluation of one model

There are three different modes for the network part.`--mode` parameter defines the behaviour of `main_net.py`. Select between `train`, `test` \
`train`: training a network model with training and validation dataset. \
`test`: running an existing network model on testing dataset to obtain concatenated trajectories and metrics. \

### 1. Training:

**Parameters:** 

`--root_dir`: dataset root directory. Each subfolder of root directory is a dataset. \
`--out_dir`: training output directory, where `checkpoints` and `logs` folders will be created to store trained models and tensorboard logs respectively. A `parameters.json` file will also be saved.

**Example:** 
```shell script
python3 src/main_net.py \
--mode train \
--root_dir local_data/tlio_golden \
--out_dir models/resnet \
--epochs 100
```

**Note:** We offer multiple types of dataloaders to help speed up training.
The `--dataset_style` arg can be `ram`, `mmap`, or `iter`. 
`ram` stores all the sequences in RAM, `mmap` uses memmapping to only keep part of the 
sequences in RAM at once, and `iter` is an iterable-style dataloader for larger datasets,
which sacrifices true randomness. 
The default is `mmap`, which offers the best tradeoff, and typically works for 
the dataset provided.
However, we found that on server-style machines, the memmapping can cause RAM to fill up
for some reason (it seems to work best on personal desktops).
If the training is getting killed by your OS or taking up too much RAM,
you may try setting `--workers` to 1, `--dataset_style` to `ram`, and/or `--no-persistent_workers`.

```shell script
tensorboard --logdir models/resnet/logs/
```

### 2. Testing:

**Parameters:** 

`--model_path`: path of the trained model to test with. \
`--out_dir`: testing output directory, where a folder for each dataset tested will be created containing estimated trajectory as `trajectory.txt` and plots if specified. `metrics.json` contains the statistics for each dataset. 

**Example:**
```shell script
python3 src/main_net.py \
--mode test \
--root_dir local_data/tlio_golden \
--model_path models/resnet/checkpoint_*.pt \
--out_dir test_outputs
```


**Warning:** network testing use the ground truth orientations for displacement integration. 
Please do not consider them as benchmarks, they are more like a debugging tool.

## For batch testing on multiple models

Batch scripts are under src/batch_analysis module. Execute batch scripts from the src folder.

Batch testing tests a list of datasets using multiple models and for each model save the trajectories, plots and metrics into a separate model folder. Output tree structure looks like this:
```
batch_test_outputs
├── model1
│   ├── seq1
│   │   ├── trajectory.txt
│   │   └── *.png
│   ├── seq2
...
│   └── metrics.json
├── model2
│   ├── seq1
...
│   └── metrics.json
...
```

Create an output directory and go to the src folder
```shell script
mkdir batch_test_outputs
cd src
```
Run batch tests. `--model_globbing` is the globbing pattern to find all models to test. Here we only have one.
```shell script
python -m batch_runner.net_test_batch \
--root_dir ../local_data/tlio_golden \
--model_globbing "../models/*/checkpoint_best.pt" \
--out_dir ../batch_test_outputs \
--save_plot
```
If you saved plot, you can visualize there:

```shell script
feh ../	batch_test_outputs/models-resnet/*/view.png # example using the `feh` image visualizer, use your favorite one
```


## Running analysis and generating plots

After running testing and evaluation in batches, the statistics are saved in either `metrics.json`. To visualize the results and compare between models, we provide scripts that display the results in an interactive shell through iPython. The scripts are under `src/analysis` module.

To visualize network testing results from `metrics.json` including trajectory metrics and testing losses, go to `src` folder and run
```shell script
python -m analysis.display_json \
--glob_dataset "../batch_test_outputs/*/"
```
This will leave you in an interactive shell with a preloaded panda DataFrame `d`. You can use it to visualize all metrics with the following helper function:
```shell script
plot_all_stats_net(d)
```

# Running EKF with network displacement estimates

## Converting model to torchscript
The EKF expects the model to be in torchscript format.

**Example:**
From the repo root:
```shell script
python3 src/convert_model_to_torchscript.py \
--model_path models/resnet/checkpoint_best.pt \
--model_param_path models/resnet/parameters.json \
--out_dir models/resnet/
```
which will create `models/resnet/model_torchscript.pt`.


## Running EKF with one network model
 
Use `src/main_filter.py` for running the filter and parsing parameters. The program supports running multiple datasets on one specified network model.

**Parameters:**

`--model_path`: path to saved model checkpoint file. \
`--model_param_path`: path to parameter json file for this model. \
`--out_dir`: filter output directory. This will include a `parameters.json` file with filter parameters, and a folder for each dataset containing the logged states, default to `not_vio_state.txt`. \
`--erase_old_log`: overwrite old log files. If set to `--no-erase_old_log`, the program would skip running on the datasets if the output file already exists in the output directory. \
`--save_as_npy`: convert the output txt file to npy file and append file extension (e.g. `not_vio_state.txt.npy`) to save space. \
`--initialize_with_offline_calib`: initialize with offline calibration of the IMU. If set to `--no-initialize_with_offline_calib` the initial IMU biases will be initialized to 0. \
`--visualize`: if set, open up an Open3D window to visualize the filter running. This is of course optional.

**Example:**
```shell script
python3 src/main_filter.py \
--root_dir local_data/tlio_golden \
--model_path models/resnet/model_torchscript.pt \
--model_param_path models/resnet/parameters.json \
--out_dir filter_outputs \
--erase_old_log \
--save_as_npy \
--initialize_with_offline_calib \
--dataset_number 22 \
--visualize
```
Please refer to `main_filter.py` for a full list of parameters.

## Batch running filter on multiple models and parameters

Batch script `batch_runner/filter_batch` provides functionality to run the main file in batch settings. Go to `src` folder to run the module and you can set the parameters to test within the script (e.g. different update frequencies).

**Example:**
```shell script
cd src
python -m batch_runner.filter_batch \
--root_dir ../local_data/tlio_golden \
--model_globbing "../models/*/model_torchscript.pt" \
--out_dir ../batch_filter_outputs
```

## Batch running metrics and plot generation

To generate plots of the states of the filter and to generate `metrics.json` file for both the filter and network concatenation approaches, batch run `plot_state.py` on the existing filter and network testing outputs.

**Parameters:**

`--runname_globbing`: globbing pattern for all the model names to plot. This pattern should match between filter and ronin and exist in both `--filter_dir` and `--ronin_dir`. \
`--no_make_plots`: not to save plots. If removed plots will be saved in the filter output folders for each trajectory.

**Example:**
```shell script
python -m batch_runner.plot_batch \
--root_dir ../local_data/tlio_golden \
--runname_globbing "*" \
--filter_dir ../batch_filter_outputs_uf20 \
--ronin_dir ../batch_test_outputs
```

Up to now a `metrics.json` file will be added to each model folder, and the tree structure would look like this:
```
batch_filter_outputs
├── model1
│   ├── seq1
│   │   ├── *.png
│   │   ├── not_vio_state.txt.npy
│   │   └── vio_states.npy
│   ├── seq1
│   │   ├── *.png
│   │   ├── not_vio_state.txt.npy
│   │   └── vio_states.npy
...
│   ├── metrics.json
│   └── parameters.json
├── model2
...
```

Visualize the plot from the filter and ronin:
```shell script
feh ../batch_filter_outputs_uf20/models-resnet/*/position-2d.png # example using the `feh` image visualizer, use your favorite one
```

To generate plots from the metrics:
```shell script
python -m analysis.display_json \
--glob_dataset "../batch_filter_outputs_uf20/*/"

# then run in the interactive session
# plot_sysperf_cdf(d)
```
