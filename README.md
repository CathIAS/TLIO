_This code is a supplementary material to the paper "TLIO: Tight Learned Inertial Odometry". To use the code here requires the user to generate its own dataset and retrain. For more information about the paper and the video materials, please refer to our [website](https://cathias.github.io/TLIO/)._


# Installation
Dependencies tree can be retrieved from `pyproject.toml`.
It is written for the poetry tool. 
All dependencies can thus be installed at once in a new virtual environment with:
```shell script
cd src
poetry install
```
Then the virtual environment is accessible with:
```shell script
poetry shell
```

Alternatively, the dependencies are also specified and can be installed through `requirements.txt`. First create a virtual environment with python3 interpreter, then run
``` 
pip install -r requirements.txt
```

Next commands should be run from this environment.

# Dataset
A dataset is needed in the format of hdf5 to run with this code. The dataset tree structure looks like this under root directory `Dataset`:
```
Dataset
├── test.txt
├── train.txt
├── val.txt
├── seq1
│   ├── atttitude.txt
│   ├── calib_state.txt
│   ├── evolving_state.txt
│   └── data.hdf5
├── seq22
│   ├── atttitude.txt
│   ├── calib_state.txt
│   ├── evolving_state.txt
│   └── data.hdf5
...
```

`data.hdf5` contains raw and calibrated IMU data and processed ground truth data. It is used for both the network and the filter. `calib_state.txt` contains calibration states from VIO and is used for filter initialization. `atttitude.txt` and `evolving_state.txt` are the outputs from AHRS attitude filter and VIO pose estimates. These are not used by the filter, but loaded for comparison / debug purposes.

The generation of `data.hdf5` is specified in `gen_fb_data.py`, which requires interpolated stamped IMU measurement files and time-aligned VIO states files. The user can generate his/her own dataset with a different procedure to obtain the same fields to be used for network training and filter inputs.

## File formats

### Dataset lists

`test.txt`, `train.txt` and `val.txt` are all the same. Each row is the name of a sequence to be used for one of the purposes of test, train or validate. The name should be the same of the sequence directory for example `seq1` and `seq22` as above.


### Used to generate `data.hdf5`
Timestamps (t) are in microseconds (us). Each row corresponds to a single timestamp. All data is delimited by commas.

- `my_timestamps_p.txt` VIO timestamps. Used to select data time range.
  - [ t ]
  - Note: single column, skipped first 20 frames
- `imu_measurements.txt` IMU data, raw and calibrated using VIO calibration
  - [ t, ax_raw, ay_raw, az_raw, ax_calib, ay_calib, az_calib, wx_raw, wy_raw, wz_raw, wx_calib, wy_calib, wz_calib]
  - [t, acc_raw (3), acc_cal (3), gyr_raw (3), gyr_cal (3), has_vio] 
  - Note: has been interpolated evenly between images. Every timestamp in my_timestamps_p.txt will have a corresponding timestamp in this file (has_vio==1)
- `evolving_state.txt` VIO evolving states at IMU rate.
  - [t, q_wxyz (4), p (3), v (3)]
  - Note: not interpolated, raw IMU timestamps. Every timestamp in my_timestamps_p.txt will have a corresponding timestamp in this file
- `calib_state.txt` VIO calibration states at image rate (used in data_io.py)
  - [t, acc_scale_inv (9), gyr_scale_inv (9), gyro_g_sense (9), b_acc (3), b_gyr (3)]
  - Note: A single row with IMU calibration information.
- `atttitude.txt` AHRS attitude from IMU
  - [ t, qw, qx, qy, qz ]

# Network training and evaluation

## For training or evaluation of one model

There are three different modes for the network part.`--mode` parameter defines the behaviour of `main_net.py`. Select between `train`, `test` and `eval`. \
`train`: training a network model with training and validation dataset. \
`test`: running an existing network model on testing dataset to obtain concatenated trajectories and metrics. \
`eval`: running an exising network model and save all statistics of data samples for network performance evaluation.

### 1. Training:

**Parameters:** 

`--root_dir`: dataset root directory. Each subfolder of root directory is a dataset. \
`--train_list`: directory of the txt file with a list of training datasets. It should contain name of subfolder in root. \
`--val_list`: directory of the txt file with a list of validation datasets.  \It should contain name of subfolder in root.\
`--out_dir`: training output directory, where `checkpoints` and `logs` folders will be created to store trained models and tensorboard logs respectively. A `parameters.json` file will also be saved.

**Example:** 
```shell script
python3 src/main_net.py \
--mode train \
--root_dir data/Dataset \
--train_list data/Dataset/train.txt \
--val_list data/Dataset/val.txt \
--out_dir train_outputs
```

### 2. Testing:

**Parameters:** 

`--test_list`: path of the txt file with a list of testing datasets. \
`--model_path`: path of the trained model to test with. \
`--out_dir`: testing output directory, where a folder for each dataset tested will be created containing estimated trajectory as `trajectory.txt` and plots if specified. `metrics.json` contains the statistics for each dataset. 

**Example:**
```
python3 src/main_net.py \
--mode test \
--root_dir data/Dataset \
--test_list data/Dataset/test.txt \
--model_path models/resnet/checkpoint_*.pt \
--out_dir test_outputs
```

### 3. Evaluation:

**Parameters:** 

`--out_dir`: evaluation pickle file output directory. \
`--sample_freq`: the frequency of network input data sample tested in Hz. \
`--out_name`: (optional) output pickle file name.

**Example:**
```shell script
python3 src/main_net.py \
--mode eval \
--root_dir data/Dataset \
--test_list data/Dataset/test.txt \
--model_path models/resnet/checkpoint_*.pt \
--out_dir eval_outputs \
--sample_freq 5 \
--out_name resnet.pkl
```
Please refer to `main_net.py` for a full list of parameters.

## For batch evaluation on multiple models

Batch scripts are under src/batch_analysis module. Execute batch scripts from the src folder.

### Testing:

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
Run batch tests. `--model_globbing` is the globbing pattern to find all models to test.
```shell script
python -m batch_runner.net_test_batch \
--root_dir ../data/Dataset \
--data_list ../data/Dataset/test.txt \
--model_globbing "../models/*/checkpoint_*.pt" \
--out_dir ../batch_test_outputs \
```
To save plots as well, change parameter `save_plot` to True in `main_net.py`.

### Evaluation:

Batch evaluation runs the eval mode for multiple models, with various perturbation settings. Different perturbations result in a separate pickle file under each model folder. Output tree structure:
```
net_eval_outputs
├── model1
│   ├── d-bias-0.0-0.025-grav-0.0.pkl
│   ├── d-bias-0.0-0.05-grav-0.0.pkl
│   ├── d-bias-0.0-0.075-grav-0.0.pkl
│   ├── d-bias-0.0-0.0-grav-0.0.pkl
│   ├── d-bias-0.0-0.0-grav-10.0.pkl
│   ├── d-bias-0.0-0.0-grav-2.0.pkl
│   ├── d-bias-0.0-0.0-grav-4.0.pkl
│   ├── d-bias-0.0-0.0-grav-6.0.pkl
│   ├── d-bias-0.0-0.0-grav-8.0.pkl
│   ├── d-bias-0.0-0.1-grav-0.0.pkl
│   ├── d-bias-0.1-0.0-grav-0.0.pkl
│   ├── d-bias-0.2-0.0-grav-0.0.pkl
│   ├── d-bias-0.3-0.0-grav-0.0.pkl
│   ├── d-bias-0.4-0.0-grav-0.0.pkl
│   └── d-bias-0.5-0.0-grav-0.0.pkl
├── model2
│   ├── d-bias-0.0-0.025-grav-0.0.pkl
│   ├── d-bias-0.0-0.05-grav-0.0.pkl
...
```

In the current script, the following perturbation values are used: \
Accelerometer bias perturbation range: [0, 0.1, 0.2, 0.3, 0.4, 0.5] (m/s^2) \
Gyroscope bias perturbation range: [0, 0.025, 0.05, 0.075, 0.1] (rad/s) \
Gravity direction perturbation range: [0, 0, 2, 4, 6, 8, 10] (degrees) \
These can be changed in the script `batch_runner/net_eval_batch.py`, and for each perturbation range a pkl file will be saved with the range in the filename.

Create an output directory and go to the src folder
```shell script
mkdir batch_eval_outputs
cd src
```
Run batch evaluation
```shell script
python -m batch_runner.net_eval_batch \
--root_dir ../data/Dataset \
--data_list ../data/Dataset/test.txt \
--model_globbing "../models/*/checkpoint_*.pt" \
--out_dir ../net_eval_outputs \
--sample_freq 5.0
```

## Running analysis and generating plots

After running testing and evaluation in batches, the statistics are saved in either `metrics.json` or the generated pickle files. To visualize the results and compare between models, we provide scripts that display the results in an interactive shell through iPython. The scripts are under `src/analysis` module.

To visualize network testing results from `metrics.json` including trajectory metrics and testing losses, go to `src` folder and run
```shell script
python -m analysis.display_json \
--glob_dataset "../batch_test_output/*/"
```
This will leave you in an interactive shell with a preloaded panda DataFrame `d`. You can use it to visualize all metrics with the following helper function:
```shell script
plot_all_stats_net(d)
```

To visualize evaluation results from pickle files, run
```shell script
python -m analysis.display_pickle \
--glob_pickle "../batch_eval_outputs/*/*.pkl"
```
This gives access to all the sample data 3D displacement gt and errors, sigmas, mse and likelihood losses, 2D norm and angle gt and errors, and mahalanobis distance based on the regressed covariance. To plot sigmas vs. errors for example, run
```shell script
plot_sigmas(d)
```

# Running EKF with network displacement estimates

## Running EKF with one network model
 
Use `src/main_filter.py` for running the filter and parsing parameters. The program supports running multiple datasets on one specified network model.

**Parameters:**

`--model_path`: path to saved model checkpoint file. \
`--model_param_path`: path to parameter json file for this model. \
`--out_dir`: filter output directory. This will include a `parameters.json` file with filter parameters, and a folder for each dataset containing the logged states, default to `not_vio_state.txt`. \
`--erase_old_log`: overwrite old log files. If set to `--no-erase_old_log`, the program would skip running on the datasets if the output file already exists in the output directory. \
`--save_as_npy`: convert the output txt file to npy file and append file extension (e.g. `not_vio_state.txt.npy`) to save space. \
`--initialize_with_offline_calib`: initialize with offline calibration of the IMU. If set to `--no-initialize_with_offline_calib` the initial IMU biases will be initialized to 0.

**Example:**
```shell script
python3 src/main_filter.py \
--root_dir data/Dataset \
--data_list data/Dataset/test.txt \
--model_path models/resnet/checkpoint_75.pt \
--model_param_path models/resnet/parameters.json \
--out_dir filter_outputs \
--erase_old_log \
--save_as_npy \
--initialize_with_offline_calib
```
Please refer to `main_filter.py` for a full list of parameters.

## Batch running filter on multiple models and parameters

Batch script `batch_runner/filter_batch` provides functionality to run the main file in batch settings. Go to `src` folder to run the module and you can set the parameters to test within the script (e.g. different update frequencies).

**Example:**
```shell script
python -m batch_runner.run_batch \
--root_dir ../data/Dataset \
--data_list ../data/Dataset/test.txt \
--model_globbing "../models/*/checkpoint_*.pt" \
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
--root_dir ../data/Dataset \
--data_list ../data/Dataset/test.txt \
--runname_globbing "*" \
--filter_dir ../batch_filter_outputs \
--ronin_dir ../batch_test_outputs \
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

To generate plots from the metrics:
```shell script
python -m analysis.display_json \
--glob_dataset "../batch_filter_outputs/*/"
```
