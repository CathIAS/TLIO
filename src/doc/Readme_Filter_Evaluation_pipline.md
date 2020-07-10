**WARNING** All the command should be run from `source` subfolder

## Run with one model/configuration of filter, on the entire test set
    
    python3 main_filter.py \
    --data_list ~/vo_output/split/golden/golden_test.txt \
    --root_dir ~/vo_output/ \
    --model_path ../models/200hz/1s-1s/checkpoint_latest.pt  \
    --out_dir ./output_tmp \
    --initialize_with_offline \
    --erase_old_log

## Run with multiple models/same filter configuration on the entire test set

    python -m batch_runner.run_batch

This assumes models are in `../models/XXXhz/` (modify the globbing expression if necessary).

This load models information from the `conf.yaml` in `../models/XXXhz/*/` (past data, imu freq).

This uses the checkpoint file in `../models/XXXhz/*/` assuming it is unique.

The output hierarchy will be by default

    all_output
    ├── 200hz-1s-1s
    │   ├── loop_hidacori0511_20180705_0948
    │   ├── [...]
    │   └── parameters.json
    ├── 200hz-1s-1s_no-ptrb_b1
    │   ├── [...]

## Run batch_plotting pipeline on `all_output` folder

    python -m batch_runner.plot_batch
It adds images of plot and metrics.json in the all_output folder
    
    all_output
    ├── 200hz-1s-1s
    │   ├── loop_hidacori0511_20180705_0948
    │   │   ├── acc-bias-err.png
    │   │   ├── acc-bias.png
    │   │   ├── displacement-err.png
    │   │   ├── displacement.png
    │   │   ├── gyr-bias-err.png
    │   │   ├── gyr-bias.png
    │   │   ├── innovation.png
    │   │   ├── not_vio_state.txtdebug.txt
    │   │   ├── not_vio_state.txt.npy
    │   │   ├── position-2d.png
    │   │   ├── position-3d.png
    │   │   ├── position-err.png
    │   │   ├── position.png
    │   │   ├── rotation-err.png
    │   │   ├── rotation.png
    │   │   ├── velocity-err.png
    │   │   ├── velocity.png
    │   │   └── vio_states.npy
    │   ├── [...]
        ├── metrics.json
        
  **WARNING** : it does not currently takes the ronin output for comparison. If you want to also include RONIN metric
  you should run main_test to get ronin csv output and modify the script plot_batch.py to give the 
   `--ronin_dir ` argument set to the value where ronin log were written.
   
## display aggregated metrics per mode/configuration

    python -m analysis.display_metrics_json
 It drops you to a Ipython shell after having loaded all the metrics.json 
 for all methods and put them in a Panda Dataframe.
 
 The main command for display is
 
     plot_all_stats(d) 
     plt.show()
 But you can play with the dataframe d that contains all the data.
 