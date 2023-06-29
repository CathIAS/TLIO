import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
#from dataloader.dataset_fb import FbSequenceDataset
from dataloader.tlio_data import TlioData
from dataloader.memmapped_sequences_dataset import MemMappedSequencesDataset
from network.losses import get_loss
from network.model_factory import get_model
from torch.utils.data import DataLoader
from utils.dotdict import dotdict
from utils.utils import to_device
from utils.logging import logging
from network.test import torch_to_numpy, get_inference, get_datalist, arg_conversion, get_inference

def net_eval(args):
    """
    Main function for network evaluation
    Generate pickle file containing all network sample results
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            logging.info(f"Testing output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {args.model_path} loaded to device {device}.")

    all_targets, all_errors, all_sigmas = [], [], []
    all_norm_targets, all_angle_targets, all_norm_errors, all_angle_errors = (
        [],
        [],
        [],
        [],
    )
    mse_losses, likelihood_losses, avg_mse_losses, avg_likelihood_losses = (
        [],
        [],
        [],
        [],
    )
    all_mahalanobis = []

    test_list = get_datalist(os.path.join(args.root_dir, "test_list.txt"))

    blacklist = ["loop_hidacori058_20180519_1525"]
    # blacklist = []
    for data in test_list:
        if data in blacklist:
            logging.info(f"skipping blacklist {data}")
            continue
        logging.info(f"Processing {data}...")

        try:
            #seq_dataset = FbSequenceDataset(
            #    args.root_dir, [data], args, data_window_config, mode="eval"
            #)

            seq_dataset = MemMappedSequencesDataset(
                args.root_dir,
                "test",
                data_window_config,
                sequence_subset=[data],
                store_in_ram=True,
            )

            seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
        except OSError as e:
            logging.error(e)
            continue

        attr_dict = get_inference(network, seq_loader, device, epoch=50)
        norm_targets = np.linalg.norm(attr_dict["targets"][:, :2], axis=1)
        angle_targets = np.arctan2(
            attr_dict["targets"][:, 0], attr_dict["targets"][:, 1]
        )
        norm_preds = np.linalg.norm(attr_dict["preds"][:, :2], axis=1)
        angle_preds = np.arctan2(attr_dict["preds"][:, 0], attr_dict["preds"][:, 1])
        norm_errors = norm_preds - norm_targets
        angle_errors = angle_preds - angle_targets
        sigmas = np.exp(attr_dict["preds_cov"])
        errors = attr_dict["preds"] - attr_dict["targets"]
        a1 = np.expand_dims(errors, axis=1)
        a2 = np.expand_dims(np.multiply(np.reciprocal(sigmas), errors), axis=-1)
        mahalanobis_dists = np.einsum("tip,tpi->t", a1, a2)

        all_targets.append(attr_dict["targets"])
        all_errors.append(errors)
        all_sigmas.append(sigmas)
        mse_losses.append(errors ** 2)
        avg_mse_losses.append(np.mean(errors ** 2, axis=1).reshape(-1, 1))
        likelihood_losses.append(attr_dict["losses"])
        avg_likelihood_losses.append(
            np.mean(attr_dict["losses"], axis=1).reshape(-1, 1)
        )
        all_norm_targets.append(norm_targets.reshape(-1, 1))
        all_angle_targets.append(angle_targets.reshape(-1, 1))
        all_norm_errors.append(norm_errors.reshape(-1, 1))
        all_angle_errors.append(angle_errors.reshape(-1, 1))
        all_mahalanobis.append(mahalanobis_dists.reshape(-1, 1))

    arr_targets = np.concatenate(all_targets, axis=0)
    arr_errors = np.concatenate(all_errors, axis=0)
    arr_sigmas = np.concatenate(all_sigmas, axis=0)
    arr_mse_losses = np.concatenate(mse_losses, axis=0)
    arr_avg_mse_losses = np.concatenate(avg_mse_losses, axis=0)
    arr_likelihood_losses = np.concatenate(likelihood_losses, axis=0)
    arr_avg_likelihood_losses = np.concatenate(avg_likelihood_losses, axis=0)
    arr_norm_targets = np.concatenate(all_norm_targets, axis=0)
    arr_norm_errors = np.concatenate(all_norm_errors, axis=0)
    arr_angle_targets = np.concatenate(all_angle_targets, axis=0)
    arr_angle_errors = np.concatenate(all_angle_errors, axis=0)
    arr_mahalanobis = np.concatenate(all_mahalanobis, axis=0)

    arr_data = np.concatenate(
        (
            arr_targets,
            arr_errors,
            arr_sigmas,
            arr_mse_losses,
            arr_avg_mse_losses,
            arr_likelihood_losses,
            arr_avg_likelihood_losses,
            arr_norm_targets,
            arr_norm_errors,
            arr_angle_targets,
            arr_angle_errors,
            arr_mahalanobis,
        ),
        axis=1,
    )

    dataset = pd.DataFrame(
        arr_data,
        index=range(arr_mahalanobis.shape[0]),
        columns=[
            "targets_x",
            "targets_y",
            "targets_z",
            "errors_x",
            "errors_y",
            "errors_z",
            "sigmas_x",
            "sigmas_y",
            "sigmas_z",
            "mse_losses_x",
            "mse_losses_y",
            "mse_losses_z",
            "avg_mse_losses",
            "likelihood_losses_x",
            "likelihood_losses_y",
            "likelihood_losses_z",
            "avg_likelihood_losses",
            "norm_targets",
            "norm_errors",
            "angle_targets",
            "angle_errors",
            "mahalanobis",
        ],
    )

    dstr = "d"
    if args.do_bias_shift:
        dstr = f"{dstr}-bias-{args.accel_bias_range}-{args.gyro_bias_range}"
    if args.perturb_gravity:
        dstr = f"{dstr}-grav-{args.perturb_gravity_theta_range}"
    dstr = f"{dstr}.pkl"
    if args.out_name is not None:
        dstr = args.out_name
    outfile = os.path.join(args.out_dir, dstr)
    dataset.to_pickle(outfile)
    logging.info(f"Data saved to {outfile}")

    return
