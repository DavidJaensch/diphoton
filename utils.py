# python libraries import
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zuko


def standardize(input_tensor, conditions_tensor, path):
    # Now, the standartization -> The arrays are the same ones using during training
    input_mean_for_std = torch.tensor(np.load(path + "input_means.npy"))
    input_std_for_std = torch.tensor(np.load(path + "input_std.npy"))
    condition_mean_for_std = torch.tensor(np.load(path + "conditions_means.npy"))
    condition_std_for_std = torch.tensor(np.load(path + "conditions_std.npy"))
    input_tensor[:, :-1] = (
        input_tensor[:, :-1] - input_mean_for_std
    ) / input_std_for_std
    conditions_tensor[:, :-1] = (
        conditions_tensor[:, :-1] - condition_mean_for_std
    ) / condition_std_for_std
    return (
        input_tensor,
        conditions_tensor,
        input_mean_for_std,
        input_std_for_std,
        condition_mean_for_std,
        condition_std_for_std,
    )


def invert_standardization(input_tensor, path):
    # Load the input mean and input standard deviation from files
    input_mean = torch.tensor(np.load(path + "input_means.npy"))
    input_std = torch.tensor(np.load(path + "input_std.npy"))

    # Invert the standardization for the inputs
    input_tensor = input_tensor * input_std + input_mean

    return input_tensor


def plot_loss_cruve(training, validation, plot_path):

    fig, ax1 = plt.subplots()

    # Plot training loss on the first axis
    color = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color=color)
    ax1.plot(training, color=color, marker="o", label="Training Loss")
    ax1.plot(validation, color="tab:orange", marker="x", label="Validation Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend()

    # Title and show the plot
    plt.title("Training and Validation Loss")

    plt.savefig(plot_path + "loss_plot.png")
    plt.close()


def apply_flow(
    input_tensor: torch.tensor, conditions_tensor: torch.tensor, flow
) -> torch.tensor:
    """
    This function is responsable for applying the normalizing flow to MC samples
    it takes as input
    """

    # making sure flow and input tensors have the same type
    flow = flow.type(input_tensor.dtype)
    conditions_tensor = conditions_tensor.type(input_tensor.dtype)

    # Use cuda if avaliable - maybe this is causing the meory problems?
    device = torch.device("cpu")
    flow = flow.to(device)
    input_tensor = input_tensor.to(device)
    conditions_tensor = conditions_tensor.to(device)

    # Disabling pytorch gradient calculation so operation uses less memory and is faster
    with torch.no_grad():

        # Now the flow transformation is done!
        trans = flow(conditions_tensor).transform
        sim_latent = trans(input_tensor)

        conditions_tensor = torch.tensor(
            np.concatenate(
                [
                    conditions_tensor[:, :-1].cpu(),
                    np.ones_like(conditions_tensor[:, 0].cpu()).reshape(-1, 1),
                ],
                axis=1,
            )
        ).to(device)

        trans2 = flow(conditions_tensor).transform
        samples = trans2.inv(sim_latent)

    return samples
