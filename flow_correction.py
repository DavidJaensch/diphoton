import glob
import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.utils.data as t_data
import utils as utlis
import yaml
import zuko
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from yaml import Loader

# Set the mplhep style to CMS for plots
hep.style.use("CMS")

path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"
device = torch.device("cpu")

# Load test_inputs and test_conditions from files
test_inputs = torch.load(path + "test_inputs.pt")
test_conditions = torch.load(path + "test_conditions.pt")
test_weights = torch.load(path + "test_weights.pt")

test_inputs_diphoton = test_inputs[test_inputs[:, -1] == 1]
test_conditions_diphoton = test_conditions[test_inputs[:, -1] == 1]
test_weights_diphoton = test_weights[test_inputs[:, -1] == 1]

test_inputs_gjet = test_inputs[test_inputs[:, -1] == 0]
test_conditions_gjet = test_conditions[test_inputs[:, -1] == 0]
test_weights_gjet = test_weights[test_inputs[:, -1] == 0]


# Load the model
model = torch.load(path + "results/saved_states/best_model_.pth")

stream = open(
    "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/flow_config.yaml", "r"
)
dictionary = yaml.load(stream, Loader)

for key in dictionary:

    # network configurations
    n_transforms = dictionary[key]["n_transforms"]  # number of transformation
    aux_nodes = dictionary[key]["aux_nodes"]  # number of nodes in the auxiliary network
    aux_layers = dictionary[key][
        "aux_layers"
    ]  # number of auxiliary layers in each flow transformation
    n_splines_bins = dictionary[key][
        "n_splines_bins"
    ]  # Number of rationale quadratic spline flows bins

    # Some general training parameters
    max_epoch_number = dictionary[key]["max_epochs"]
    initial_lr = dictionary[key]["initial_lr"]
    batch_size = dictionary[key]["batch_size"]

flow = zuko.flows.NSF(
    4,
    context=5,
    bins=n_splines_bins,
    transforms=n_transforms,
    hidden_features=[aux_nodes] * aux_layers,
)
flow.to(device)

flow.load_state_dict(
    torch.load(
        path + "results/saved_states/best_model_.pth", map_location=torch.device("cpu")
    )
)

samples_diphoton = utlis.apply_flow(
    test_inputs_diphoton[:, :-1], test_conditions_diphoton, flow
)
samples_gjet = utlis.apply_flow(test_inputs_gjet[:, :-1], test_conditions_gjet, flow)

samples_diphoton = utlis.invert_standardization(samples_diphoton, path)
samples_gjet = utlis.invert_standardization(samples_gjet, path)


path = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/Diphoton_samples/"
df_data = pd.read_parquet(path + "Data_postEE.parquet")
df_Diphoton = pd.read_parquet(path + "Diphoton_postEE.parquet")
df_GJEt = pd.read_parquet(path + "GJEt_postEE.parquet")

# %% plot histograms
var_list = [
    "r9",
    "etaWidth",
    "phiWidth",
    "s4",
]

plot_list = ["lead_" + s for s in var_list]


def plot_hist_subplots(
    df_diphoton,
    df_g_jet,
    df_data,
    samples_diphoton,
    samples_gjet,
    var,
    test_weights_diphoton,
    test_weights_gjet,
    i,
):
    plt.clf()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.style.use(hep.style.CMS)

    # Calculate the mean and standard deviation of the data
    mean = np.mean(np.concatenate([df_diphoton[var], df_g_jet[var], df_data[var]]))
    std = np.std(np.concatenate([df_diphoton[var], df_g_jet[var], df_data[var]]))

    num_bins = 100
    min_value = mean - 2 * std
    max_value = mean + 4 * std
    bin_width = (max_value - min_value) / num_bins

    # Create the diphoton and g_jet histograms and stack them
    hist_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df_diphoton[var], weight=df_diphoton["weight"])
    )

    hist_g_jet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df_g_jet[var], weight=df_g_jet["weight"])
    )
    sum_data = hist_diphoton.values().sum() + hist_g_jet.values().sum()
    hist_diphoton = hist_diphoton / (sum_data * bin_width)
    hist_g_jet = hist_g_jet / (sum_data * bin_width)
    hep.histplot(
        [hist_diphoton, hist_g_jet],
        stack=True,
        histtype="fill",
        label=["Diphoton", "G Jet"],
        ax=axs[0],
    )

    # Create the data histogram and plot it with only the top marks
    # Create the data histogram and plot it with only the top marks
    hist_data = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df_data[var], weight=df_data["weight"])
    )
    hist_data = hist_data / (hist_data.values().sum() * bin_width)
    print(hist_data.values().sum())

    bin_edges = hist_data.axes[0].edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs[0].errorbar(
        bin_centers, hist_data.values(), fmt="x", color="black", label="Data"
    )
    axs[0].set_xlabel(var)
    axs[0].set_ylabel("Frequency")
    axs[0].legend()
    hep.cms.label("Work in Progress", data=False, ax=axs[0])

    # Create the diphoton and g_jet sample histograms and stack them
    hist_samples_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            samples_diphoton[:, i].numpy(),
            weight=test_weights_diphoton.numpy(),
        )  # Flatten the array
    )
    hist_samples_gjet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            samples_gjet[:, i].numpy(), weight=test_weights_gjet.numpy()
        )  # Flatten the array
    )

    # Normalize the histograms
    sum_sim = hist_samples_diphoton.values().sum() + hist_samples_gjet.values().sum()
    hist_samples_diphoton = hist_samples_diphoton / (sum_sim * bin_width)
    hist_samples_gjet = hist_samples_gjet / (sum_sim * bin_width)

    hep.histplot(
        [hist_samples_diphoton, hist_samples_gjet],
        stack=True,
        histtype="fill",
        label=["Diphoton samples", "GJet samples"],
        ax=axs[1],
    )

    # Create the data histogram and plot it with only the top marks
    hist_data = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df_data[var], weight=df_data["weight"])
    )

    # Normalize the data histogram
    hist_data = hist_data / (hist_data.values().sum() * bin_width)

    bin_edges = hist_data.axes[0].edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs[1].errorbar(
        bin_centers, hist_data.values(), fmt="x", color="black", label="Data"
    )

    axs[1].set_xlabel(var)
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/sample_compare_hist_{var}.png"
    )
    plt.show()


i = 0
for var in plot_list:

    plot_hist_subplots(
        df_Diphoton,
        df_GJEt,
        df_data,
        samples_diphoton,
        samples_gjet,
        var,
        test_weights_diphoton,
        test_weights_gjet,
        i,
    )
    i += 1


# %%
def plot_hist(
    df_diphoton,
    df_g_jet,
    df_data,
    samples_diphoton,
    samples_gjet,
    var,
    test_weights_diphoton,
    test_weights_gjet,
    i,
):
    plt.clf()
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.style.use(hep.style.CMS)

    # Calculate the mean and standard deviation of the data
    mean = np.mean(np.concatenate([df_diphoton[var], df_g_jet[var], df_data[var]]))
    std = np.std(np.concatenate([df_diphoton[var], df_g_jet[var], df_data[var]]))

    num_bins = 100
    min_value = mean - 2 * std
    max_value = mean + 4 * std

    bin_width = (max_value - min_value) / num_bins

    # Create the diphoton and g_jet histograms and stack them
    hist_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df_diphoton[var], weight=df_diphoton["weight"])
    )

    hist_g_jet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df_g_jet[var], weight=df_g_jet["weight"])
    )

    hist_samples_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            samples_diphoton[:, i].numpy(), weight=test_weights_diphoton.numpy()
        )  # Flatten the array
    )
    hist_samples_gjet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            samples_gjet[:, i].numpy(), weight=test_weights_gjet.numpy()
        )  # Flatten the array
    )

    hist_samples_diphoton = hist_samples_diphoton / (
        hist_samples_diphoton.values().sum() * bin_width
    )
    hist_samples_gjet = hist_samples_gjet / (
        hist_samples_gjet.values().sum() * bin_width
    )
    hist_g_jet = hist_g_jet / (hist_g_jet.values().sum() * bin_width)
    hist_diphoton = hist_diphoton / (hist_diphoton.values().sum() * bin_width)

    hep.histplot(
        [hist_diphoton, hist_samples_diphoton],
        stack=False,
        histtype="step",
        label=["MC_Diphoton", "MC_corrected"],
        ax=axs[0],
    )

    axs[0].set_xlabel(var)
    axs[0].set_ylabel("Frequency")
    axs[0].legend()
    hep.cms.label("Work in Progress", data=False, ax=axs[0])

    hep.histplot(
        [hist_g_jet, hist_samples_gjet],
        stack=False,
        histtype="step",
        label=["MC_GJet", "MC_corrected"],
        ax=axs[1],
    )

    axs[1].set_xlabel(var)
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/direct_compare_hist_{var}.png"
    )
    plt.show()


i = 0
for var in plot_list:

    plot_hist(
        df_Diphoton,
        df_GJEt,
        df_data,
        samples_diphoton,
        samples_gjet,
        var,
        test_weights_diphoton,
        test_weights_gjet,
        i,
    )
    i += 1

# %%
