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

device = torch.device("cpu")


def load_data(path):
    # Load test_inputs and test_conditions from files
    test_inputs = torch.load(path + "test_inputs.pt")
    test_conditions = torch.load(path + "test_conditions.pt")
    test_meta_data = torch.load(
        path + "test_meta_data.pt"
    )  # contains columns ['min_mvaID'] + ['ScEta'] + ['mass'] + ["origin"]
    test_weights = torch.load(path + "test_weights.pt")

    return test_inputs, test_conditions, test_meta_data, test_weights


def split_data(test_inputs, test_conditions, test_meta_data, test_weights):
    test_inputs_diphoton = test_inputs[test_meta_data[:, -1] == 1]
    test_conditions_diphoton = test_conditions[test_meta_data[:, -1] == 1]
    test_meta_data_diphoton = test_meta_data[test_meta_data[:, -1] == 1]
    test_weights_diphoton = test_weights[test_meta_data[:, -1] == 1]

    test_inputs_gjet = test_inputs[test_meta_data[:, -1] == 0]
    test_conditions_gjet = test_conditions[test_meta_data[:, -1] == 0]
    test_meta_data_gjet = test_meta_data[test_meta_data[:, -1] == 0]
    test_weights_gjet = test_weights[test_meta_data[:, -1] == 0]

    return (
        test_inputs_diphoton,
        test_conditions_diphoton,
        test_meta_data_diphoton,
        test_weights_diphoton,
    ), (test_inputs_gjet, test_conditions_gjet, test_meta_data_gjet, test_weights_gjet)


def load_model(path):
    # Load the model
    model = torch.load(path + "results/saved_states/best_model_.pth")

    stream = open(path + "flow_config.yaml", "r")
    dictionary = yaml.load(stream, Loader)

    return model, dictionary


def create_flow(test_inputs, test_conditions, dictionary, device):
    for key in dictionary:
        # network configurations
        n_transforms = dictionary[key]["n_transforms"]  # number of transformation
        aux_nodes = dictionary[key][
            "aux_nodes"
        ]  # number of nodes in the auxiliary network
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
        test_inputs.size()[1],
        context=test_conditions.size()[1],
        bins=n_splines_bins,
        transforms=n_transforms,
        hidden_features=[aux_nodes] * aux_layers,
    )
    flow.to(device)

    flow.load_state_dict(
        torch.load(
            path + "results/saved_states/best_model_.pth",
            map_location=torch.device("cpu"),
        )
    )

    return flow


def apply_flow_and_invert_standardization(test_inputs, test_conditions, flow, path):
    samples = utlis.apply_flow(test_inputs, test_conditions, flow)
    samples = utlis.invert_standardization(samples, path)

    return samples


def load_parquet_files(path_df):
    df_data = pd.read_parquet(path_df + "Data_postEE.parquet")
    df_Diphoton = pd.read_parquet(path_df + "Diphoton_postEE.parquet")
    df_GJEt = pd.read_parquet(path_df + "GJEt_postEE.parquet")

    return df_data, df_Diphoton, df_GJEt


# Set the path
path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"
path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/Diphoton_samples/"

# Load the data
test_inputs, test_conditions, test_meta_data, test_weights = load_data(path)

# Split the data
(diphoton_inputs, diphoton_conditions, diphoton_meta_data, diphoton_weights), (
    gjet_inputs,
    gjet_conditions,
    gjet_meta_data,
    gjet_weights,
) = split_data(test_inputs, test_conditions, test_meta_data, test_weights)

# Load the model and dictionary
model, dictionary = load_model(path)

# Create the flow
flow = create_flow(test_inputs, test_conditions, dictionary, device)

# Apply the flow and invert standardization
samples_diphoton = apply_flow_and_invert_standardization(
    diphoton_inputs, diphoton_conditions, flow, path
)
samples_gjet = apply_flow_and_invert_standardization(
    gjet_inputs, gjet_conditions, flow, path
)

# Load parquet files
df_data, df_Diphoton, df_GJEt = load_parquet_files(path_df)

var_list = [
    "r9",
    "etaWidth",
    "phiWidth",
    "s4",
]

conditions_list = ["pt", "ScEta", "phi", "fixedGridRhoAll", "mass", "is_data"]
meta_data_list = ["min_mvaID", "mass", "origin"]


def df_with_corr_mvaid(
    samples,
    conditions,
    meta_data,
    weights,
    var_list,
    conditions_list,
    meta_data_list,
    process,
):
    # Convert the samples, conditions, and meta data to pandas DataFrame
    df_samples = pd.DataFrame(samples, columns=var_list)
    df_conditions = pd.DataFrame(conditions, columns=conditions_list)
    df_meta_data = pd.DataFrame(meta_data, columns=meta_data_list)
    df_weights = pd.DataFrame(weights, columns=["weights"])

    # Concatenate the samples, conditions, and meta data DataFrames
    df_samples_corr = pd.concat(
        [df_samples, df_conditions, df_meta_data, df_weights], axis=1
    )

    # Add corrected photon id mva
    corr_mvaID = utlis.add_corr_photonid_mva_run3_zmmg(df_samples_corr, process)
    df_samples_corr["photon_corr_mvaID_run3"] = corr_mvaID

    utlis.comparison_mvaID(df_samples_corr, title=process)

    return df_samples_corr


# Use the function for diphoton and gjet
df_samples_diphoton = df_with_corr_mvaid(
    samples_diphoton,
    diphoton_conditions,
    diphoton_meta_data,
    diphoton_weights,
    var_list,
    conditions_list,
    meta_data_list,
    "Diphoton",
)
df_samples_gjet = df_with_corr_mvaid(
    samples_gjet,
    gjet_conditions,
    gjet_meta_data,
    gjet_weights,
    var_list,
    conditions_list,
    meta_data_list,
    "GJet",
)

plot_list = ["lead_" + s for s in var_list]
# %%


def analysis_generel(
    df_Diphoton,
    df_GJEt,
    df_data,
    samples_diphoton,
    samples_gjet,
    test_weights_diphoton,
    test_weights_gjet,
):
    i = 0
    for var in plot_list:

        utlis.plot_hist_subplots(
            df_Diphoton,
            df_GJEt,
            df_data,
            samples_diphoton,
            samples_gjet,
            var,
            test_weights_diphoton,
            test_weights_gjet,
            i,
            "No Mask",
        )
        i += 1

    i = 0
    for var in plot_list:

        utlis.plot_hist(
            df_Diphoton,
            df_GJEt,
            df_data,
            samples_diphoton,
            samples_gjet,
            var,
            test_weights_diphoton,
            test_weights_gjet,
            i,
            "No Mask",
        )
        i += 1


analysis_generel(
    df_Diphoton,
    df_GJEt,
    df_data,
    samples_diphoton,
    samples_gjet,
    diphoton_weights,
    gjet_weights,
)


# %%
# analysis with m between 150-180 and min mvid > 0.25
def analysis_m_cut(
    df_Diphoton,
    df_GJEt,
    df_data,
    test_meta_data_diphoton,
    test_meta_data_gjet,
    test_inputs_diphoton,
    test_conditions_diphoton,
    test_weights_diphoton,
    test_inputs_gjet,
    test_conditions_gjet,
    test_weights_gjet,
    m_low,
    m_high,
    mva_ID_cut,
):

    mask_diphoton = (
        (test_meta_data_diphoton[:, 0] > mva_ID_cut)
        & (test_meta_data_diphoton[:, 1] > m_low)
        & (test_meta_data_diphoton[:, 1] < m_high)
    )
    mask_gjet = (
        (test_meta_data_gjet[:, 0] > mva_ID_cut)
        & (test_meta_data_gjet[:, 1] > m_low)
        & (test_meta_data_gjet[:, 1] < m_high)
    )

    def apply_mask(df):
        mask = (
            (df["min_mvaID"] > mva_ID_cut)
            & (df["mass"] > m_low)
            & (df["mass"] < m_high)
        )
        return df[mask]

    df_Diphoton = apply_mask(df_Diphoton)
    df_GJEt = apply_mask(df_GJEt)
    df_data = apply_mask(df_data)

    test_inputs_diphoton = test_inputs_diphoton[mask_diphoton]
    test_conditions_diphoton = test_conditions_diphoton[mask_diphoton]
    test_weights_diphoton = test_weights_diphoton[mask_diphoton]

    test_inputs_gjet = test_inputs_gjet[mask_gjet]
    test_conditions_gjet = test_conditions_gjet[mask_gjet]
    test_weights_gjet = test_weights_gjet[mask_gjet]

    samples_diphoton = utlis.apply_flow(
        test_inputs_diphoton, test_conditions_diphoton, flow
    )
    samples_gjet = utlis.apply_flow(test_inputs_gjet, test_conditions_gjet, flow)

    samples_diphoton = utlis.invert_standardization(samples_diphoton, path)
    samples_gjet = utlis.invert_standardization(samples_gjet, path)

    i = 0
    for var in plot_list:

        utlis.plot_hist_subplots(
            df_Diphoton,
            df_GJEt,
            df_data,
            samples_diphoton,
            samples_gjet,
            var,
            test_weights_diphoton,
            test_weights_gjet,
            i,
            "Mask m and mvaID",
        )
        i += 1

    i = 0
    for var in plot_list:

        utlis.plot_hist(
            df_Diphoton,
            df_GJEt,
            df_data,
            samples_diphoton,
            samples_gjet,
            var,
            test_weights_diphoton,
            test_weights_gjet,
            i,
            "Mask m and mvaID",
        )
        i += 1


analysis_m_cut(
    df_Diphoton,
    df_GJEt,
    df_data,
    diphoton_meta_data,
    gjet_meta_data,
    diphoton_inputs,
    diphoton_conditions,
    diphoton_weights,
    gjet_inputs,
    gjet_conditions,
    gjet_weights,
    m_low=150,
    m_high=180,
    mva_ID_cut=0.25,
)
