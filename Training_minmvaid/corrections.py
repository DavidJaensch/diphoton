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
    test_weights = torch.load(path + "test_weights.pt")

    return test_inputs, test_conditions, test_weights


def split_data(test_inputs, test_conditions, all_info_test, test_weights):
    test_inputs_diphoton = test_inputs[all_info_test["origin"] == 1]
    test_conditions_diphoton = test_conditions[all_info_test["origin"] == 1]
    all_info_test_diphoton = all_info_test[all_info_test["origin"] == 1]
    test_weights_diphoton = test_weights[all_info_test["origin"] == 1]

    test_inputs_gjet = test_inputs[all_info_test["origin"] == 0]
    test_conditions_gjet = test_conditions[all_info_test["origin"] == 0]
    all_info_test_gjet = all_info_test[all_info_test["origin"] == 0]
    test_weights_gjet = test_weights[all_info_test["origin"] == 0]

    return (
        test_inputs_diphoton,
        test_conditions_diphoton,
        all_info_test_diphoton,
        test_weights_diphoton,
    ), (test_inputs_gjet, test_conditions_gjet, all_info_test_gjet, test_weights_gjet)


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
path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/Training_minmvaid/"
path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/Diphoton_samples/"

all_info_test = pd.read_csv(path + "all_info_test.csv")
# Load the data
test_inputs, test_conditions, test_weights = load_data(path)

# Split the data
(diphoton_inputs, diphoton_conditions, diphoton_all_info, diphoton_weights), (
    gjet_inputs,
    gjet_conditions,
    gjet_all_info,
    gjet_weights,
) = split_data(test_inputs, test_conditions, all_info_test, test_weights)

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
    "sieie",
    "sieip",
]

conditions_list = ["pt", "ScEta", "phi", "fixedGridRhoAll", "mass"]

# Define the variables you want to select
other_vars = [
    "energyRaw",
    "hoe",
    "ecalPFClusterIso",
    "hcalPFClusterIso",
    "trkSumPtHollowConeDR03",
    "trkSumPtSolidConeDR04",
    "pfChargedIso",
    "pfChargedIsoWorstVtx",
    "esEffSigmaRR",
    "esEnergyOverRawE",
]
variables = var_list + conditions_list + other_vars

variables_no_prefix = ["mass", "fixedGridRhoAll"]

# For each variable, add a new column to the DataFrame that contains the value of the variable with the minimum mvaID
for var in variables:
    if var in variables_no_prefix:
        df_data[var + "_minmvaID"] = df_data[var]
    else:
        df_data[var + "_minmvaID"] = np.where(
            df_data["lead_mvaID"] < df_data["sublead_mvaID"],
            df_data["lead_" + var],
            df_data["sublead_" + var],
        )

var_list = [
    "r9",
    "etaWidth",
    "phiWidth",
    "s4",
    "sieie",
    "sieip",
]
var_list = [s + "_minmvaID_corr_of" for s in var_list]


df_samples_diphoton = pd.DataFrame(samples_diphoton, columns=var_list)
df_samples_gjet = pd.DataFrame(samples_gjet, columns=var_list)


diphoton_all_info_reset = diphoton_all_info.reset_index(drop=True)
gjet_all_info_reset = gjet_all_info.reset_index(drop=True)
df_samples_diphoton_reset = df_samples_diphoton.reset_index(drop=True)
df_samples_gjet_reset = df_samples_gjet.reset_index(drop=True)
diphoton_all_info = pd.concat(
    [diphoton_all_info_reset, df_samples_diphoton_reset], axis=1
)
gjet_all_info = pd.concat([gjet_all_info_reset, df_samples_gjet_reset], axis=1)


def add_corr_mvaid(df_samples_corr, title, data_name, corr=True):

    # Add corrected photon id mva
    if corr == True:
        corr_mvaID = utlis.add_corr_photonid_mva_run3_of(df_samples_corr)
        df_samples_corr["photon_corr_mvaID_run3"] = corr_mvaID
    else:
        corr_mvaID = utlis.add_corr_photonid_mva_run3_data(df_samples_corr)
        df_samples_corr["photon_corr_mvaID_run3"] = corr_mvaID

    utlis.comparison_mvaID(df_samples_corr, title=title, data_name=data_name)

    return df_samples_corr


add_corr_mvaid(diphoton_all_info, title="Diphoton", data_name="di_sap")
add_corr_mvaid(gjet_all_info, title="GJet", data_name="gjet_sap")
add_corr_mvaid(df_data, title="Data", data_name="data", corr=False)


utlis.plot_hist_subplots(
    diphoton_all_info,
    gjet_all_info,
    df_data,
    var="photon_corr_mvaID_run3",
    title="No Mask",
    var_uncorr="min_mvaID",
)

# %%


def analysis_generel(
    diphoton_all_info,
    gjet_all_info,
    df_data,
):
    for var in var_list:

        utlis.plot_hist_subplots(
            diphoton_all_info,
            gjet_all_info,
            df_data,
            var,
            "No Mask",
        )

    for var in var_list:

        utlis.plot_hist(
            diphoton_all_info,
            gjet_all_info,
            df_data,
            var,
            "No Mask",
        )


analysis_generel(
    diphoton_all_info,
    gjet_all_info,
    df_data,
)


# %%
