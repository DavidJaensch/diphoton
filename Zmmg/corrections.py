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


# %%
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

path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/Training_minmvaid/"
path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/"
path_save = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/Zmmg/"

zmmg = pd.read_parquet(path_df + "Zmmg/DY_postEE_merged.parquet")
df_data = pd.read_parquet(path_df + "Zmmg/DY_postEE_v13")

zmmg["mass"] = zmmg["mmy_mass"]
zmmg["weight"] = zmmg["weight_central"] + zmmg["genWeight"]
df_data["mass"] = df_data["mmy_mass"]
df_data["weight"] = df_data["weight_central"] + df_data["genWeight"]


def create_mask(df):
    return (
        (df["photon_pt"] > 20)
        & (df["photon_eta"].abs() < 2.5)
        & (df["photon_muon_near_dR"] < 0.8)
        & (df["dimuon_mass"] > 35)
        & (df["mmy_mass"].between(60, 120))
        & ((df["dimuon_mass"] + df["mmy_mass"]) < 180)
        & (df["muon_far_pt"] > 20)
    )


zmmg_mask = create_mask(zmmg)
df_data_mask = create_mask(df_data)

zmmg = zmmg[zmmg_mask]
df_data = df_data[df_data_mask]
zmmg = zmmg.reset_index(drop=True)
df_data = df_data.reset_index(drop=True)
# %%

zmmg_var_list = ["photon_raw_" + s for s in var_list]
zmmg_conditions_list = ["photon_" + s for s in conditions_list[:-1]]
zmmg_conditions_list.append("mass")
zmmg_inputs = torch.tensor(np.array(zmmg[zmmg_var_list]))
zmmg_conditions = torch.tensor(np.array(zmmg[zmmg_conditions_list]))
zeros = torch.zeros((zmmg_conditions.shape[0], 1))
zmmg_conditions = torch.cat((zmmg_conditions, zeros), dim=1)

(
    zmmg_inputs,
    zmmg_conditions,
    input_mean_for_std,
    input_std_for_std,
    condition_mean_for_std,
    condition_std_for_std,
) = utlis.standardize(
    zmmg_inputs,
    zmmg_conditions,
    path=path,
)

# Load the model and dictionary
model, dictionary = load_model(path)

# Create the flow
flow = create_flow(zmmg_inputs, zmmg_conditions, dictionary, device)

# Apply the flow and invert standardization
corr_zmmg = apply_flow_and_invert_standardization(
    zmmg_inputs, zmmg_conditions, flow, path
)


var_list = [
    "r9",
    "etaWidth",
    "phiWidth",
    "s4",
    "sieie",
    "sieip",
]
var_list = ["photon_" + s + "_minmvaID_corr_of" for s in var_list]


df_zmmg_corr = pd.DataFrame(corr_zmmg, columns=var_list)


zmmg_all_info = pd.concat([zmmg, df_zmmg_corr], axis=1)

# %%
# Add corrected photon id mva


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


add_corr_mvaid(zmmg_all_info, title="Zmmg", data_name="zmmg_sap")


# %%


def analysis_generel(
    diphoton_all_info,
    df_data,
):
    for var in var_list:

        utlis.plot_hist_subplots(
            diphoton_all_info,
            df_data,
            var,
            "No Mask",
        )

    for var in var_list:

        utlis.plot_hist(
            diphoton_all_info,
            df_data,
            var,
            "No Mask",
        )


analysis_generel(
    zmmg_all_info,
    df_data,
)


# %%
