import glob
import math
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
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from yaml import Loader

# Set the mplhep style to CMS for plots
hep.style.use("CMS")

device = torch.device("cpu")

mva_id_mask = True
scaled = True
stacked = True

path_base = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"


# %%
def load_model(path):
    # Load the model
    model = (
        torch.load(path + "results/saved_states/best_model_mva_id_mask.pth")
        if mva_id_mask
        else torch.load(path + "results/saved_states/best_model_.pth")
    )

    stream = open(path_base + "flow_config.yaml", "r")
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
            path + "results/saved_states/best_model_mva_id_mask.pth",
            map_location=torch.device("cpu"),
        )
        if mva_id_mask
        else torch.load(
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


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


path_df = "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/"
if scaled == False:
    path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/unscaled/"
    path_save = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/Zmmg_lead_train/unscaled/"

elif scaled == True and stacked == True:
    path = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/scaled/"
    path_save = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/Zmmg_lead_train/scaled/"


elif scaled == True and stacked == False:
    path = (
        "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/scaled_only_diphoton/"
    )
    path_save = "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/Zmmg_lead_train/scaled_only_diphoton/"

create_directory(path_save)

zmmg = pd.read_parquet(path_df + "Zmmg/DY_postEE_merged.parquet")
df_data = pd.read_parquet(path_df + "Zmmg/DY_postEE_v13")

zmmg["mass"] = zmmg["mmy_mass"]
zmmg["weight"] = zmmg["weight_central"] * zmmg["genWeight"]
df_data["mass"] = df_data["mmy_mass"]
df_data["weight"] = df_data["weight_central"] * df_data["genWeight"]


def create_mask(df):
    return (
        (df["photon_pt"] > 20)
        & (df["photon_eta"].abs() < 2.5)
        & (df["photon_muon_near_dR"].between(0.15, 0.8))
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
# Convert tensor to numpy array
zmmg_inputs_np = zmmg_inputs.numpy()

# Convert numpy array to DataFrame
zmmg_inputs_df = pd.DataFrame(zmmg_inputs_np)

# Plot histograms
zmmg_inputs_df.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()

# Load the model and dictionary
model, dictionary = load_model(path)

# Create the flow
flow = create_flow(zmmg_inputs, zmmg_conditions, dictionary, device)

utlis.basespace(zmmg_inputs, zmmg_conditions, flow, path_save)

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
var_list = ["photon_" + s + "_lead_corr_of" for s in var_list]


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

    utlis.comparison_mvaID(
        df_samples_corr,
        df_data,
        title=title,
        data_name=data_name,
        path_save=path_save,
        subplot=True,
    )

    return df_samples_corr


add_corr_mvaid(zmmg_all_info, title="Zmmg", data_name="zmmg_sap")

# %%


def calculate_uncertainty_ratio(A, B, sigma_A, sigma_B):
    """
    Calculate the ratio A/B and its uncertainty.
    """
    ratio = A / B
    sigma_ratio = np.abs(ratio) * np.sqrt((sigma_A / A) ** 2 + (sigma_B / B) ** 2)
    return ratio, sigma_ratio


def mvaID_unc(data_df, simulation_df):

    # Calculate totals and their uncertainties
    total_simulation_mask = simulation_df["photon_corr_mvaID_run3"] > -0.95
    total_data_mask = data_df["photon_mvaID"] > -0.95
    total_simulation_mvaID_zee_mask = simulation_df["photon_mvaID"] > -0.95

    total_simulation = np.sum(
        simulation_df["weight_central"][total_simulation_mask]
        * simulation_df["genWeight"][total_simulation_mask]
    )
    total_simulation_unc = np.sqrt(
        np.sum(
            (
                simulation_df["weight_central"][total_simulation_mask]
                * simulation_df["genWeight"][total_simulation_mask]
            )
            ** 2
        )
    )

    total_data = np.sum(total_data_mask)
    total_data_unc = np.sqrt(total_data)

    total_simulation_mvaID_zee = np.sum(
        simulation_df["weight_central"][total_simulation_mvaID_zee_mask]
        * simulation_df["genWeight"][total_simulation_mvaID_zee_mask]
    )
    total_simulation_mvaID_zee_unc = np.sqrt(
        np.sum(
            (
                simulation_df["weight_central"][total_simulation_mvaID_zee_mask]
                * simulation_df["genWeight"][total_simulation_mvaID_zee_mask]
            )
            ** 2
        )
    )

    simulation_mask = simulation_df["photon_corr_mvaID_run3"] > 0.25
    data_mask = data_df["photon_mvaID"] > 0.25
    simulation_mvaID_zee_mask = simulation_df["photon_mvaID"] > 0.25

    simulation_after_cut = np.sum(
        simulation_df["weight_central"][simulation_mask]
        * simulation_df["genWeight"][simulation_mask]
    )
    simulation_after_cut_unc = np.sqrt(
        np.sum(
            (
                simulation_df["weight_central"][simulation_mask]
                * simulation_df["genWeight"][simulation_mask]
            )
            ** 2
        )
    )

    data_after_cut = np.sum(data_mask)
    data_after_cut_unc = np.sqrt(data_after_cut)

    simulation_mvaID_zee_after_cut = np.sum(
        simulation_df["weight_central"][simulation_mvaID_zee_mask]
        * simulation_df["genWeight"][simulation_mvaID_zee_mask]
    )
    simulation_mvaID_zee_after_cut_unc = np.sqrt(
        np.sum(
            (
                simulation_df["weight_central"][simulation_mvaID_zee_mask]
                * simulation_df["genWeight"][simulation_mvaID_zee_mask]
            )
            ** 2
        )
    )

    # Calculate efficiencies and their uncertainties
    eff_simulation = simulation_after_cut / total_simulation
    eff_simulation_unc = eff_simulation * np.sqrt(
        (total_simulation_unc / total_simulation) ** 2
        + (simulation_after_cut_unc / simulation_after_cut) ** 2
    )

    eff_data = data_after_cut / total_data
    eff_data_unc = eff_data * np.sqrt(
        (total_data_unc / total_data) ** 2 + (data_after_cut_unc / data_after_cut) ** 2
    )

    eff_simulation_mvaID_zee = (
        simulation_mvaID_zee_after_cut / total_simulation_mvaID_zee
    )
    eff_simulation_mvaID_zee_unc = eff_simulation_mvaID_zee * np.sqrt(
        (total_simulation_mvaID_zee_unc / total_simulation_mvaID_zee) ** 2
        + (simulation_mvaID_zee_after_cut_unc / simulation_mvaID_zee_after_cut) ** 2
    )

    # Print results
    print(f"Eff data: {eff_data:.5f} ± {eff_data_unc:.5f}")
    print(f"Eff simulation: {eff_simulation:.5f} ± {eff_simulation_unc:.5f}")
    print(
        f"Eff simulation (photon_mvaID_zee): {eff_simulation_mvaID_zee:.5f} ± {eff_simulation_mvaID_zee_unc:.5f}"
    )

    # Now plotting
    # Plotting
    labels = ["Data", "Simulation"]
    efficiencies = [eff_data, eff_simulation]
    uncertainties = [eff_data_unc, eff_simulation_unc]
    labels.append("Zee corrected")
    efficiencies.append(eff_simulation_mvaID_zee)
    uncertainties.append(eff_simulation_mvaID_zee_unc)

    x = range(len(labels))  # the label locations
    fig, ax = plt.subplots()
    ax.errorbar(
        x,
        efficiencies,
        yerr=uncertainties,
        fmt="o",
        capsize=5,
        capthick=2,
        ecolor="black",
        linestyle="None",
        marker="^",
        markersize=10,
        label="Stat uncertanty",
    )
    plt.legend()
    ax.set_ylabel("Efficiency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('"Efficiency" - MVA ID > 0.25 - Post EE - Zmmg', fontsize=24)
    ax.yaxis.grid(True)

    ax.set_ylim(0.75, 0.85)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(path_save + "Efficiency_MVAID_Zmmg.png")

    # Lets also do it for the nominal for comparison
    nominal_simulation_mask = simulation_df["photon_mvaID_raw"] > 0.25
    nominal_simulation_after_cut = np.sum(
        simulation_df["weight_central"][nominal_simulation_mask]
        * simulation_df["genWeight"][nominal_simulation_mask]
    )
    nominal_simulation_after_cut_unc = np.sqrt(
        np.sum(
            (
                simulation_df["weight_central"][simulation_mask]
                * simulation_df["genWeight"][simulation_mask]
            )
            ** 2
        )
    )

    # Calculate efficiencies and their uncertainties
    nominal_eff_simulation = nominal_simulation_after_cut / total_simulation
    nominal_eff_simulation_unc = nominal_eff_simulation * np.sqrt(
        (total_simulation_unc / total_simulation) ** 2
        + (nominal_simulation_after_cut_unc / nominal_simulation_after_cut) ** 2
    )

    # Now calculating the ratio and its uncertainty
    ratio, ratio_unc = calculate_uncertainty_ratio(
        eff_data, eff_simulation, eff_data_unc, eff_simulation_unc
    )
    ratio_nominal, ratio_unc_nominal = calculate_uncertainty_ratio(
        eff_data, nominal_eff_simulation, eff_data_unc, nominal_eff_simulation_unc
    )
    ratio_mvaID_zee, ratio_unc_mvaID_zee = calculate_uncertainty_ratio(
        eff_data, eff_simulation_mvaID_zee, eff_data_unc, eff_simulation_mvaID_zee_unc
    )

    # Plotting
    fig, ax = plt.subplots()
    # Plot the ratio with uncertainties
    ax.errorbar(
        0,
        ratio,
        yerr=ratio_unc,
        fmt="o",
        capsize=5,
        capthick=2,
        ecolor="black",
        linestyle="None",
        marker="^",
        markersize=10,
        label="FlowEfficiency Ratio",
    )
    ax.errorbar(
        1,
        ratio_nominal,
        yerr=ratio_unc_nominal,
        fmt="o",
        capsize=5,
        capthick=2,
        ecolor="black",
        linestyle="None",
        marker="^",
        markersize=10,
        label="Nominal Efficiency Ratio",
    )

    ax.errorbar(
        2,
        ratio_mvaID_zee,
        yerr=ratio_unc_mvaID_zee,
        fmt="o",
        capsize=5,
        capthick=2,
        ecolor="black",
        linestyle="None",
        marker="^",
        markersize=10,
        label="Zee Efficiency Ratio",
    )

    # Draw a horizontal line at 1
    ax.axhline(y=1, color="r", linestyle="--")

    ax.set_ylabel("Efficiency Ratio (Data/Simulation)")
    ax.set_xticks([])
    ax.legend()
    ax.set_title("Efficiency ratio - MVA ID > 0.25 - Post EE - Zmmg", fontsize=24)

    ax.set_ylim(0.93, 1.05)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(path_save + "Efficiency_ratio_MVAID_Zmmg.png")

    return


mvaID_unc(df_data, zmmg_all_info)


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
            path_save=path_save,
        )

    for var in var_list:

        utlis.plot_hist(
            diphoton_all_info,
            df_data,
            var,
            "No Mask",
            path_save=path_save,
        )


analysis_generel(
    zmmg_all_info,
    df_data,
)


# %%
