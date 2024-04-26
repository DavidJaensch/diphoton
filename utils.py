# python libraries import
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import awkward
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import torch
import xgboost
import zuko


def standardize(input_tensor, conditions_tensor, path):
    # Now, the standartization -> The arrays are the same ones using during training
    input_mean_for_std = torch.tensor(np.load(path + "input_means.npy"))
    input_std_for_std = torch.tensor(np.load(path + "input_std.npy"))
    condition_mean_for_std = torch.tensor(np.load(path + "conditions_means.npy"))
    condition_std_for_std = torch.tensor(np.load(path + "conditions_std.npy"))
    input_tensor = (input_tensor - input_mean_for_std) / input_std_for_std
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


# Stacked comparison of the histograms with data points
def plot_hist_subplots(
    diphoton_all_info,
    gjet_all_info,
    df_data,
    var,
    title,
    var_uncorr="",
):
    plt.clf()
    if var_uncorr == "":
        var_uncorr = var[:-8]
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    plt.style.use(hep.style.CMS)

    # Calculate the mean and standard deviation of the data
    mean = np.mean(
        np.concatenate(
            [diphoton_all_info[var], gjet_all_info[var], df_data[var_uncorr]]
        )
    )
    std = np.std(
        np.concatenate(
            [diphoton_all_info[var], gjet_all_info[var], df_data[var_uncorr]]
        )
    )

    num_bins = 35
    min_value = mean - 3 * std
    max_value = mean + 3 * std
    bin_width = (max_value - min_value) / num_bins

    # Create the diphoton and g_jet histograms and stack them
    hist_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(diphoton_all_info[var_uncorr], weight=diphoton_all_info["weight"])
    )

    hist_g_jet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(gjet_all_info[var_uncorr], weight=gjet_all_info["weight"])
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
        .fill(df_data[var_uncorr], weight=df_data["weight"])
    )
    hist_data = hist_data / (hist_data.values().sum() * bin_width)
    print((hist_data.values() * bin_width).sum())

    bin_edges = hist_data.axes[0].edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs[0].errorbar(
        bin_centers, hist_data.values(), fmt="x", color="black", label="Data"
    )
    axs[0].set_xlabel(var)
    axs[0].set_ylabel("Frequency")
    axs[0].legend(
        title="Uncorrected",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
    )
    hep.cms.label("Work in Progress", data=False, ax=axs[0])

    # Create the diphoton and g_jet sample histograms and stack them
    hist_samples_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(
            diphoton_all_info[var],
            weight=diphoton_all_info["weight"],
        )
    )
    hist_samples_gjet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(gjet_all_info[var], weight=gjet_all_info["weight"])
    )

    # Normalize the histograms
    sum_sim = hist_samples_diphoton.values().sum() + hist_samples_gjet.values().sum()
    hist_samples_diphoton = hist_samples_diphoton / (sum_sim * bin_width)
    hist_samples_gjet = hist_samples_gjet / (sum_sim * bin_width)

    hep.histplot(
        [hist_samples_diphoton, hist_samples_gjet],
        stack=True,
        histtype="fill",
        label=["Diphoton MC", "GJet MC"],
        ax=axs[1],
    )

    bin_edges = hist_data.axes[0].edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs[1].errorbar(
        bin_centers, hist_data.values(), fmt="x", color="black", label="Data"
    )

    axs[1].set_xlabel(var)
    axs[1].set_ylabel("Frequency")
    axs[1].legend(
        title="Corrected",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
    )
    axs[0].text(
        0.03,
        0.93,
        title,
        transform=axs[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Calculate the residuals
    residuals_samples = hist_data.values() / (
        hist_samples_diphoton.values() + hist_samples_gjet.values()
    )
    residuals_hist = hist_data.values() / (hist_diphoton.values() + hist_g_jet.values())

    # Plot the residuals
    axs[2].errorbar(
        bin_centers,
        residuals_samples,
        fmt="v",
        color="blue",
        alpha=0.8,
        label="Corrected MC ratio to data",
    )
    axs[2].errorbar(
        bin_centers,
        residuals_hist,
        fmt="^",
        color="red",
        alpha=0.8,
        label="MC ratio to data",
    )
    axs[2].axhline(1, color="black")
    axs[2].set_xlabel(var)
    axs[2].set_ylabel("Ratio")
    axs[2].set_ylim(0.2, 1.8)
    axs[2].legend(
        frameon=True, fancybox=True, edgecolor="black", facecolor="white", framealpha=1
    )

    plt.tight_layout()
    plt.savefig(
        "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"
        + title
        + f"/_sample_compare_hist_{var}.png"
    )
    plt.show()


# Direct Comparison of diphoton and g_jet histograms with the corrected samples
def plot_hist(
    diphoton_all_info,
    gjet_all_info,
    df_data,
    var,
    title,
    var_uncorr="",
):
    plt.clf()
    if var_uncorr == "":
        var_uncorr = var[:-8]
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.style.use(hep.style.CMS)

    # Calculate the mean and standard deviation of the data
    mean = np.mean(
        np.concatenate(
            [diphoton_all_info[var], gjet_all_info[var], df_data[var_uncorr]]
        )
    )
    std = np.std(
        np.concatenate(
            [diphoton_all_info[var], gjet_all_info[var], df_data[var_uncorr]]
        )
    )
    num_bins = 50
    min_value = mean - 3 * std
    max_value = mean + 3 * std

    bin_width = (max_value - min_value) / num_bins

    # Create the diphoton and g_jet histograms and stack them
    hist_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(diphoton_all_info[var_uncorr], weight=diphoton_all_info["weight"])
    )

    hist_g_jet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(gjet_all_info[var_uncorr], weight=gjet_all_info["weight"])
    )

    hist_samples_diphoton = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(diphoton_all_info[var], weight=diphoton_all_info["weight"])
    )
    hist_samples_gjet = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(gjet_all_info[var], weight=gjet_all_info["weight"])
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
        label=["MC uncorrected", "MC corrected"],
        ax=axs[0],
    )

    axs[0].set_xlabel(var)
    axs[0].set_ylabel("Frequency")
    axs[0].legend(
        title="Diphoton",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
    )
    hep.cms.label("Work in Progress", data=False, ax=axs[0])

    hep.histplot(
        [hist_g_jet, hist_samples_gjet],
        stack=False,
        histtype="step",
        label=["MC uncorrected", "MC corrected"],
        ax=axs[1],
    )

    axs[1].set_xlabel(var)
    axs[1].set_ylabel("Frequency")
    axs[1].legend(
        title="GJet",
        frameon=True,
        fancybox=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
    )
    axs[0].text(
        0.03,
        0.93,
        title,
        transform=axs[0].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(
        "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/"
        + title
        + f"/_direct_compare_hist_{var}.png"
    )
    plt.show()


def comparison_mvaID(
    df,
    var1="photon_corr_mvaID_run3",
    var2="lead_mvaID",
    title="Comparison MVAID",
    data_name="",
):
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.style.use(hep.style.CMS)

    # Calculate the mean and standard deviation of the data
    mean = 0

    num_bins = 35
    min_value = mean - 1
    max_value = mean + 1
    bin_width = (max_value - min_value) / num_bins

    # Create the histograms
    hist_var1 = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df[var1], weight=df["weight"])
    )

    hist_var2 = (
        hist.Hist.new.Reg(num_bins, min_value, max_value)
        .Weight()
        .fill(df[var2], weight=df["weight"])
    )

    hist_var1 = hist_var1 / (hist_var1.values().sum() * bin_width)
    hist_var2 = hist_var2 / (hist_var2.values().sum() * bin_width)

    hep.histplot(
        hist_var1,
        stack=False,
        histtype="step",
        label="corr_mvaID",
        ax=ax,
        linestyle="-",  # solid line
    )

    hep.histplot(
        hist_var2,
        stack=False,
        histtype="step",
        label="uncorr_mvaID",
        ax=ax,
        linestyle="--",  # dashed line
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_xlim(-1.1, 1.1)
    ax.legend(
        title=title,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        loc="best",
    )
    hep.cms.label("Work in Progress", data=False, ax=ax)

    plt.tight_layout()
    plt.savefig(
        "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/mva_id_corr/"
        + data_name
        + f"_direct_compare_hist_{var1}_{var2}.png"
    )
    plt.show()


def load_photonid_mva_run3(fname: str) -> Optional[xgboost.Booster]:
    """Reads and returns both the EB and EE Xgboost run3 mvaID models"""

    photonid_mva_EB = xgboost.Booster()
    photonid_mva_EB.load_model(fname + "model.json")

    photonid_mva_EE = xgboost.Booster()
    photonid_mva_EE.load_model(fname + "model_endcap.json")

    return photonid_mva_EB, photonid_mva_EE


def calculate_photonid_mva_run3(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    photon: awkward.Array,
) -> awkward.Array:
    """Recompute PhotonIDMVA on-the-fly. This step is necessary considering that the inputs have to be corrected
    with the QRC process. Following is the list of features (barrel has 12, endcap two more):
    EB:
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll

    EE: +
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_hcalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll
        events.Photon.esEffSigmaRR
        events.Photon.esEnergyOverRawE
    """
    photonid_mva, var_order = mva

    if photonid_mva is None:
        return awkward.ones_like(photon.pt)

    bdt_inputs = {}

    bdt_inputs = np.column_stack([np.array(photon[name]) for name in var_order])

    tempmatrix = xgboost.DMatrix(bdt_inputs)

    mvaID = photonid_mva.predict(tempmatrix)

    # Only needed to compare to TMVA
    mvaID = 1.0 - 2.0 / (1.0 + np.exp(2.0 * mvaID))

    return mvaID


def add_corr_photonid_mva_run3_of(photons: awkward.Array) -> awkward.Array:

    preliminary_path = (
        "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/run3_mvaID/"
    )
    photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

    # Now mvaID for the corrected variables

    inputs_EB = [
        "energyRaw",
        "r9_corr_of",
        "sieie",
        "etaWidth_corr_of",
        "phiWidth_corr_of",
        "sieip",
        "s4_corr_of",
        "hoe",
        "ecalPFClusterIso",
        "trkSumPtHollowConeDR03",
        "trkSumPtSolidConeDR04",
        "pfChargedIso",
        "pfChargedIsoWorstVtx",
        "ScEta",
        "fixedGridRhoAll",
    ]

    inputs_EE = [
        "energyRaw",
        "r9_corr_of",
        "sieie",
        "etaWidth_corr_of",
        "phiWidth_corr_of",
        "sieip",
        "s4_corr_of",
        "hoe",
        "ecalPFClusterIso",
        "hcalPFClusterIso",
        "trkSumPtHollowConeDR03",
        "trkSumPtSolidConeDR04",
        "pfChargedIso",
        "pfChargedIsoWorstVtx",
        "ScEta",
        "fixedGridRhoAll",
        "esEffSigmaRR",
        "esEnergyOverRawE",
    ]

    photon_types = ["lead"]
    corrected_mva_id = []

    for photon_type in photon_types:

        # Now calculating the corrected mvaID
        isEB = awkward.to_numpy(np.abs(np.array(photons[photon_type + "_ScEta"])) < 1.5)

        inputs_EB_corr = [
            photon_type + "_" + s if "fixedGridRhoAll" not in s else s
            for s in inputs_EB
        ]

        corr_mva_EB = calculate_photonid_mva_run3(
            [photonid_mva_EB, inputs_EB_corr], photons
        )

        inputs_EE_corr = [
            photon_type + "_" + s if "fixedGridRhoAll" not in s else s
            for s in inputs_EE
        ]

        corr_mva_EE = calculate_photonid_mva_run3(
            [photonid_mva_EE, inputs_EE_corr], photons
        )
        corrected_mva_id.append(awkward.where(isEB, corr_mva_EB, corr_mva_EE))

    return corrected_mva_id[0]


def add_corr_photonid_mva_run3_data(photons: awkward.Array) -> awkward.Array:

    preliminary_path = (
        "/home/home1/institut_3a/jaensch/Documents/BA/BA/Diphoton/run3_mvaID/"
    )
    photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

    # Now mvaID for the corrected variables

    inputs_EB = [
        "energyRaw",
        "r9",
        "sieie",
        "etaWidth",
        "phiWidth",
        "sieip",
        "s4",
        "hoe",
        "ecalPFClusterIso",
        "trkSumPtHollowConeDR03",
        "trkSumPtSolidConeDR04",
        "pfChargedIso",
        "pfChargedIsoWorstVtx",
        "ScEta",
        "fixedGridRhoAll",
    ]

    inputs_EE = [
        "energyRaw",
        "r9",
        "sieie",
        "etaWidth",
        "phiWidth",
        "sieip",
        "s4",
        "hoe",
        "ecalPFClusterIso",
        "hcalPFClusterIso",
        "trkSumPtHollowConeDR03",
        "trkSumPtSolidConeDR04",
        "pfChargedIso",
        "pfChargedIsoWorstVtx",
        "ScEta",
        "fixedGridRhoAll",
        "esEffSigmaRR",
        "esEnergyOverRawE",
    ]

    photon_types = ["lead"]
    corrected_mva_id = []

    for photon_type in photon_types:

        # Now calculating the corrected mvaID
        isEB = awkward.to_numpy(np.abs(np.array(photons[photon_type + "_ScEta"])) < 1.5)

        inputs_EB_corr = [
            photon_type + "_" + s if "fixedGridRhoAll" not in s else s
            for s in inputs_EB
        ]

        corr_mva_EB = calculate_photonid_mva_run3(
            [photonid_mva_EB, inputs_EB_corr], photons
        )

        inputs_EE_corr = [
            photon_type + "_" + s if "fixedGridRhoAll" not in s else s
            for s in inputs_EE
        ]

        corr_mva_EE = calculate_photonid_mva_run3(
            [photonid_mva_EE, inputs_EE_corr], photons
        )
        corrected_mva_id.append(awkward.where(isEB, corr_mva_EB, corr_mva_EE))

    return corrected_mva_id[0]
