Description:
This repository contains Python scripts for correcting Monte Carlo (MC) simulations of diphoton production. The main script, flow.py, trains a model to perform the correction, while flow_correction.py applies the correction to the simulations and visualizes the results.

Files:

    flow.py: This script trains a model for correcting MC simulations of diphoton production. It utilizes deep learning (normalizing flows) to learn the correction.

    flow_correction.py: Once the correction model is trained using flow.py, this script applies the correction to the MC simulations. It then visualizes the corrected results.
