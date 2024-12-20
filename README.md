# IMDEA Collaboration

This repository contains the traffic converter code which can convert the pcap file to the feature file. Also, it contains some preprocessing, making it possible to change them based on different tasks and applications.

It contains an AutoEncoder with constant and flexible threshold codes. The AutoEncoders are designed for the specific dataset available in the Dataset repository. Moreover, the Autoencoder is extended with a denoising autoencoder to increase the robustness of the model. 

It contains an algorithm for binning the KPI --> finding out in which bin the outliers happen --> cutting the data to the bins that have outliers and applying the XGBoost classifier model to the data. 

## Future research
make the network traffic datasets similar using the traffic converter code to increase the generalization and training models with more data. 

Test the AE with other network traffic datasets to investigate their generalizability.

In the current pipeline, a logarithmic filter is applied to understand where the outliers happen --> it is possible to change it with the AutoEncoder or Self-Supervised Contrastive Learning for more complicated data in which there is a chance that outliers happen in different bins. 

** Using intrusion detection data as there are some ground-truth labels 
** How to generate the ground truth labels?
  - Transformers
  - SSCL-IDS

** how to get the normal flows?
keep the ones that are normals --> based on the threshold
