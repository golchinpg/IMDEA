# IMDEA Collaboration

This repository contains the traffic converter code which can convert the pcap file to the feature file. Also, it contains some preprocessing, making it possible to change them based on different tasks and applications.

It contains an AutoEncoder with constant and flexible threshold codes. The AutoEncoders are designed for the specific dataset available in the Dataset repository. Moreover, the Autoencoder is extended with a denoising autoencoder to increase the robustness of the model. 

It contains an algorithm for binning the KPI --> finding out in which bin the outliers happen --> cutting the data to the bins that have outliers and applying the XGBoost classifier model to the data. 


