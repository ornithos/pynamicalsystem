# pynamicalsystem
<**IN PROGRESS**>

A collection of (discrete time) latent Markov model implementations written in python. 

The intention was to port the dynamicalSystems MATLAB repo into python. However, I'm taking a step back to think more carefully about what this should look like. For instance, to what extent can we now take advantage of automatic differentiation tools? I'm hoping to develop this further once I have a better idea of where I'm going, but for the time being this is just a record of some time series stuff in python that will serve as a useful reference for me.

A brief introduction to the files as of Jan 2018:
* core -- the beginning of the MATLAB port: sketching out the dynamicalSystem class (*currently mothballed*)
* numpylds -- a relatively simple implementation of a linear dynamical system using numpy
* torchlds -- a port of `numpylds` into pytorch
* torchlds-module -- creating a pytorch module for an LDS which is implemented similarly to a neural network.
