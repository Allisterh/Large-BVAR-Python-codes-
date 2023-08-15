

# Python Codes for Large Bayesian VAR of the US Economy with Covid Volatility (Preliminary Stage)

## Overview

This repository contains the Python functions translated from original MATLAB files to run a large Bayesian VAR (BVAR) model. The details of the model are present in the paper titled "Large BVAR of the US Economy" authored by Richard Crump, Stefano Eusepi, Domenico Giannone, Eric Qian, and Argia Sbordone in 2021. Using this model as a benchmark on a smaller scale, Lenza and Primiceri extended it to account for volatility arising due to covid that struck and affected the US economy in March 2020. Harnessing the power of big data on Bayesian VARs, I am in the process of translating all MATLAB functions and main scripts that fit both types of model - with and without covid volatility.

## Functions

The large_bvar.py file includes several functions that implement different components of the BVAR model from reading the data, transforming and fitting the model to constructing and plotting forecasts and scenario analyses. 
The code is in the preliminary testing stage, and most of the functions are working as expected. I am still validating, optimizing, and debugging certain parts of large functions, and will update the repository as I progress.
I will add more functions in the large_bvar.py file, and add other Python scripts to run the model, construct scenario analyses, plot forecasts, fan charts, and other distributions of the parameters. 

## Reference

I translated the codes based on the following paper:

Crump, Richard K., Stefano Eusepi, Domenico Giannone, Eric Qian, and Argia M. Sbordone. “A Large Bayesian VAR of the United States Economy.” SSRN Electronic Journal, 2021. https://doi.org/10.2139/ssrn.3908154.

\vsapce{1cm}

Lenza, Michele, and Giorgio Primiceri. “How to Estimate a VAR after March 2020.” Journal of Applied Economics 37, no. 4 (July 2022): 688–99. https://doi/10.1002/jae.2895

## Usage

Please refer to the docstrings within individual functions in the large_bvar.py file for details on how to use each function, including input parameters and return values.

## Contributing

I welcome suggestions to improve and test the Python code. Please feel free to open an issue or submit a pull request. You may contact me for inquiries at joshi27s@uw.edu.



