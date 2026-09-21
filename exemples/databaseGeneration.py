import numpy as np
import torch

from MaxEntML.dataRepresentation.serie import serie
from MaxEntML.dataRepresentation.qr2xyData import qr2xyData
from MaxEntML.ploting.plotLambdas2D import plotLambdas
from MaxEntML.dtbGeneration.dtbGen import generateTargets, generateTrainingDatabase, generateValidationDatabase

#_________________________________________________________

dtbFile = "../databases/grid_with_lambdas.csv"

storingDevice = 'cuda:0'
storingDtype = torch.float64

#_________________________________________________________

## Read validation dtb for reference
data = np.loadtxt(dtbFile, delimiter=",", skiprows=1)

xy = serie(data[:, 0:2], colName=['x', 'y'], storingDtype=storingDtype, storingDevice=storingDevice)
qr = serie(data[:, 2:4], colName=['q', 'r'], storingDtype=storingDtype, storingDevice=storingDevice)
lambdas = serie(data[:, 4:9], colName=[f'l{i}' for i in range(5)], storingDtype=storingDtype, storingDevice=storingDevice)

## generate lambdas with the exact same points for validating the lambda generation code
lamb_new = generateTargets(qr)

lambda_diff = serie(torch.abs(lambdas.data - lamb_new.data), colName=lamb_new.colName, storingDtype=storingDtype, storingDevice=storingDevice)

plotLambdas(qr, lambda_diff)

## generate a full deterministic database
qr2xyDataObj = qr2xyData(q_max=3, r_max=15, eps=1e-1, beta_q=1.5, gamma_r=1.5)
qr_new, lambda_new = generateTrainingDatabase(qr2xyDataObj, Nx=50, Ny=20, storingDtype=storingDtype, storingDevice=storingDevice)
plotLambdas(qr_new, lambda_new)

## generate a validation database
qr_val, lambda_val = generateValidationDatabase(qr2xyDataObj, Nsamples=500, scramble=True, storingDtype=storingDtype, storingDevice=storingDevice)
