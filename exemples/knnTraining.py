import torch

from MaxEntML.ploting.plotLambdas import plotLambdas, plotLambdasLog, plotLambdasSymLog
from MaxEntML.models.knn import kNNRegressorModel
from MaxEntML.dtbProcessing.qr2xy import qr2xyScaler
from MaxEntML.dtbProcessing.identityScaler import identityScaler
from MaxEntML.dtbProcessing.arcsinhScaler import arcsinhScaler
from MaxEntML.dtbProcessing.minMaxScaler import minMaxScaler
from MaxEntML.dtbProcessing.featureAugmentation import featureAugmentation
from MaxEntML.dtbProcessing.inverse import inverse
from MaxEntML.dtbProcessing.stdScaler import stdScaler
from MaxEntML.dataRepresentation.serie import serie
from MaxEntML.trainingProcedures.knnTraining import train
from MaxEntML.dtbGeneration.dtbGen import generateTrainingDatabase, generateValidationDatabase, clean
from MaxEntML.validationProcedure.generalValidation import validate
from MaxEntML.dataRepresentation.qr2xyData import qr2xyData

# Set to True for (x, y) geometric space, or False for (q, r) space
use_xy = True

dtbFile = "../databases/grid_with_lambdas.csv"
cleanedDtbFile = "../databases/grid_with_lambdas_clean.csv"

storingDevice = 'cpu'
storingDtype = torch.float64

computeDevice = 'cuda:1'
computeDtype = torch.float64

## generation de la base de données de train
qr2xyDataObj = qr2xyData(q_max=3, r_max=15, eps=1e-1, beta_q=1.5, gamma_r=1.5)

qr, lambdas = generateTrainingDatabase(
    qr2xyDataObj, Nx=int(0.5**(0.5)*200), Ny=int(0.5**(0.5)*80), batch_size=30000,
    storingDtype=storingDtype, storingDevice=storingDevice,
    computeDevice=computeDevice, computeDtype=computeDtype,
)

plotLambdasSymLog(qr, lambdas)

## nettoyage de la base de données
qr, lambdas = clean(qr, lambdas, verbose=True)

## generation de la base de données de validation
qr_val, lambdas_val = generateValidationDatabase(qr2xyDataObj, Nsamples=2000, storingDtype=storingDtype, storingDevice=storingDevice, computeDevice=computeDevice, computeDtype=computeDtype)
plotLambdasSymLog(qr_val, lambdas_val)
qr_val, lambdas_val = clean(qr_val, lambdas_val, verbose=True)

#_________________________________________________________

## scaling des sorties : arcsinh (lisse en 0) + standardisation
lambdasScaler = arcsinhScaler() + stdScaler()
lambdasScaled = lambdasScaler.fit_transform(lambdas)

## scaling des entrées selon la variable use_xy
if use_xy:
    print("--> Using geometric input scaler (x, y)")
    inputScaler = qr2xyScaler(qr2xyDataObj)
else:
    print("--> Using standard input scaler (q, r)")
    inputScaler = stdScaler()

## Transform training inputs to chosen feature space
inData = inputScaler.fit_transform(qr)

## Plot output values in transformed input space
plotLambdas(inData, lambdasScaled)

## model creation
model = kNNRegressorModel(n_neighbors=5)

## model training
inDataClean, lambdasClean = clean(inData, lambdasScaled, verbose=True)
model.inputScaler = inputScaler
model.outputScaler = lambdasScaler
train(model, inDataClean, lambdasClean)

## sauvegarde de tous les elements
qr2xyDataObj.save('qr2xy.json')
qr.save('qr.pt')
lambdas.save('lambdas.pt')
model.save('model.pkl')

## model validation
lambda_err = validate(model, qr_val, lambdas_val)

## Plot validation error in raw (q, r) space
plotLambdasLog(qr_val, lambda_err)

## Plot validation error in chosen input space (qr / xy)
in_val = model.inputScaler.transform(qr_val)
plotLambdasLog(in_val, lambda_err)