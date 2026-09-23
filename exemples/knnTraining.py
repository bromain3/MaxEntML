import torch

from MaxEntML.ploting.plotLambdas import plotLambdas3D, plotLambdasLog3D, plotLambdasSymLog3D
from MaxEntML.models.knn import kNNRegressorModel
from MaxEntML.dtbProcessing.qr2xy import qr2xyScaler
from MaxEntML.dtbProcessing.identityScaler import identityScaler
from MaxEntML.dtbProcessing.symLogScaler import symlogScaler
from MaxEntML.dtbProcessing.minMaxScaler import minMaxScaler
from MaxEntML.dtbProcessing.featureAugmentation import featureAugmentation
from MaxEntML.dtbProcessing.inverse import inverse
from MaxEntML.dtbProcessing.stdScaler import stdScaler
from MaxEntML.trainingProcedures.knnTraining import train
from MaxEntML.dtbGeneration.dtbGen import generateTrainingDatabase, generateValidationDatabase, clean
from MaxEntML.validationProcedure.generalValidation import validate
from MaxEntML.dataRepresentation.qr2xyData import qr2xyData


#_________________________________________________________

dtbFile = "../databases/grid_with_lambdas.csv"
cleanedDtbFile = "../databases/grid_with_lambdas_clean.csv"

storingDevice = 'cpu'
storingDtype = torch.float64

computeDevice = 'cuda:0'
computeDtype = torch.float64

#_________________________________________________________

## generation de la base de données de train
qr2xyDataObj = qr2xyData(q_max=3, r_max=15, eps=1e-1, beta_q=1.5, gamma_r=1.5)
qr, lambdas = generateTrainingDatabase(qr2xyDataObj, Nx=int(0.5**(0.5)*200), Ny=int(0.5**(0.5)*80), batch_size=50000, storingDtype=storingDtype, storingDevice=storingDevice, computeDevice=computeDevice, computeDtype=computeDtype)
plotLambdasSymLog3D(qr, lambdas)
qr, lambdas = clean(qr, lambdas, verbose=True)

## generation de la base de données de validation
qr_val, lambdas_val = generateValidationDatabase(qr2xyDataObj, Nsamples=2000, storingDtype=storingDtype, storingDevice=storingDevice, computeDevice=computeDevice, computeDtype=computeDtype)
plotLambdasSymLog3D(qr_val, lambdas_val)
qr_val, lambdas_val = clean(qr_val, lambdas_val, verbose=True)

## generation de la base de données de validation

## scaling
#xyScaler = qr2xyScaler(qr2xyDataObj)
#xy = xyScaler.fit_transform(qr)
#inScaler = qr2xyScaler(q_max=3, r_max=15, eps=0, beta_q=1.5, gamma_r=2) + featureAugmentation([[inverse() + minMaxScaler()], [inverse() + minMaxScaler()]])
#inData = inScaler.fit_transform(qr)
#print(inData.data)
#print(inData.colName)
qrScaler = stdScaler()
qrScaled = qrScaler.fit_transform(qr)
lambdasScaler = symlogScaler() + stdScaler()
lambdasScaled = lambdasScaler.fit_transform(lambdas)

## ploting the lambdas in the xy plane to check
plotLambdas3D(qrScaled, lambdasScaled)


## model creation
model = kNNRegressorModel(n_neighbors=5)

## model training (and associating scalers to the model depending on inputs)
inData, lambdas = clean(qrScaled, lambdasScaled, verbose=True)
model.inputScaler = qrScaler
model.outputScaler = lambdasScaler
train(model, inData, lambdasScaled)

## sauvegarde de tous les elements
qr2xyDataObj.save('qr2xy.json')
qr.save('qr.pt')
lambdas.save('lambdas.pt')
model.save('model.pkl')

## model validation
lambda_err = validate(model, qr_val, lambdas_val)
plotLambdasLog3D(qr_val, lambda_err)


