
def train(knnModel: "kNNRegressorModel", scaledInputData: "serie", scaledOutputData: "serie"):
    knnModel.fit(scaledInputData, scaledOutputData)
