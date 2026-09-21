import torch

def validate(model, physicalInputs, physicalOutput):
    preds = model.predict(physicalInputs)

    error = torch.abs(preds.data - physicalOutput.data)

    return(error)