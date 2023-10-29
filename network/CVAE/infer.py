import torch

def infer(model, condition):
    # Load pre-trained weights
    model.load_state_dict(torch.load('model_weights.pth'))

    z = torch.randn(1, model.decoder[0].in_features - condition.shape[0])
    c = torch.tensor(condition).unsqueeze(0)
    return model.decoder(torch.cat([z, c], dim=1)).detach().numpy()