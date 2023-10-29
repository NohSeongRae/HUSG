from dataloader import DataLoader
from model import CVAE
from train import train
from infer import infer
import torch

def main():
    # Load and preprocess data
    dataloader = DataLoader('data/building_parcel_polygon.csv', 'data/bounding_box.csv')
    data, conditions = dataloader.load_data()

    # Initialize model
    model = CVAE(input_dim=data.shape[1], condition_dim=conditions.shape[1])

    # Load pre-trained weights if they exist
    try:
        model.load_state_dict(torch.load('model_weights.pth'))
    except FileNotFoundError:
        pass

    # Train model
    for epoch in range(100):
        train(model, data, conditions)

        # Save model weights every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'model_weights.pth')

    # Infer new data
    new_data = infer(model, conditions[0])

    print(new_data)

if __name__ == "__main__":
    main()



