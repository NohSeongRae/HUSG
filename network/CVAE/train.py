import torch
from torch import optim

def train(model, data, conditions, epochs=100, batch_size=64, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss(reduction='sum')

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            x = torch.tensor(data[i:i+batch_size])
            c = torch.tensor(conditions[i:i+batch_size])

            optimizer.zero_grad()

            recon_x, mu, logvar = model(x, c)

            # Compute loss
            BCE = criterion(recon_x, x)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + KLD

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs} Loss: {loss.item()}')