import torch
from torch.cuda import is_available 
import torch.nn as nn
import tqdm

def train_ddpm(
    model,
    dataloader,
    optimizer,
    sinu_embedder,
    fD,
    epochs = 10,
    loss_function = nn.MSELoss(), 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    print(f'Running With {device}')

    # Model and fD to device
    model.to(device)
    model.train()

    for epoch in tqdm.tqdm(range(epochs)):
        epoch_loss = 0.0

        for x_0, label in dataloader:
            optimizer.zero_grad()
            
            # Extract random time vector
            _t = torch.randint(low=0,high=fD.get_max_time(),size=(x_0.size(0),1))

            # Apply the forward diffusion
            x_t , noise = fD.diffuse(x_0,_t)
            x_t = x_t.to(device)
            noise = noise.to(device)

            # Apply sinusoidal embedding here
            t_emb = torch.index_select(sinu_embedder,dim=0,index=_t.squeeze())
            t_emb = t_emb.to(device)

            # Apply model to x_t
            noise_pred = model(x_t,t_emb)

            # Evaluate loss between y_t and noise
            loss = loss_function(noise,noise_pred)

            # Gradient descent
            loss.backward()
            optimizer.step()

            # Add to epoch_loss
            epoch_loss += loss.item()
        
        print(f"Training Loss at epoch {epoch}: {epoch_loss / len(dataloader)}")


def test_ddpm(
    model,
    dataloader,
    sinu_embedder,
    fD,
    loss_function = nn.MSELoss(), 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    print(f"Running with {device}...")

    model.to(device)
    model.eval()

    loss = 0.0
    with torch.no_grad():
        for x_0, label in dataloader:
            # Extract random time vector
            _t = torch.randint(low=0,high=fD.get_max_time(),size=(x_0.size(0),1))

            # Apply the forward diffusion
            x_t , noise = fD.diffuse(x_0,_t)
            x_t = x_t.to(device)
            noise = noise.to(device)

            # Apply sinusoidal embedding here
            t_emb = torch.index_select(sinu_embedder,dim=0,index=_t.squeeze())
            t_emb = t_emb.to(device)

            # Apply model to x_t
            noise_pred = model(x_t,t_emb)

            # Evaluate loss between y_t and noise
            loss += loss_function(noise,noise_pred).item()
    print(f"Evaluation loss is: {loss/len(dataloader)}")


def sample_ddpm(
    model,
    dataset,
    digit,
    sinu_embedder,
    fD,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):


