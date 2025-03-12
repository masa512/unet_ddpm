import torch
from torch.cuda import is_available 
import torch.nn as nn
import tqdm
import random
from torch.utils.data import DataLoader, Subset

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


def reverse_ddpm(
    model,
    dataloader,
    target_labels,
    sinu_embedder,
    fD,
    n_sample = 1,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    # Choose digit of interest using subset
    dataset = dataloader.dataset
    filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in target_labels]

    # Random select from the list
    indices = random.choices(filtered_indices,k = min(len(filtered_indices),n_sample))

    # Extract subset
    subset = Subset(dataset,indices)

    # Create dataloader and extract next data
    dataloader = DataLoader(subset,batch_size=len(indices),shuffle=True)
    x_0, label = next(iter(dataloader))

    # Initialize _t to Tmax. We will first diffuse the clean imaages x_0 forward to t = Tmax
    Tmax = fD.get_max_time()
    batch_size = x_0.size(0)

    _t = torch.full(size=(batch_size,),fill_value=Tmax-1)
    x_t,noise  = fD.diffuse(x_0,_t)

    # Launch necessary NN settings
    model.to(device)
    model.eval()

    # Save snapshots
    snapshots = []

    # Iteratively diffuse downward
    with torch.no_grad():
        for t in reversed(range(Tmax)):
            snapshots.append(x_t)
            x_t = x_t.to(device)
            # Apply sinu embedding to _t
            t_emb = torch.index_select(sinu_embedder,dim=0,index=_t.squeeze())
            t_emb = t_emb.to(device)
            # Apply model to x_t
            noise_pred = model(x_t,t_emb)
            # Detach from GPU
            x_t= x_t.to('cpu')
            noise_pred= noise_pred.to('cpu')
            # Apply the reverse DDPM on x_t
            x_t = fD.reverse(x_t,_t,noise_pred)
            # Update the time vector
            _t = _t - 1  

    snapshots.reverse()  
    return snapshots,x_0

def generate_sample(
    model,
    sinu_embedder,
    fD,
    d1,
    d2,
    n_sample,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    # Initialize Noises
    x_t = torch.randn(size=(n_sample,1,d1,d2)).to(device)

    # Time vector
    Tmax = fD.get_max_time()
    _t = torch.full(size=(n_sample,),fill_value=Tmax-1)

    # Launch necessary NN settings
    model.to(device)
    model.eval()

    # Iteratively diffuse downward
    with torch.no_grad():
        for t in reversed(range(Tmax)):
            x_t = x_t.to(device)
            # Apply sinu embedding to _t
            t_emb = torch.index_select(sinu_embedder,dim=0,index=_t.squeeze())
            t_emb = t_emb.to(device)
            # Apply model to x_t
            noise_pred = model(x_t,t_emb)
            # Detach from GPU
            x_t= x_t.to('cpu')
            noise_pred= noise_pred.to('cpu')
            # Apply the reverse DDPM on x_t
            x_t = fD.reverse(x_t,_t,noise_pred)
            # Update the time vector
            _t = _t - 1  

    return x_t



