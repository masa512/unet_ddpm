import torch 
import torch.nn as nn
import tqdm

def train_ddpm(
    model,
    dataloader,
    optimizer,
    fD,
    epochs = 10,
    loss_function = nn.MSELoss(), 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    # Model and fD to device
    model.to(device)
    #fD.to(device)
    model.train()

    for epoch in tqdm.tqdm(range(epochs)):
        epoch_loss = 0.0

        for x_0,noise in dataloader:
            optimizer.zero_grad()
            # Load the data to gpu
            
            # Extract random time vector
            _t = torch.randint(low=0,high=fD.get_max_time(),size=(x_0.size(0),1))

            # Apply the forward diffusion
            x_t , noise = fD.diffuse(x_0,_t)
            x_t = x_t.to(device)
            noise = noise.to(device)
            
            # Apply model to x_t
            noise_pred = model(x_t,_t)

            # Evaluate loss between y_t and noise
            loss = loss_function(noise,noise_pred)

            # Gradient descent
            loss.backward()
            optimizer.step()

            # Add to epoch_loss
            epoch_loss += loss.item()
        
        print(f"Training Loss at epoch {epoch}: {epoch_loss / len(dataloader)}")
