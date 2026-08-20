import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# include whatever other imports you need here

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("using device:", device)


class OrbitalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DecayPredictionNetwork(nn.Module):
    def __init__(self):
            super(DecayPredictionNetwork, self).__init__()
            self.layers = nn.Sequential(
            nn.Linear(6, 64), 
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 128), # 
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 256), 
            nn.GELU(),
            nn.Linear(256, 128), 
            nn.GELU(),
            nn.Linear(128, 64), 
            nn.GELU(),
            nn.Linear(64, 1)
            )
    
    def forward(self, x):
        x = self.layers(x)
        return x


def predict(parameters):
    # Determine which device the input tensor is on
    device = torch.device("cuda" if parameters.is_cuda else "cpu")

    model = DecayPredictionNetwork() # Add your model init parameters here if you have any
    # Move to same device as input
    model = model.to(device)
    # Load network weights
    model.load_state_dict(torch.load('weights_decay.pth',map_location=torch.device(device)))
    # Put model in evaluation mode
    model.eval()

    # Optional: do whatever preprocessing you do on the inputs
    # if not included as transformations inside the model

    #log
    X_mean  = torch.tensor([3.98277270e+02, 3.49523047e+00, 5.72043343e-01, 2.52370038e-02, 1.57890964e+02, 2.24501227e+00], dtype=torch.float32)
    X_scale = torch.tensor([1.12349110e+02, 1.61882699e+00, 6.08788465e-01, 1.45807047e-02, 5.24628879e+01, 4.22512174e-01], dtype=torch.float32)
    y_mean  = 5.88786729
    y_scale =  1.64759833

    parameters[:, 1] = torch.log1p(parameters[:, 1])  # satellite_mass_kg
    parameters[:, 2] = torch.log1p(parameters[:, 2])  # cross_sectional_area_m2

    parameters = parameters.to(device)
    parameters = (parameters - X_mean) / X_scale
    
    with torch.no_grad():
        # Pass inputs to model
        predicted_decay_times = model(parameters)

    # If your output needs any post-processing, do it here
    # Return predicted decay times in days

    return torch.expm1 (predicted_decay_times * y_scale + y_mean)

