import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os

# include whatever other imports you need here

class DiceDataset(Dataset):
    def __init__(self, image_file, df, transform=None, train=True):
        self.image_file = image_file
        self.df = df
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_file, row['filename'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = PRODUCT_TO_IDX[int(row['product'])]     #get label as class number
        return img, label

dataset = 'dice_images/train/images'
labels_csv = 'dice_images/train/labels.csv'

labels_df = pd.read_csv(labels_csv)

all_products = sorted(labels_df['product'].unique())        # Get sorted list of unique products
NUM_CLASSES = len(all_products)
PRODUCT_TO_IDX = {product: idx for idx, product in enumerate(all_products)}     #create mapping from product to index/class no.(eg. product 10 -> index 0, product 60 -> index 1 etc.)
IDX_TO_PRODUCT = {idx: product for idx, product in enumerate(all_products)}     #create mapping from index to product

class DiceNN(nn.Module):
    def __init__(self,num_classes):
            super(DiceNN, self).__init__()
            self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),       
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                         
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            )
    
    def forward(self, x):
        x = self.layers(x)
        return x



def predict(parameters):
    # Determine which device the input tensor is on
    device = torch.device("cuda" if parameters.is_cuda else "cpu")

    model = DiceNN(NUM_CLASSES) # Add your model init parameters here if you have any
    # Move to same device as input
    model = model.to(device)
    # Load network weights
    model.load_state_dict(torch.load('weights_product.pth',map_location=torch.device(device)))
    # Put model in evaluation mode
    model.eval()

    # Optional: do whatever preprocessing you do on the inputs
    # if not included as transformations inside the model

    X_mean  = torch.tensor([0.0764, 0.1152, 0.3394]).view(1,3,1,1) 
    X_scale = torch.tensor([1.4133, 1.4786, 1.4816]).view(1,3,1,1) 

    parameters = (parameters - X_mean) / X_scale
    
    with torch.no_grad():
        # Pass inputs to model
        predicted_decay_times = model(parameters)

    # If your output needs any post-processing, do it here
            
    # Return predicted products
    return torch.tensor([IDX_TO_PRODUCT[i.item()] for i in predicted_decay_times.argmax(dim=1)],
                        dtype=torch.long).unsqueeze(1)

