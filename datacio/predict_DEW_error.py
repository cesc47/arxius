import pandas as pd
from torchvision import models, transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm


# Path configurations
src_path = '/media/francesc2/datasets/arxius'
tsv_path = '/media/francesc2/datasets/arxius/Extraccio_12345.tsv'

# Load DataFrame
df = pd.read_csv(tsv_path, sep='\t')
df['filename'] = df['filename'].astype(str)
df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1])
df['data_inici'] = pd.to_datetime(df['data_inici'], format='%d/%m/%Y', errors='coerce')
df['data_fi'] = pd.to_datetime(df['data_fi'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['data_inici', 'data_fi'])

# Define model
weights = models.ConvNeXt_Base_Weights.DEFAULT
model = models.convnext_base(weights=weights)
model.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(1024, 14, bias=True)  # 14 classes
)

# Load checkpoint
checkpoint = torch.load('convnext_DEWB.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, src_path, df, folder_num, transform=None):
        self.src_path = src_path
        self.df = df
        self.folder = f'Extraccio_{folder_num}'
        self.images = os.listdir(f'{src_path}/{self.folder}/images')
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = f"{self.src_path}/{self.folder}/images/{image_name}"

        # Process image filename and get corresponding row
        image_filename = image_name.split('_512x512')[0] + '.jpg'
        row = self.df[self.df['filename'] == image_filename]

        # Open image and apply transformations
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Retrieve min and max year from the DataFrame row
        if not row.empty:
            min_year = row['data_inici'].dt.year.values[0]
            max_year = row['data_fi'].dt.year.values[0]
        else:
            min_year, max_year = -1, -1  # Return None if no match in DataFrame

        return image, min_year, max_year


# Batch processing function
def batch_process(images, min_years, max_years):
    input_batch = images.to(device)

    with torch.no_grad():
        output = model(input_batch)
    pred_classes = torch.argmax(output, dim=1).cpu().numpy()
    year_preds = dataset.min_year + np.floor(0.5 + ((dataset.max_year - dataset.min_year) / num_classes) * pred_classes).astype(int)

    # Calculate error
    errors = []
    for year_pred, min_year, max_year in zip(year_preds, min_years, max_years):
        if min_year == torch.tensor(-1) or max_year == torch.tensor(-1):
            continue
        if min_year <= year_pred <= max_year:
            errors.append(0)
        else:
            errors.append(min(abs(year_pred - min_year), abs(year_pred - max_year)))
    return errors


# Main processing loop
total_error = 0
total_images = 0

for i in range(1, 6):
    # Create DataLoader for each folder
    dataset = ImageDataset(src_path, df, folder_num=i, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    print(f'Processing folder Extraccio_{i} with {len(dataset)} images')

    # Loop through DataLoader
    for images, min_years, max_years in tqdm(dataloader):
        errors = batch_process(images, min_years, max_years)
        total_error += sum(errors)
        total_images += len(errors)

print(f"Total images processed: {total_images}")
print(f"Total error: {total_error}")
print(f"Mean error: {total_error / total_images}")

