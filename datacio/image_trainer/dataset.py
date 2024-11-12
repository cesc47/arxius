import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class DATES_ARXIUS(Dataset):
    def __init__(self, args, min_year=1900, max_year=2025, transform=None):
        self.root_dir = os.path.join(args.data_path, 'arxius')
        self.df = self.load_annotations(os.path.join(self.root_dir, f'Extraccio_12345.tsv'))
        self.min_year = min_year
        self.max_year = max_year
        self.classes = int((max_year - min_year) / 5) + 1
        self.df['data_mean'] = self.df['data_mean'].clip(self.min_year, self.max_year)
        self.transform = transform  # Add transform argument

    def __getitem__(self, idx):
        # Load image and apply transform
        row = self.df.iloc[idx]
        extraccio = row['extracció']
        extraccio = extraccio[0].upper() + extraccio[1:]
        filename = row['filename'].split('/')[-1]
        filename = filename.split('.')[0] + '_512x512.jpg'
        img_name = os.path.join(self.root_dir, extraccio, 'images', filename)

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = int(row['data_mean'] / 5) - int(self.min_year / 5)
        return np.array(image), labels, row['data_inici'], row['data_fi']

    def __len__(self):
        return len(self.df)

    def load_annotations(self, path):
        """
        Load the annotations from the csv file.
        """
        df = pd.read_csv(path, sep='\t')

        # Extract filename and set it to the correct format
        df['filename'] = df['filename'].astype(str).str.split('/').str[-1]

        # Convert date columns, drop rows with NaT (missing dates), and extract years
        df['data_inici'] = pd.to_datetime(df['data_inici'], format='%d/%m/%Y', errors='coerce').dt.year
        df['data_fi'] = pd.to_datetime(df['data_fi'], format='%d/%m/%Y', errors='coerce').dt.year
        df.dropna(subset=['data_inici', 'data_fi'], inplace=True)

        # Take the midpoint of the date range
        df['data_mean'] = (df['data_inici'] + df['data_fi']) / 2

        # Check if all images are available using vectorized file existence check
        def construct_img_path(row):
            extraccio = row['extracció'].capitalize()  # Capitalize first letter
            filename = row['filename'].split('.')[0] + '_512x512.jpg'
            return os.path.join(self.root_dir, extraccio, 'images', filename)

        img_paths = df.apply(construct_img_path, axis=1)
        exists = [os.path.exists(img_path) for img_path in img_paths]
        df = df[exists]  # Filter rows where images exist

        print(f"Removed {len(exists) - sum(exists)} images from the dataset because they were not found.")

        return df


if __name__ == '__main__':
    from argparse import Namespace
    args = Namespace(data_path='/media/francesc2/datasets', classes=14, min_year=1930, train_crop_size=224,
                     interpolation='bilinear', val_crop_size=224, val_resize_size=256, split='train')
    dataset = DATES_ARXIUS(args)

    for i in range(10):
        img, label = dataset[i]
        print(img.shape, label)