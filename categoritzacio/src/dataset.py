import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from hierarchy import load_graph


class CATEGORITZACIO(Dataset):
    def __init__(self, args, transform=None):
        self.root_dir = os.path.join(args.data_path, 'arxius')
        self.df = self.load_annotations(os.path.join(self.root_dir, f'Extraccio_12345.tsv'))
        self.tree = load_graph('simple_ontology_graph.gexf')
        self.labels = list(self.tree.nodes)
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}  # Mapping for fast lookup
        self.gestionate_categories()
        self.num_classes = len(self.labels)
        self.transform = transform

    def load_annotations(self, annotations_file):
        # Load TSV file as a DataFrame
        df = pd.read_csv(annotations_file, sep='\t').dropna(subset=['categories']).reset_index(drop=True)
        df['categories'] = df['categories'].astype(str)
        df['filename'] = df['filename'].astype(str).str.split('/').str[-1]

        # Vectorized file path construction
        df['img_path'] = df.apply(
            lambda row: os.path.join(
                self.root_dir,
                row['extracció'].capitalize(),
                'images',
                row['filename'].split('.')[0] + '_512x512.jpg'
            ),
            axis=1
        )

        # Filter for existing files using numpy for performance
        img_paths = df['img_path'].values
        exists = np.vectorize(os.path.exists)(img_paths)
        df = df[exists]

        print(f"Removed {len(exists) - sum(exists)} images from the dataset because they were not found.")

        return df.reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = Image.open(row['img_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load labels with optimized lookup
        categories = row['categories'].split(';')
        labels = np.zeros(len(self.labels), dtype=np.float32)
        for category in categories:
            if category in self.label_to_index:
                labels[self.label_to_index[category]] = 1

        return np.array(image), labels

    def gestionate_categories(self):
        """
        Remove categories that are not present in the dataset.
        """
        label_counts = np.zeros(len(self.labels), dtype=np.int32)

        # Fast label occurrence counting
        for categories in self.df['categories']:
            for category in categories.split(';'):
                if category in self.label_to_index:
                    label_counts[self.label_to_index[category]] += 1

        # Retain only used labels
        used_indices = np.where(label_counts > 0)[0]
        self.labels = [self.labels[idx] for idx in used_indices]
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Loader')
    parser.add_argument('--data_path', type=str, default='/media/francesc2/datasets', help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the DataLoader')
    args = parser.parse_args()

    # Define transformations (if any)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Initialize dataset
    dataset = CATEGORITZACIO(args, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    plot_label_distribution_with_missing(dataset)

    # Iterate over data
    """
    for images, labels in dataloader:
        print(f"Images: {images.shape}")
        print(f"Labels: {labels}")
    """