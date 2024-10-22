import os
from torch.utils.data import Dataset
import presets
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np

class DATES(Dataset):
    """Date estimation in the wild dataset"""

    def __init__(self, args, split='train'):
        self.root_dir = os.path.join(args.data_path, 'Date_Estimation_in_the_Wild')
        self.split = split
        self.classes = args.classes
        self.min_year = args.min_year
        self.filenames, self.labels = self.load_annotations(os.path.join(self.root_dir, f'gt_{split}_ok.csv'))


        if self.split == 'train':
            self.transforms = presets.ClassificationPresetTrain(
                crop_size=args.train_crop_size,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                interpolation=InterpolationMode(args.interpolation),
                auto_augment_policy=getattr(args, "auto_augment", None),
                random_erase_prob=getattr(args, "random_erase", 0.0),
                ra_magnitude=getattr(args, "ra_magnitude", None),
                augmix_severity=getattr(args, "augmix_severity", None),
            )
        else:
            self.transforms = presets.ClassificationPresetEval(
                crop_size=args.val_crop_size,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                resize_size=args.val_resize_size,
                interpolation=InterpolationMode(args.interpolation)
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.filenames[idx][0], self.filenames[idx][1:3], f'{self.filenames[idx]}.jpg')
        image = Image.open(img_name).convert('RGB')
        image = self.transforms(image)

        labels = np.array(int(self.labels[idx] / 5) - int(self.min_year/5))

        return np.array(image), labels

    def load_annotations(self, path):
        """
        Load the annotations from the csv file.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line[:-1] for line in lines]

        filenames = [line.split(',')[1] for line in lines]
        labels = [int(line.split(',')[0]) for line in lines]

        return filenames, labels
