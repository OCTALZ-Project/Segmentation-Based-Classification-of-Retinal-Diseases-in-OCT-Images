import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2

import random
import albumentations as A
import matplotlib as plt

from torchvision import transforms
from PIL import Image
import random
from skimage.util import random_noise
import matplotlib as plt

import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2

import random
import albumentations as A
import matplotlib as plt

from torchvision import transforms
from PIL import Image
import random
from skimage.util import random_noise
import matplotlib as plt

class RandomShift:
    """Randomly shift the image and masks by a small number of pixels."""
    def __init__(self, max_shift=5, probability=0.5):
        self.max_shift = max_shift
        self.probability = probability

    def __call__(self, image, masks):
        if random.random() < self.probability:
            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)
            image = Image.fromarray(ndimage.shift(np.array(image), (shift_y, shift_x, 0)))
            masks = [Image.fromarray(ndimage.shift(np.array(mask), (shift_y, shift_x))) for mask in masks]
        return image, masks

class RandomRotation:
    """Randomly rotate the image and masks by a small angle."""
    def __init__(self, max_angle=15, probability=0.5):
        self.max_angle = max_angle
        self.probability = probability

    def __call__(self, image, masks):
        if random.random() < self.probability:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image = image.rotate(angle)
            masks = [mask.rotate(angle) for mask in masks]
        return image, masks

class RandomZoom:
    """Randomly zoom into the image and masks."""
    def __init__(self, min_zoom=0.9, max_zoom=1.1, probability=0.5):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.probability = probability

    def __call__(self, image, masks):
        if random.random() < self.probability:
            zoom_factor = random.uniform(self.min_zoom, self.max_zoom)
            w, h = image.size
            x1, y1 = int(w * (1 - zoom_factor) / 2), int(h * (1 - zoom_factor) / 2)
            x2, y2 = int(w * (1 + zoom_factor) / 2), int(h * (1 + zoom_factor) / 2)
            image = image.crop((x1, y1, x2, y2)).resize((w, h), Image.LANCZOS)
            masks = [mask.crop((x1, y1, x2, y2)).resize((w, h), Image.NEAREST) for mask in masks]
        return image, masks


class DualCompose:
    """Apply a list of transformations to both the image and the masks."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, masks):
        for t in self.transforms:
            image, masks = t(image, masks)
        return image, masks

class RandomHorizontalFlip:
    """Randomly horizontally flips the Image and Masks with the probability *p*"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, masks):
        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT), [mask.transpose(Image.FLIP_LEFT_RIGHT) for mask in masks]
        return image, masks

class RandomVerticalFlip:
    """Randomly vertically flips the Image and Masks with the probability *p*"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, masks):
        if random.random() < self.p:
            return image.transpose(Image.FLIP_TOP_BOTTOM), [mask.transpose(Image.FLIP_TOP_BOTTOM) for mask in masks]
        return image, masks

class AddGaussianNoise:
    """Add Gaussian noise to the image only with a given probability. Ensures uniform noise across all channels for grayscale images."""
    def __init__(self, mean=0., std=0.12, probability=0.5):
        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, image, masks):
        if random.random() < self.probability:
            np_image = np.array(image).astype(np.float32)
    
            # Generate noise once and replicate it across all channels if needed
            if np_image.shape[-1] == 3 and np.all(np_image[:,:,0] == np_image[:,:,1]) and np.all(np_image[:,:,1] == np_image[:,:,2]):
                single_channel_noise = np.random.normal(self.mean, self.std * 255, np_image[:,:,0].shape).astype(np.float32)
                noise = np.stack([single_channel_noise]*3, axis=-1)  # Replicate the noise across all channels
            else:
                # Apply independent noise to each channel
                noise = np.random.normal(self.mean, self.std * 255, np_image.shape).astype(np.float32)

            np_image += noise
            np_image = np.clip(np_image, 0, 255)  # Clip to ensure valid image values
            image = Image.fromarray(np_image.astype(np.uint8))

        return image, masks
class AddVerticalLines:
    """Add vertical lines to the image, averaging the previous and next vertical slices."""
    def __init__(self, probability=0.5, num_lines=20):
        self.probability = probability
        self.num_lines = num_lines

    def __call__(self, image, masks):
        if random.random() < self.probability:
            np_image = np.array(image)
            height, width, channels = np_image.shape
            line_positions = random.sample(range(1, width - 1), self.num_lines)  # Randomly select positions for the lines
            for x in line_positions:
                np_image[:, x, :] = (np_image[:, x - 1, :] + np_image[:, x + 1, :]) / 2
            image = Image.fromarray(np_image.astype(np.uint8))
        return image, masks

class AddSaltPepperNoise:
    """Add salt and pepper noise to the image only, maintaining the grayscale nature across all channels."""
    def __init__(self, amount=0.025, probability=0.5):
        self.amount = amount
        self.probability = probability
        self.gray_value = 128
    """
    def __call__(self, image, masks):
        if random.random() < self.probability:
            np_image = np.array(image).astype(np.float32)  # Convert to float32
            # Apply noise directly on the 3-channel grayscale image
            noisy_img = random_noise(np_image, mode='s&p', amount=self.amount, clip=True)
            noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)  # Scale back to uint8
            return Image.fromarray(noisy_img), masks  # Ensure the mode is correctly set to 'RGB'
        return image, masks
    """
    def __call__(self, image, masks):
        if random.random() < self.probability:
            np_image = np.array(image).astype(np.float32) 
            row, col, c = np_image.shape
            num_noise = int(self.amount * row * col)
            
            # Number of salt vs pepper points
            num_salt = int(num_noise)
            num_pepper = num_noise - num_salt
            
            # Add gray noise
            coords = [np.random.randint(0, i - 1, num_noise) for i in (row, col)]
            np_image[coords[0], coords[1], :] = [self.gray_value, self.gray_value, self.gray_value]  # Set to gray value

            # Add salt noise
            #coords_salt = [np.random.randint(0, i - 1, num_salt) for i in (row, col)]
            #np_image[coords_salt[0], coords_salt[1], :] = [255,255,255]  # Assuming image is in the range [0, 255]
            
            # Add pepper noise
            #coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in (row, col)]
            #np_image[coords_pepper[0], coords_pepper[1], :] = [0,0,0]
            np_image = np_image.astype(np.uint8)
            return Image.fromarray(np_image), masks
        return image, masks

class Synapse_dataset(Dataset):
    def __init__(self, images, masks, ids, mode = 'train'):
        super().__init__()
        self.mode = mode
        # Load data from .npz file
        self.images = images
        self.labels = masks
        self.id_num = ids

        if self.mode == 'train':
            augmented_images, augmented_labels = self.augment_data_train(self.images, self.labels)
            indices = np.arange(len(augmented_images))
            np.random.shuffle(indices)
            self.augmented_images = augmented_images[indices]
            self.augmented_labels = augmented_labels[indices]
        elif self.mode == 'test':
            augmented_images, augmented_labels = self.augment_data_test(self.images, self.labels)
            indices = np.arange(len(augmented_images))
            self.augmented_images = augmented_images[indices]
            self.augmented_labels = augmented_labels[indices]
        else:
            self.augmented_images = self.images
            self.augmented_labels = self.labels

        
    def augmentation(self, images, masks, transform, augment_times=19):
        """Augment images and masks using the provided transformation and include originals."""
        augmented_images = []  # Initialize empty list for storing images
        augmented_masks = []   # Initialize empty list for storing masks
    
        if augment_times > 1:
        # Add original images and masks first
            for img, mask in zip(images, masks):
                augmented_images.append(img)
                augmented_masks.append(mask)
        
        # Perform augmentation
        for _ in range(augment_times):
            for img, mask in zip(images, masks):
                pil_img = Image.fromarray(img.astype(np.uint8))  # Convert numpy array to PIL Image
                pil_masks = [Image.fromarray(m.astype(np.uint8)) for m in mask]  # Convert each mask channel to PIL Image
                transformed_img, transformed_masks = transform(pil_img, pil_masks)  # Apply transformation
                np_img = np.array(transformed_img)  # Convert back to numpy array
                np_masks = np.stack([np.array(m) for m in transformed_masks], axis=0)  # Stack masks along new axis
                augmented_images.append(np_img)
                augmented_masks.append(np_masks)
    
        return np.array(augmented_images), np.array(augmented_masks)
    
    def augment_data_test(self, images, masks):
        # Compose the transformations for test data
        transform = DualCompose([
            RandomShift(max_shift=30, probability=0.6),
            RandomRotation(max_angle=10, probability=0.6),
            RandomZoom(min_zoom=0.85, max_zoom=1.1, probability=0.6),
        ])

        # Apply transformations without increasing the number of samples
        augmented_images, augmented_masks = self.augmentation(images, masks, transform, augment_times=1)
        return augmented_images, augmented_masks
   
    def augment_data_train(self, images, masks):
        # Compose the transformations with probabilities
        transform = DualCompose([
            #RandomHorizontalFlip(),
            #RandomVerticalFlip(),
            AddGaussianNoise(std=0.12, probability=0.4),  # 70% chance to apply Gaussian noise
            AddSaltPepperNoise(amount=0.025, probability=0.4),  # 70% chance to apply salt and pepper noise
            AddVerticalLines(probability=0.4, num_lines=20),  # 70% chance to add vertical lines
        ])

        # Augment data, including originals
        augmented_images, augmented_masks = self.augmentation(images, masks, transform, augment_times=1)
        return augmented_images, augmented_masks

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Processing the image
        #get as item
        image = self.augmented_images[idx]
        label = self.augmented_labels[idx]
        ids = self.id_num[idx]
        # create dict
        sample = {'image': image, 'label': label, "id_num": ids}
        
        # continue with transformed data    
        image, label, ids = sample['image'], sample['label'], sample['id_num']

        # covert appropriate shape
        image = image.transpose((2, 0, 1)) 
        image = torch.from_numpy(image.astype('float32'))

        label = label.transpose((0, 1, 2))
        label = torch.from_numpy(label.astype('float32'))
        
        label = torch.argmax(label, dim=0)
        
        return {'image': image, 'label': label, 'id_num': ids}

    def __len__(self):
        return len(self.augmented_images)