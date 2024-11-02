import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
import pandas as pd
import h5py
import sys
# Set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Custom shift augmentation function
def shift_image(image, direction='up', shift_amount=0, fill_value=0):
    """
    Shifts the image up or down by shift_amount and fills new areas with fill_value.
    """
    h, w, c = image.shape
    if direction == 'up':
        shifted_image = np.full_like(image, fill_value)
        shifted_image[:-shift_amount, :] = image[shift_amount:, :]
    elif direction == 'down':
        shifted_image = np.full_like(image, fill_value)
        shifted_image[shift_amount:, :] = image[:-shift_amount, :]
    else:
        raise ValueError("Direction should be 'up' or 'down'")
    return shifted_image

class ShiftImage(A.ImageOnlyTransform):
    def __init__(self, shift_amount=10, direction='up', always_apply=False, p=1.0):
        super(ShiftImage, self).__init__(always_apply, p)
        self.shift_amount = shift_amount
        self.direction = direction

    def apply(self, img, **params):
        return shift_image(img, direction=self.direction, shift_amount=self.shift_amount)


# Custom boundary shifting function
def shift_boundaries(mask):
    mask = mask.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i + 1 < mask.shape[0] and not np.array_equal(mask[i + 1, j], mask[i, j]):
                shift_amount = np.random.randint(3, 7)
                direction = np.random.choice([-1, 1])
                color = mask[i, j] if direction == -1 else mask[i + 1, j]
                if not np.array_equal(color, [255, 165, 0]):
                    if 0 <= i + shift_amount * direction < mask.shape[0]:
                        if i + shift_amount * direction >= i:
                            mask[i : i + shift_amount * direction, j] = color
                        else:
                            mask[i + shift_amount * direction : i, j] = color
    return mask

class ShiftBoundaries(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ShiftBoundaries, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return shift_boundaries(img)
        
# Define the augmentation pipeline
augmentation_pipeline = A.Compose([
    A.CoarseDropout(max_holes=2, max_height=20, max_width=20, min_holes=1, min_height=6, min_width=6, fill_value=5, p=1.0),
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.HorizontalFlip(p=0.8),
    A.RandomRotate90(p=1.0),
    #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.6, border_mode=cv2.BORDER_CONSTANT, value=5),
    #A.RandomCrop(height=224, width=224, p=0.6),
    #A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.6),
    ToTensorV2()
])

# Define the individual augmentation pipelines
flip_pipeline = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

rotate_pipeline = A.Compose([
    A.RandomRotate90(p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

dropout_pipeline = A.Compose([
    A.CoarseDropout(max_holes=2, max_height=20, max_width=20, min_holes=1, min_height=6, min_width=6, fill_value=0, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

shift_up_pipeline = A.Compose([
    ShiftImage(shift_amount=10, direction='up', p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

shift_down_pipeline = A.Compose([
    ShiftImage(shift_amount=10, direction='down', p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

boundary_shift_pipeline = A.Compose([
    ShiftBoundaries(p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Normalize and convert to tensor (for original images without augmentation)
transform_to_tensor = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def compute_entropy(probabilities):
    """
    Compute the entropy for a set of probabilities.
    
    Parameters:
    - probabilities: a 1D array of probabilities for each class.
    
    Returns:
    - entropy: the entropy value.
    """
    return -np.sum(probabilities * np.log(probabilities + 1e-9))

def augment_probability_maps(raw_results):
    N, C, H, W = raw_results.shape
    augmented_results = np.zeros_like(raw_results)
    
    # Ensure probabilities are non-negative and normalize them
    epsilon = 1e-8  # Small value to ensure no probabilities are zero
    raw_results = np.clip(raw_results, epsilon, None)  # Ensure all values are at least epsilon
    raw_results /= np.sum(raw_results, axis=1, keepdims=True)  # Normalize probabilities
    
    for i in range(N):
        for h in range(H):
            for w in range(W):
                # Get the probability distribution for the current pixel
                probabilities = raw_results[i, :, h, w]
                max_prob = np.max(probabilities)
                max_prob_index = np.argmax(probabilities)
                
                if max_prob > 0.5:
                    # Keep the class with the highest probability
                    new_class = max_prob_index
                else:
                    # Exclude the class with the highest probability
                    excluded_probabilities = np.delete(probabilities, max_prob_index)
                    excluded_classes = np.delete(np.arange(C), max_prob_index)
                    # Normalize the remaining probabilities
                    excluded_probabilities /= np.sum(excluded_probabilities)
                    # Sample a new class based on the remaining probabilities
                    new_class = np.random.choice(excluded_classes, p=excluded_probabilities)
                
                # Set the new class in the one-hot encoded format
                augmented_results[i, :, h, w] = 0
                augmented_results[i, new_class, h, w] = 1
    
    return augmented_results

def extract_contour(segmentation_image):
    #print(segmentation_image.shape)
    height, width = segmentation_image.shape
    contour = np.zeros(width, dtype=int)
    
    for x in range(width):
        for y in range(height - 1, 0, -1):
            if segmentation_image[y, x] != segmentation_image[y - 1, x]:
                contour[x] = y
                break    
        """    
        for y in range(height - 1):
            if segmentation_image[y, x] == 4 and segmentation_image[y + 1, x] == 5:
                contour[x] = y
                break
        """
    return contour
    
def flatten_image(image, contour, target_y):
    height, width = image.shape
    # create an image with the same shape as the input image with all 5s
    flattened_image = np.zeros_like(image)
    flattened_image.fill(5)


    #flattened_image = np.zeros_like(image)
    
    for x in range(width):
        shift = target_y - contour[x]
        for y in range(height):
            new_y = y + shift
            if 0 <= new_y < height:
                flattened_image[new_y, x] = image[y, x]
    
    return flattened_image

def flatten_contour(contour, contour_base, target_y):
    flattened_contour = np.zeros_like(contour)

    for x in range(len(contour)):
        shift = target_y - contour_base[x]
        new_y = contour[x] + shift
        flattened_contour[x] = new_y
    
    return flattened_contour

class ClassifierDataset(Dataset):
    def __init__(self, bscan, vscan, path, weight_pathB, weight_pathV, mode):
        self.dataB = bscan
        self.dataV = vscan
        self.bscans, self.vscans, self.patIds, self.labels, self.num_sli = self.extract_data(path)
        self.rawsegresultsB, self.masksB = segmentation.inference(weight_pathB, self.bscans)
        self.rawsegresultsV, self.masksV = segmentation.inference(weight_pathV, self.vscans)
        #self.masks = np.stack((self.masks,)*3, axis=-1)
        #self.rawsegresults = augment_probability_maps(self.rawsegresults)
        #self.masks = np.argmax(self.rawsegresults, axis=1)
        #self.fusionMasks = self.fusion(self.masks)
        
        
        #self.masks = apply_unet(self.bscans)
        self.mode = mode
        self.transform = augmentation_pipeline
        self.transform_to_tensor = transform_to_tensor
        self.colormap = {
            0: [255, 165, 0],    # Orange
            1: [255, 0, 0],      # Red
            2: [0, 255, 0],      # Green
            3: [0, 0, 255],      # Blue
            4: [255, 255, 0],    # Yellow
            5: [255, 255, 255]         # Black
        }
        # Augment the dataset
        self.augmented_imagesB, self.augmented_imagesV, self.augmented_labels, self.augmentedIds = self.augment_dataset()

    def __len__(self):
        return len(self.augmented_imagesB)
        
    def extract_data(self, csv_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
    
        # Extract the necessary columns from the DataFrame
        ids_to_match = df['ID_nid'].values
        diseases_from_csv = df['Disease'].values
    
        # Extract the data from the .npz file
        bscans = self.dataB['bscan']        
        ids = self.dataB['id']
        num_sli = self.dataB['slicen']
        #masks = self.dataB['mask']
        #masks = np.argmax(masks, axis=1)  
        #-------------Vertical Slices----------------
        vscans = self.dataV['bscan']
        idsV = self.dataV['id']
        num_sliV = self.dataV['slicen']
        
        
        # Find the indices of the images and labels that match the IDs in the CSV file
        matching_indices = [i for i, id_val in enumerate(ids) if id_val in ids_to_match]
        matching_indicesV = [i for i, id_val in enumerate(idsV) if id_val in ids_to_match]
        
        # Filter based on num_sli values being 200 or 152
        valid_indices = [i for i in matching_indices if num_sli[i] in [200, 152]]
        valid_indicesV = [i for i in matching_indicesV if num_sliV[i] in [200, 152]]
        
        # Extract the matching images, ids, and labels
        bscans = bscans[valid_indices]
        patIds = ids[valid_indices]
        num_sli = num_sli[valid_indices]
        #masks = masks[valid_indices]
        
        vscans = vscans[valid_indicesV]
        patIdsV = idsV[valid_indicesV]
        num_sliV = num_sliV[valid_indicesV]
        
        print(np.array(patIdsV).shape, np.array(patIds).shape)
        if (patIdsV != patIds).any():
            print("patIdsV patIds ile eşleşmiyor, program durduruluyor...")
            sys.exit(1)  # Stop the program with error code 1
        else:
            print("CONTINUEEEEEEEEEEEEEEEEEE")
        # Extract the corresponding diseases from the CSV file
        diseases = [diseases_from_csv[ids_to_match.tolist().index(id_val)] for id_val in patIds]
    
        return bscans, vscans, patIds, diseases, num_sli, 
    
    def fusion(self, scan, mask):
        fusion_masks = np.zeros_like(scan)
        #print("fusion_masks", fusion_masks.shape)
        for ind, vals in enumerate([[2], [0, 3], [1, 4], [5]]):
            maskF = np.zeros_like(mask, dtype=np.uint8)
            for val in vals:
                maskF[mask == val] = 1
            if ind != 3:
                if ind < fusion_masks.shape[2]:  # Check if the index is within bounds
                    #print(f"Applying mask to channel {ind}")
                    fusion_masks[:, :, ind] = np.where(maskF == 1, scan[:, :, ind], fusion_masks[:, :, ind])
                    #print(f"Updated fusion_masks at index {ind}")
                else:
                    print(f"Warning: Index {ind} is out of bounds for axis 2 with size {fusion_masks.shape[2]}")
            else:
                for c in range(fusion_masks.shape[2]):
                    fusion_masks[:, :, c] = np.where(maskF == 1, 255, fusion_masks[:, :, c])
                    #print(f"Set fusion_masks channel {c} to white where maskF is 1")

        return fusion_masks

    
    """
    def fusion(self, masks):
        fusion_masks = np.zeros_like(self.bscans)
        #print("fusion_masks", fusion_masks.shape)
        #print("mask", self.masks.shape)
        for ind, vals in enumerate([[0, 2], [1, 4], [3, 5]]):
            mask = np.zeros_like(masks, dtype=np.uint8)
            for val in vals:
                mask[masks == val] = 1
            if ind != 3:
                for i in range(fusion_masks.shape[0]):  # iterate over the batch
                    fusion_masks[i, :, :, ind][mask[i, :, :] == 1] = self.bscans[i, :, :, ind][mask[i, :, :] == 1]
            
            else:

                for i in range(fusion_masks.shape[0]):  # iterate over the batch
                    for c in range(fusion_masks.shape[3]):
                        fusion_masks[i, :, :, c][mask[i, :, :] == 1] = 255  # Set to white

            
        return fusion_masks
        """
        
    def apply_colormap(self, image):
        # Create an empty array with the shape of the label
        colored_image = np.zeros_like(image, dtype=np.uint8)
        for key, color in self.colormap.items():
            mask = np.all(image == key, axis=-1)
            colored_image[mask] = self.colormap[key]
        return colored_image

    def augment_dataset(self):
        augmented_imagesB = []
        augmented_imagesV = []
        augmented_labels = []
        augmentedIds = []

        #----FOR BSCANS---------------------
        rawsegresultsB = augment_probability_maps(self.rawsegresultsB)
        aug_masksB = np.argmax(rawsegresultsB, axis=1)
        
        #aug_masks = np.stack((aug_masks,)*3, axis=-1)
        #augfusion = self.fusion(aug_masks)
        
        for i in range(len(self.masksB)):
            
            bscan, maskb = self.bscans[i, :, :, 0].astype('float32'), self.masksB[i].astype('float32')
            
            contour = extract_contour(maskb)
            # Set the target y-position
            target_y = 150
        
            # Flatten the image based on the contour
            maskb = flatten_image(maskb, contour, target_y)
            #maskb = np.stack((maskb,)*3, axis=-1)
            
            bscan = flatten_image(bscan, contour, target_y)
            bscan = np.stack((bscan,)*3, axis=-1)
            
            image1 = self.fusion(bscan, maskb)
            #image1 = np.stack((image1,)*3, axis=-1)
            #print("before coloring", len(np.unique(image)))
            #image1 = self.apply_colormap(image1)
            #print("after coloring", len(np.unique(image)))
            label = self.labels[i].astype('float32')

            aug_image = self.transform_to_tensor(image=image1)['image']
            augmented_imagesB.append(aug_image)
            augmented_labels.append(torch.tensor(label))
            augmentedIds.append(self.patIds[i])
            
            augment_times = 1 if label == 0 else 1
            if self.mode == 'train':
                bscan2, maskb2 = self.bscans[i, :, :, 0].astype('float32'), aug_masksB[i].astype('float32')
                
                contour = extract_contour(maskb2)
                
                maskb2 = flatten_image(maskb2, contour, target_y)
                #maskb2 = np.stack((maskb2,)*3, axis=-1)
                
                bscan2 = flatten_image(bscan2, contour, target_y)
                bscan2 = np.stack((bscan2,)*3, axis=-1)
                
                image2 = self.fusion(bscan2, maskb2)
                #image2 = self.apply_colormap(image2)
                label = self.labels[i].astype('float32')
                aug_image2 = self.transform_to_tensor(image=image2)['image']
                augmented_imagesB.append(aug_image2)
                augmented_labels.append(torch.tensor(label))
                augmentedIds.append(self.patIds[i])
                

                for transform in [flip_pipeline]:
                    augmented = transform(image=image1)
                    #print(transform)
                    #print("before aug",  len(np.unique(image)))
                    aug_image = augmented['image']
                    #print("after aug", len(np.unique(aug_image)))
                    augmented_imagesB.append(aug_image)
                    augmented_labels.append(torch.tensor(label))
                    augmentedIds.append(self.patIds[i])

        #----FOR VSCANS---------------------
        rawsegresultsV = augment_probability_maps(self.rawsegresultsV)
        aug_masksV = np.argmax(rawsegresultsV, axis=1)  

        for i in range(len(self.masksV)):
            
            vscan, maskv = self.vscans[i, :, :, 0].astype('float32'), self.masksV[i].astype('float32')
            
            contour = extract_contour(maskv)
            # Set the target y-position
            target_y = 150
        
            # Flatten the image based on the contour
            maskv = flatten_image(maskv, contour, target_y)
            #maskv = np.stack((maskv,)*3, axis=-1)
            
            vscan = flatten_image(vscan, contour, target_y)
            vscan = np.stack((vscan,)*3, axis=-1)
            
            image1 = self.fusion(vscan, maskv)
            #image1 = np.stack((image1,)*3, axis=-1)
            #print("before coloring", len(np.unique(image)))
            #image1 = self.apply_colormap(image1)
            #print("after coloring", len(np.unique(image)))
            label = self.labels[i].astype('float32')

            aug_image = self.transform_to_tensor(image=image1)['image']
            augmented_imagesV.append(aug_image)
            augmented_labels.append(torch.tensor(label))
            augmentedIds.append(self.patIds[i])
            
            augment_times = 1 if label == 0 else 1
            if self.mode == 'train':
                vscan2, maskv2 = self.vscans[i, :, :, 0].astype('float32'), aug_masksV[i].astype('float32')
                
                contour = extract_contour(maskv2)
                
                maskv2 = flatten_image(maskv2, contour, target_y)
                #maskv2 = np.stack((maskv2,)*3, axis=-1)
                
                vscan2 = flatten_image(vscan2, contour, target_y)
                vscan2 = np.stack((vscan2,)*3, axis=-1)
                
                image2 = self.fusion(vscan2, maskv2)
                #image2 = self.apply_colormap(image2)
                label = self.labels[i].astype('float32')
                aug_image2 = self.transform_to_tensor(image=image2)['image']
                augmented_imagesV.append(aug_image2)
                augmented_labels.append(torch.tensor(label))
                augmentedIds.append(self.patIds[i])
                

                for transform in [flip_pipeline]:
                    augmented = transform(image=image1)
                    #print(transform)
                    #print("before aug",  len(np.unique(image)))
                    aug_image = augmented['image']
                    #print("after aug", len(np.unique(aug_image)))
                    augmented_imagesV.append(aug_image)

        if self.mode == 'train':
            np.savez('/ari/users/oeren/SegClass/Training/RecData', recB=augmented_imagesB, maskB=self.masksB, recV=augmented_imagesV, maskV=self.masksV)
        return augmented_imagesB, augmented_imagesV, augmented_labels, augmentedIds

    def __getitem__(self, idx):
        imageB = self.augmented_imagesB[idx]
        imageV = self.augmented_imagesV[idx]
        label = self.augmented_labels[idx]
        ids = self.augmentedIds[idx]

        return imageB, imageV, label, ids