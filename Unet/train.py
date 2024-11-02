import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
from tqdm import tqdm
from scipy.io import savemat
import sys
sys.path.append('/ari/users/oeren/SegClass/Training/Unet') #  /app/UNET/Unet   /home/oeren/Documents/Unet /ari/users/oeren/Unet
from utils import DiceLoss
from utils_octa import dice_coef
from dataset import Synapse_dataset
from networks.network import MultiUNetModel
import pandas as pd
import h5py

class Segmentation:
    def __init__(self, data_path=None):
        super(Segmentation, self).__init__()
        self.data = None
        if data_path:
            self.data = np.load(data_path)

    def extract_data(self, csv_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Extract the necessary columns from the DataFrame
        ids_to_match = df['ID_nid'].values

        # Extract the data from the .npz file
        images = self.data['bscan']
        labels = self.data['mask']
        ids = self.data['id']

        # Find the indices of the images and labels that match the IDs in the CSV file
        matching_indices = [i for i, id_val in enumerate(ids) if id_val in ids_to_match]

        # Extract the matching images and labels
        matched_images = images[matching_indices]
        matched_labels = labels[matching_indices]
        matched_ids = ids[matching_indices]

        return matched_images, matched_labels, matched_ids

    
    def trainSeg(self, train_path, test_path, save_path, fold_n):
        
        
        train_images, train_masks, train_ids = self.extract_data(train_path)
        test_images, test_masks, test_ids = self.extract_data(test_path)
        print("Data is extracted.")
                
        # Check shapes of the loaded data
        print(f"Images shape: {train_images.shape}")
        print(f"Masks shape: {train_masks.shape}")
        print(f"Disease labels shape: {train_ids.shape}")
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Directory to save model checkpoints
        checkpoint_dir = os.path.join(save_path, f'unet')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_test_scores = []
        
        # Number of epochs
        num_epochs = 150
        batch_size = 24
        args_seed = 42  # Replace with your desired seed value
        num_classes = 6
        dice_loss = DiceLoss(num_classes)
        
        all_test_scores = []
        
        # Log file paths
        train_log_path = os.path.join(checkpoint_dir, f'training_log_fold{fold_n}.txt')
        test_log_path = os.path.join(checkpoint_dir, f'testing_log_fold{fold_n}.txt')
        
        print("Datasets has been creating.")
        # Load the datasets
        db_train = Synapse_dataset(train_images, train_masks, train_ids, mode='train')
        db_test = Synapse_dataset(test_images, test_masks, test_ids, mode='val')
        
        print("DataLoaaders has been creating.")
        train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False)
        
        #print the shapes of the datasets
        print(f"Training set: {len(db_train)}")
        print(f"Test set: {len(db_test)}")
        
        # Define the model
        model = MultiUNetModel(n_classes=6, IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CHANNELS=3)
        model = model.to(device)
        # Define loss function and optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        

        print("Training has started.")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
        
            for batch in train_progress_bar:
                images_batch = batch['image'].to(device)
                labels_batch = batch['label'].to(device)
                id_num = batch["id_num"].to(device)
                # Forward pass
                outputs = model(images_batch)
                loss_dice = dice_loss(outputs, labels_batch, softmax=True)
                loss = loss_dice
        
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                train_loss += loss.item()
                train_progress_bar.set_postfix({'Train Loss': loss.item()})
        
            # Calculate average training loss
            train_loss = train_loss / len(train_loader)
        
            # Log training loss
            train_log_message = f'Epoch {epoch+1}, Train Loss: {train_loss}\n'
            print(train_log_message)
            with open(train_log_path, 'a') as train_log:
                train_log.write(train_log_message)


            model_path = os.path.join(checkpoint_dir, f'best_model_fold_{fold_n}.pth')
            print(f'Model is updated')
            torch.save(model.state_dict(), model_path)

                    
        # Final evaluation on the test set at the end of training
        model.eval()
        
        all_images = []
        all_true_masks = []
        all_pred_masks = []
        all_dice_scores_octa = []
        predictions = []
        test_ids_labels = []
        
        # Directory to save test results
        test_results_dir = os.path.join(checkpoint_dir, f'test_results')
        os.makedirs(test_results_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Final Testing'):
                h, w = batch["image"].size()[2:]
                image_batch, true_mask, id_num = batch["image"], batch["label"], batch["id_num"]
                image_batch, true_mask = image_batch.to(device), true_mask.to(device)
                outputs = model(image_batch)
        
                _, predicted_class = torch.max(outputs, dim=1)
                one_hot_predictions = F.one_hot(predicted_class, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
                true_mask = F.one_hot(true_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
                dice_scores = dice_coef(true_mask, one_hot_predictions, num_classes)
                all_dice_scores_octa.append(dice_scores)
                # Append images, true masks, predicted masks, and disease labels to lists
                for i in range(image_batch.size(0)):
                    all_images.append(image_batch[i].cpu().numpy().transpose((1, 2, 0)))
                    all_true_masks.append(true_mask[i].cpu().numpy().argmax(axis=0))
                    all_pred_masks.append(predicted_class[i].cpu().numpy())
                    predictions.append(outputs[i].cpu().numpy())
                    test_ids_labels.append(id_num[i])
        
            # Save all images, true masks, predicted masks, and disease labels after processing the entire test set
            hdf5_file_path = os.path.join(test_results_dir, f'results_fold_{fold_n}.h5')
            with h5py.File(hdf5_file_path, 'w') as hf:
                hf.create_dataset('images', data=np.array(all_images))
                hf.create_dataset('true_masks', data=np.array(all_true_masks))
                hf.create_dataset('predicted_masks', data=np.array(all_pred_masks))
                #hf.create_dataset('raw_results', data=np.array(predictions))
                hf.create_dataset('ids', data=np.array(test_ids_labels))
        
            # Calculate the average Dice coefficient
            all_dice_scores_oct = np.array(all_dice_scores_octa)
            avg_dice_per_layer_octa = np.mean(all_dice_scores_oct, axis=0)
            final_avg_dice_score = np.mean(avg_dice_per_layer_octa[:-1])
            print(f'Final Average Dice Score: {final_avg_dice_score}')
        
            # Log final average Dice score to the test log
            test_log_message = f'Dice Scores: {avg_dice_per_layer_octa}\n'
            print(test_log_message)
            with open(test_log_path, 'a') as test_log:
                test_log.write(test_log_message)
                
    def inference(self, model_path, images):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 6

        # Load the pre-trained model
        model = MultiUNetModel(n_classes=num_classes, IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CHANNELS=3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # Prepare images
        images = torch.tensor(images, dtype=torch.float32)
        images = images.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        images = images.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(images)
            _, predicted_class = torch.max(outputs, dim=1)
            predicted_masks = F.one_hot(predicted_class, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Convert to numpy arrays
        outputs = outputs.cpu().numpy()
        predicted_class = predicted_class.cpu().numpy()

        return outputs, predicted_class
    
    def inference_results(self, test_path, model_path):
        test_images, test_masks, test_ids = self.extract_data(test_path)

        
        db_test = Synapse_dataset(test_images, test_masks, test_ids, mode='val')
        
        test_loader = DataLoader(db_test, batch_size=16, shuffle=False)
        # Final evaluation on the test set at the end of training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 6

        # Load the pre-trained model
        model = MultiUNetModel(n_classes=num_classes, IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CHANNELS=3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        all_images = []
        all_true_masks = []
        all_pred_masks = []
        all_dice_scores_octa = []
        predictions = []
        test_ids_labels = []
        test_log_path = '/ari/users/oeren/SegClass/Training/Results1/resultsA/unet/testing_log_fold14.txt'
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Final Testing'):
                h, w = batch["image"].size()[2:]
                image_batch, true_mask, id_num = batch["image"], batch["label"], batch["id_num"]
                image_batch, true_mask = image_batch.to(device), true_mask.to(device)
                outputs = model(image_batch)
        
                _, predicted_class = torch.max(outputs, dim=1)
                one_hot_predictions = F.one_hot(predicted_class, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
                true_mask = F.one_hot(true_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
                dice_scores = dice_coef(true_mask, one_hot_predictions, num_classes)
                all_dice_scores_octa.append(dice_scores)

            # Calculate the average Dice coefficient
            all_dice_scores_oct = np.array(all_dice_scores_octa)
            avg_dice_per_layer_octa = np.mean(all_dice_scores_oct, axis=0)
            final_avg_dice_score = np.mean(avg_dice_per_layer_octa[:-1])
            print(f'Final Average Dice Score: {final_avg_dice_score}')
        
            # Log final average Dice score to the test log
            test_log_message = f'Dice Scores: {avg_dice_per_layer_octa}\n'
            print(test_log_message)
            with open(test_log_path, 'a') as test_log:
                test_log.write(test_log_message)