import sys
sys.path.append('/ari/users/oeren/SegClass/Training/Classifier') 
from model import ResNetClassifier
from dataset import ClassifierDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import torch.nn.functional as F
  
class custom_categorical_cross_entropy_loss(nn.Module):
    def __init__(self, class_weights=None):
        super(custom_categorical_cross_entropy_loss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # If alpha is not set, it uses uniform weights for all classes
        self.weights = class_weights if class_weights is not None else torch.tensor([1.0]*3).to(self.device)
        
    def forward(self, output, target):
        """
        Compute the custom categorical cross-entropy loss with class weights.
        Args:
            output: Predictions from the model (logits).
            target: True labels.
            class_weights: Tensor of class weights.
        Returns:
            Loss value.
        """
        # Apply softmax to get probabilities
        pred_prob = F.softmax(output, dim=1)
        
        # Create one-hot encoding of target labels
        target_one_hot = F.one_hot(target, num_classes=output.size(1)).float()
        target = target.to(self.device) 
        self.weights = self.weights.to(self.device) 
        #print(f"target device: {target.device}")
        #print(f"Weights device: {self.weights.device}")
        # Compute the weighted loss
        weights = self.weights[target]
        loss = (-(pred_prob + 1e-5).log() * target_one_hot * weights.unsqueeze(1)).sum(dim=1).mean()
        
        return loss


class Classification:
    def __init__(self, data_path):
        super(Classification, self).__init__()
        self.data = np.load(data_path)
    
    
    def func_class_weights(self, train_path):
        # Load the CSV file
        df = pd.read_csv(train_path)
        
        # Extract the 'Disease' column
        train_labels = df['Disease'].values

        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        return class_weights, classes

    def trainClass(self, train_path, test_path, save_path, i, segmentation):
        # path os weights of best unet models 
        weight_path = os.path.join(save_path, f'unet/best_model_fold_{i}.pth')
        
        print("Datasets have been creating")
        # Create datasets
        train_dataset = ClassifierDataset(self.data, train_path, weight_path, segmentation, mode='train')
        test_dataset = ClassifierDataset(self.data, test_path, weight_path, segmentation, mode='test')

        print("DataLoaders have been creating")
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        print("Train dataset shape:", len(train_dataset))
        print("Test dataset shape:", len(test_dataset))

            
        # Compute class weights
        class_weights, unique_labels = self.func_class_weights(train_path)
        
        # Paths and file names
        checkpoint_dir = os.path.join(save_path, f'class')
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_log_path = os.path.join(checkpoint_dir, f'training_log_fold_{i}.txt')
        test_log_path = os.path.join(checkpoint_dir, f'testing_log_fold_{i}.txt')

        # Directory to save model checkpoints
        weights_dir = os.path.join(checkpoint_dir, f'weights')
        plots_dir = os.path.join(checkpoint_dir, f'plots')
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 3
        model = ResNetClassifier(num_classes=num_classes)
        model = model.to(device)
        criterion = custom_categorical_cross_entropy_loss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        class_weights = class_weights.to(device)
        num_epochs = 40
        all_train_losses = []
        all_test_losses = []
        all_test_acc = []
        all_precision = []
        all_recall = []
        all_f1 = []
        all_images = []
        print("Classification training has been started.")
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
            for batch in train_progress_bar:
                images, labels, ids = batch
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(images).to(device)                      
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_images.append(images.cpu().numpy())
                
            #scheduler.step()
            
            train_accuracy = 100 * correct / total
            train_loss = train_loss / len(train_loader)
            all_train_losses.append(train_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}, Accuracy: {train_accuracy}%')
            train_log_message = f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}\n'
            with open(train_log_path, 'a') as train_log:
                train_log.write(train_log_message)
                
            model.eval()
            test_loss = 0.0
            correct_ = 0
            total_ = 0
            true_diseases = []
            predicted_diseases = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Final Testing'):
                    images, labels, ids = batch
                    images = images.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.long)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_ += labels.size(0)
                    correct_ += (predicted == labels).sum().item()
                    for j in range(images.size(0)):
                        true_diseases.append(labels[j].cpu().numpy())
                        predicted_diseases.append(predicted[j].cpu().numpy())
            test_loss = test_loss / len(test_loader)
            all_test_losses.append(test_loss)
            test_accuracy = 100 * correct_ / total_
            all_test_acc.append(test_accuracy)
            precision, recall, f1, _ = precision_recall_fscore_support(true_diseases, predicted_diseases, average='weighted')
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%, Precision: {precision}, Recall: {recall}, F1: {f1}')
            test_log_message = f'Test Acc: {test_accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}\n'
            with open(test_log_path, 'a') as test_log:
                test_log.write(test_log_message)
        
        torch.save(model.state_dict(), os.path.join(weights_dir, f'best_model_fold_{i}.pth'))

        model.load_state_dict(torch.load(os.path.join(weights_dir, f'best_model_fold_{i}.pth')))
        model.eval()
        test_loss = 0.0
        correct_ = 0
        total_ = 0
        true_diseases = []
        predicted_diseases = []
        predicted_probs = []
        test_ids = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Final Testing'):
                images, labels, ids = batch
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_ += labels.size(0)
                correct_ += (predicted == labels).sum().item()
                probs = F.softmax(outputs, dim=1)
                for j in range(images.size(0)):
                    true_diseases.append(labels[j].cpu().numpy())
                    predicted_probs.append(probs[j].cpu().numpy())
                    predicted_diseases.append(predicted[j].cpu().numpy())
                    test_ids.append(ids[j])
        test_accuracy = 100 * correct_ / total_
        print(f'Final Test Loss: {test_loss/len(test_loader)}, Final Test Accuracy: {test_accuracy}%')
        test_log_message = f'Final Test Acc: {test_accuracy}\n'
        with open(test_log_path, 'a') as test_log:
            test_log.write(test_log_message)

        with h5py.File(os.path.join(checkpoint_dir, f'predicted_masks_fold_{i}.h5'), 'w') as hf:
            hf.create_dataset('predicted_masks', data=np.array(predicted_probs))
            hf.create_dataset('true_diseases', data=np.array(true_diseases))
            hf.create_dataset('predicted_diseases', data=np.array(predicted_diseases))
            hf.create_dataset('ids', data=np.array(test_ids))
    
        labels = ["Normal", "DR", "AMD"]

        cm = confusion_matrix(true_diseases, predicted_diseases)
        print('Confusion Matrix:\n', cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_fold_{i}.png'))
    
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(all_train_losses, label='Training Loss')
        plt.plot(all_test_losses, label='Test Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'training_validation_loss_plot_fold_{i}.png'))

        scaled_test_acc = [acc / 100 for acc in all_test_acc]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.plot(all_train_losses, label='Training Loss')
        plt.plot(all_test_losses, label='Test Loss', linestyle='--')
        plt.plot(scaled_test_acc, label='Test Accuracy', linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('Loss and Accuracy')
        plt.title('Training and Validation Loss and Test Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'training_validation_loss_accuracy_plot_fold_{i}.png'))
        
        # Calculate ROC curve and AUC
        true_diseases = np.array(true_diseases)
        predicted_probs = np.array(predicted_probs)
        if predicted_probs.shape[1] > 1:
            fpr, tpr, _ = roc_curve(true_diseases, predicted_probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plots_dir, f'roc_curve_fold_{i}.png'))
        else:
            print("Not enough classes to calculate ROC curve.")

        print('Training and testing completed successfully!')