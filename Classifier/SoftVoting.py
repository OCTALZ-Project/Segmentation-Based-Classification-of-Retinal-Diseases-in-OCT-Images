import sys
sys.path.append('/ari/users/oeren/SegClass/Training/Classifier')
from model import ResNetClassifier
from datasetVot import ClassifierDataset
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
        self.weights = class_weights if class_weights is not None else torch.tensor([1.0]*3).to(self.device)
        
    def forward(self, output, target):
        pred_prob = F.softmax(output, dim=1)
        target_one_hot = F.one_hot(target, num_classes=output.size(1)).float()
        target = target.to(self.device) 
        self.weights = self.weights.to(self.device) 

        weights = self.weights[target]
        loss = (-(pred_prob + 1e-5).log() * target_one_hot * weights.unsqueeze(1)).sum(dim=1).mean()
        
        return loss

class Classification:
    def __init__(self, data_hscan, data_vscan):
        super(Classification, self).__init__()
        self.hscan = np.load(data_hscan)
        self.vscan = np.load(data_vscan)
    
        
    def func_class_weights(self, train_path):
        df = pd.read_csv(train_path)
        train_labels = df['Disease'].values

        classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        return class_weights, classes
    
    def trainClass(self, train_path, test_path, save_path, i, segmentation):
        weight_pathB = f'/ari/users/oeren/SegClass/Training/Results1/results5/unet/best_model_fold_{i}.pth'
        weight_pathV = f'/ari/users/oeren/SegClass/Training/Results1/resultsA/unet/best_model_fold_{i}.pth'
        print("Test dataset has been creating")
        test_dataset = ClassifierDataset(self.hscan, self.vscan, test_path, weight_pathB, weight_pathV, mode='test')

        print("DataLoader has been creating")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        print("Test dataset shape:", len(test_dataset))

        class_weights, unique_labels = self.func_class_weights(train_path)
        
        checkpoint_dir = os.path.join(save_path, f'class')
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_log_path = os.path.join(checkpoint_dir, f'training_log_fold_{i}.txt')
        test_log_path = os.path.join(checkpoint_dir, f'testing_log_fold_{i}.txt')
        weights_dir = os.path.join(checkpoint_dir, f'weights')
        plots_dir = os.path.join(checkpoint_dir, f'plots')
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 3
        model_hbscan = ResNetClassifier(num_classes=num_classes)
        model_vbscan = ResNetClassifier(num_classes=num_classes)
        model_hbscan = model_hbscan.to(device)
        model_vbscan = model_vbscan.to(device)
        criterion = custom_categorical_cross_entropy_loss(class_weights)
        optimizer_bscan = optim.SGD(model_bscan.parameters(), lr=0.0001)
        optimizer_slice = optim.SGD(model_slice.parameters(), lr=0.0001)
        num_epochs = 150

        print("Ensemble Approach has been started.")
        model_hbscan.load_state_dict(torch.load(f'/ari/users/oeren/SegClass/Training/Experiments/Bscan/result40/class/weights/best_model_fold_{i}.pth'))
        model_vbscan.load_state_dict(torch.load(f'/ari/users/oeren/SegClass/Training/Experiments/Vscan/result40/class/weights/best_model_fold_{i}.pth'))
        model_bscan.eval()
        model_slice.eval()
        test_loss = 0.0
        correct_ = 0
        total_ = 0
        
        true_diseases = []
        predicted_diseases = []
        predicted_probs_list = []  # Store probabilities here
        predicted_masksB = []
        predicted_masksV = []
        test_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Final Testing'):
                images_bscan, images_slice, labels, ids = batch
                images_bscan = images_bscan.to(device, dtype=torch.float)
                images_slice = images_slice.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                outputs_hbscan = model_hbscan(images_bscan)
                outputs_vbscan = model_vbscan(images_slice)
                outputs = (outputs_hbscan + outputs_vbscan) / 2.0  # Combine the outputs
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_ += labels.size(0)
                correct_ += (predicted == labels).sum().item()

                # Store probabilities for ROC curve calculation
                predicted_probs_list.extend(F.softmax(outputs, dim=1).cpu().numpy())

                for j in range(images_bscan.size(0)):
                    true_diseases.append(labels[j].cpu().numpy())
                    predicted_diseases.append(predicted[j].cpu().numpy())
                    predicted_masksB.append(images_bscan[j].cpu().numpy())
                    predicted_masksV.append(images_slice[j].cpu().numpy())
                    test_ids.append(ids[j])

        test_accuracy = 100 * correct_ / total_
        print(f'Final Test Loss: {test_loss/len(test_loader)}, Final Test Accuracy: {test_accuracy}%')

        precision, recall, f1, _ = precision_recall_fscore_support(true_diseases, predicted_diseases, average='weighted')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        test_log_message = f'Final Test Acc: {test_accuracy}\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n'
        with open(test_log_path, 'a') as test_log:
            test_log.write(test_log_message)
        
        with h5py.File(os.path.join(checkpoint_dir, f'predicted_masks_fold_{i}.h5'), 'w') as hf:
            hf.create_dataset('predicted_masks', data=np.array(predicted_masksB))
            hf.create_dataset('true_diseases', data=np.array(true_diseases))
            hf.create_dataset('predicted_diseases', data=np.array(predicted_diseases))
            hf.create_dataset('ids', data=np.array(test_ids))
    
        labels = ["Normal", "DR", "AMD"]

        cm = confusion_matrix(true_diseases, predicted_diseases)
        print('Confusion Matrix:\n', cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_fold_{i}.png'))
        
        true_diseases = np.array(true_diseases)
        predicted_probs = np.array(predicted_probs_list)
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
