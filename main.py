import argparse
import os
from Unet.train import Segmentation
# you can change the module based on the approach you will use
from Classifier.train import Classification


def main(save_path):

    # necessary paths pip install jupyterlab
    fold_dir = "/ari/users/oeren/SegClass/3foldsTrainTest"
    save_path = os.path.join(f"/ari/users/oeren/SegClass/Training/", save_path)

    #########################################################################
    # If you will use basic approach, you should use one of the data paths  #
    # But if you will use ensemble approach, you must enable both of paths. #
    #########################################################################
    
    # data path for Horizontal B-Scans
    data_path1 = "/ari/users/oeren/SegClass/SegClassAll.npz"
    # data path for Vertical B-Scans
    data_path2 = "/ari/users/oeren/SegClass/SegClassAScanData2.npz"

    print("Classes have been creating.")
    # Create Segmentation and Classification Class for basic method
    segmentation = Segmentation(data_path2)
    classification = Classification(data_path2)
    
    # Create Segmentation and Classification Class for ensemble method
    #segmentation = Segmentation(data_path2)
    #classification = Classification(data_path1, data_path2)

    # Iterate over each fold
    num_folds = 5  # Adjust this to the number of folds you have
    print("Starting to folds...")
    for i in range(num_folds):
        print(f"Started to fold_{i}")
        train_path = os.path.join(fold_dir, f'train_fold_{i}.csv')
        test_path = os.path.join(fold_dir, f'test_fold_{i}.csv')
        
        segmentation.trainSeg(train_path, test_path, save_path, i)
        print(f"For fold{i}, segmentation has completed.")
        
        classification.trainClass(train_path, test_path, save_path, i, segmentation)
        print(f"For fold{i}, classification has completed.")
        print("10101010101011001101010101010101010101010100101011010101010101010101010101010101011001101010100101")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation and classification models.")
    parser.add_argument('--saving', type=str, required=True, help="Base directory to save weights.")

    args = parser.parse_args()

    # Ensure the save_weight directory exists
    os.makedirs(args.saving, exist_ok=True)

    main(args.saving)