from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
class HAPTDataset(Dataset):

    def __init__(self, args):
        super(HAPTDataset, self).__init__(args)

        self.label_encoder = None  # Initialize label encoder to None

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading HAPT train data")
        train_dataset = pd.read_csv(r"W:\MSc-PhD\Federated-Learning-with-Blockchain-\data\HAPT\train.csv")

        self.label_encoder = LabelEncoder()  # Initialize label encoder here

        self.get_args().get_logger().debug("Finished loading HAPT train data")

        return (np.array(train_dataset.drop('Activity', axis=1)), self.label_encoder.fit_transform(train_dataset['Activity']))

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading HAPT train data")
        test_dataset = pd.read_csv(r"W:\MSc-PhD\Federated-Learning-with-Blockchain-\data\HAPT\test.csv")

        self.get_args().get_logger().debug("Finished loading HAPT train data")

        return (np.array(test_dataset.drop('Activity', axis=1)), self.label_encoder.transform(test_dataset['Activity']))

