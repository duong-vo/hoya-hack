import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from pickle import TRUE
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from cv2 import imshow, waitKey, destroyAllWindows

from collections import defaultdict
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_path = 'dataset/'
class ProductDatabase:
    def __init__(self, dataset_path, model):
        imageFolderDataset = datasets.ImageFolder(root=dataset_path)
        self.class2idx = imageFolderDataset.class_to_idx
        self.class_list = imageFolderDataset.classes
        self.encode_bucket = defaultdict(list)
        model.eval()

        for example in imageFolderDataset.imgs:
            image, label = Image.open(example[0]), example[1]
            image = image.resize((224,224))
            image = TF.to_tensor(image)
            image = image.to(DEVICE)
            image = image[None, :]
            encode = model.forward_once(image)
            self.encode_bucket[label].append(encode)

    def get_encode_bucket(self):
        return self.encode_bucket
    def get_class2idx_map(self):
        return self.class2idx
    def get_class_list(self):
        return self.class_list
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #Get pre-trained model from resnet
        resnet18 = models.resnet18(pretrained=True)

        self.feature_extractor = nn.ModuleList(resnet18.children())[:-1]
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        in_features = resnet18.fc.in_features
        self.final_block = nn.Sequential(
            nn.Linear(in_features, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 50)
        )
        
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
    def forward_once(self, x):
        output = self.feature_extractor(x)
        output = output.view(output.size()[0], -1)
        output = self.final_block(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3
class InferenceModel:
    def __init__(self, model, encode_database, idx2class):
        self.model = model
        
        self.encode_database = encode_database
        self.idx2class = idx2class
        self.classes = self.encode_database.keys()
    def inference(self, encode1, encode2):
        """Measure similarity between 2 images encoding

        Args:
            encode1 (Torch Tensor): image 1's encoding
            encode2 (Torch Tensor): image 2's encoding

        Returns:
            float: dissimilarity score, the lower the more similar
        """
        with torch.no_grad():
            euclidean_distance = (encode1 - encode2).pow(2).sum(1)
        
        return euclidean_distance

    def product_matching(self, image):
        """Matching the input product with current product in the database

        Args:
            image (numpy array): input image for matching

        Returns:
            string: name of best fit class
        """
        image_tensor = TF.to_tensor(image)
        image_tensor = image_tensor.to(DEVICE)
        image_tensor = image_tensor[None, :]
        img_encode = self.model.forward_once(image_tensor)
        

        min_score = float('inf')
        best_fit_class = None
        for class_name in self.classes:
            total_score = 0.0
            for ref_encode in self.encode_database[class_name]:
                diff_score = self.inference(ref_encode, img_encode)
                total_score+=diff_score
            avg_score = total_score / len(self.encode_database[class_name])
            #Determine min
            if avg_score < min_score:
                min_score = diff_score
                best_fit_class = class_name

        return best_fit_class