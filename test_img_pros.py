import torch
from vision.preprocessing import Preprocessing
import vision.ProductIdentification
from vision.ProductIdentification import ProductDatabase, InferenceModel, SiameseNetwork
import cv2
import numpy as np
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_item(img):
        """List all the items appeared in the images

        Args:
            img (cv2 image): image of the scanning table

        Returns:
            List: list of items appeared on the scanning table
        """
        model_path = 'weights/siamese_best_weight.pth.tar'
        model = SiameseNetwork().to(DEVICE)
        checkpoint = torch.load(model_path) if DEVICE == 'cuda' else torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model state dict'])
        print("Done loading model")

        print("Making database")
        if not os.path.exists('product_database.pth'):
            dataset_path = 'dataset/train/'
            database = ProductDatabase(dataset_path, model)
            encode_bucket = database.get_encode_bucket()
            class2idx_map = database.get_class2idx_map()
            class_list = database.get_class_list()

            #Save to pth object
            torch.save({'encoding': encode_bucket, 
                        'class_list': class_list,
                        'class2idx_map': class2idx_map,
                        },
                        "product_database.pth")
        else:
            database_info = torch.load('product_database.pth')
            encode_bucket = database_info['encoding']
            class2idx_map = database_info['class2idx_map']
            class_list = database_info['class_list']

        print(f'class2idx: {class2idx_map}')
        print(f'Class list: {class_list}')
        print("Preprocessing images")
        preprocessing = Preprocessing(img)
        obj_img_list = preprocessing.get_obj_img()
        print(len(obj_img_list))
        for obj in obj_img_list:
            cv2.imshow("show", np.asarray(obj))
            cv2.waitKey(0)
        cv2.destroyAllWindows()

        idx2class = {}
        for class_name in class2idx_map:
            idx2class[class2idx_map[class_name]] = class_name
        print("Inference")
        inference_model = InferenceModel(model, encode_bucket, idx2class)
        predicted_product_list = []
        for i, obj_img in enumerate(obj_img_list):
            print(f"OBJ {i}")
            predicted_product_list.append(inference_model.product_matching(obj_img))
        print(idx2class)
        return predicted_product_list

img = cv2.imread('dataset/train/wine/IMG_7092.jpg')
predicted_label = predict_item(img)
print(predicted_label)