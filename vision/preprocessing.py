import skimage.io, skimage.color, skimage.filters, skimage.morphology, skimage.measure
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import resize

class Preprocessing:
    def __init__(self, image):
        """Preprocessing image before feeding into Neural Network

        Args:
            image (cv2 image): image of the scan table
        """
        self.SIZE = (224,224)
        img = resize(img, self.SIZE)
        self.image = image
        self.bboxes = [] #min_row, min_col, max_row, max_col
    def get_obj_img(self):
        """Return the close crop of each ofject on the scan table

        Returns:
            List: List of PIL cropped image of each objects
        """
        img_hsv = skimage.color.rgb2hsv(self.img)

        h = img_hsv[:,:,0]
        s = img_hsv[:,:,1]
        h_range = [0.13, 0.24]
        s_range = [0.25, 0.45]
        mask = (((h<h_range[0]) | (h>h_range[1])) & ((s<s_range[0]) | (s>s_range[1])))

        mask = skimage.filters.median(mask, np.ones((5,5)))

        mask = skimage.morphology.binary_opening(mask, skimage.morphology.disk(7))
        mask = skimage.morphology.binary_closing(mask, skimage.morphology.disk(7))

        labels = skimage.measure.label(mask)
        probs = skimage.measure.regionprops(labels)

        for prob in probs:
            self.bboxes.append(prob.bbox)
        
        obj_img = []
        pil_img = Image.fromarray(np.uint8(self.img*255))
        for i, bbox in enumerate(self.bboxes):
            crop = pil_img.crop(bbox[1::-1] + bbox[-1:1:-1])
            obj_img.append(crop)

        return obj_img        
