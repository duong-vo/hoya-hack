import skimage.io, skimage.color, skimage.filters, skimage.morphology, skimage.measure, skimage.transform
import numpy as np
from PIL import Image
from cv2 import resize, imshow, waitKey, destroyAllWindows

class Preprocessing:
    def __init__(self, image):
        """Preprocessing image before feeding into Neural Network

        Args:
            image (PIL image): image of the scan table
        """
        self.SIZE = (224,224)
        self.pil_image = image
        self.pil_image.resize((800,1000))
        self.image = np.asarray(image)
        
        self.bboxes = [] #min_row, min_col, max_row, max_col
    def get_obj_img(self):
        """Return the close crop of each ofject on the scan table

        Returns:
            List: List of PIL cropped image of each objects
        """
        image = skimage.filters.gaussian(self.image)
        img_hsv = skimage.color.rgb2hsv(self.image)

        h = img_hsv[:,:,0]
        s = img_hsv[:,:,1]
        h_range = [0.05, 0.13]
        s_range = [0.25, 0.5]
        # h_range = [0.05, 0.17]
        # s_range = [0.3, 0.5]
        mask = (((h<h_range[0]) | (h>h_range[1])) & ((s<s_range[0]) | (s>s_range[1])))

        mask = skimage.filters.median(mask, np.ones((9,9)))

        mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(10))
        mask = skimage.morphology.binary_opening(mask, skimage.morphology.disk(7))
        mask = skimage.morphology.binary_closing(mask, skimage.morphology.disk(7))

        labels = skimage.measure.label(mask)
        probs = skimage.measure.regionprops(labels)

        for prob in probs:
            
            if prob.area_filled < 15000: continue
            self.bboxes.append(prob.bbox)
        obj_img = []
        
        for i, bbox in enumerate(self.bboxes):
            print(bbox)
            crop = self.image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
            crop = resize(crop, (224,224))
            obj_img.append(Image.fromarray(crop))

        return obj_img
