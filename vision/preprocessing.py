import skimage.io, skimage.color, skimage.filters, skimage.morphology, skimage.measure, skimage.transform
import numpy as np
from PIL import Image
from cv2 import resize, imshow, waitKey, destroyAllWindows

class Preprocessing:
    def __init__(self, image):
        """Preprocessing image before feeding into Neural Network

        Args:
            image (cv2 image): image of the scan table
        """
        self.SIZE = (224,224)
        self.image = skimage.transform.resize(image, (1000, 800),
                       anti_aliasing=True)
        self.image = skimage.filters.gaussian(self.image)
        self.bboxes = [] #min_row, min_col, max_row, max_col
    def get_obj_img(self):
        """Return the close crop of each ofject on the scan table

        Returns:
            List: List of PIL cropped image of each objects
        """
        img_hsv = skimage.color.rgb2hsv(self.image[:,:,::-1])

        h = img_hsv[:,:,0]
        s = img_hsv[:,:,1]
        h_range = [0.05, 0.13]
        s_range = [0.25, 0.5]
        mask = (((h<h_range[0]) | (h>h_range[1])) & ((s<s_range[0]) | (s>s_range[1])))

        mask = skimage.filters.median(mask, np.ones((9,9)))

        mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(10))
        mask = skimage.morphology.binary_opening(mask, skimage.morphology.disk(7))
        mask = skimage.morphology.binary_closing(mask, skimage.morphology.disk(7))

        # imshow("mask", np.uint8(mask)*255)
        # waitKey(0)
        # destroyAllWindows()
        labels = skimage.measure.label(mask)
        probs = skimage.measure.regionprops(labels)

        print('----area filled---')
        for prob in probs:
            
            print(prob.area_filled)
            if prob.area_filled < 15000: continue
            self.bboxes.append(prob.bbox)
        print('----')
        obj_img = []
        pil_img = Image.fromarray(np.uint8(self.image*255))
        for i, bbox in enumerate(self.bboxes):
            crop = pil_img.crop(bbox[1::-1] + bbox[-1:1:-1])
            crop.resize(self.SIZE)
            obj_img.append(crop)

        return obj_img
