import tkinter as tk
import cv2
from PIL import ImageTk, Image
from square.client import Client
from dotenv import load_dotenv
import os
import webbrowser
<<<<<<< HEAD
=======

import torch
from vision.preprocessing import Preprocessing
import vision.ProductIdentification
from vision.ProductIdentification import ProductDatabase, InferenceModel, SiameseNetwork
import numpy as np

import sqlite3
>>>>>>> MLmodel
import db


load_dotenv()
ID = os.environ.get("id")
TOKEN = os.environ.get("token")
print("xxx100", TOKEN)


LARGE_FONT = ("Verdana", 12)
SMALL_FONT = ("Verdana", 8)

CATALOG = {'adidas_cap': 4,
           'american_crew_hair_paste': 5,
           'coconut_water': 4,
           'oven_mitt': 9,
           'pink_ukelele': 49,
           'scissors': 5,
           'spoon': 2,
           'teddy_hamster': 9,
           'water_bottle': 2,
           'computer_mouse': 19,
           'disinfecting_cleaner': 5,
           'febreeze_spray': 8,
           'fork': 2,
           'keyboard': 99,
           'knife': 9,
           'ladle': 5,
           'nutella': 5,
           'omachi_mi_bap_bo': 1,
           'omachi_sot_bo_ham': 2,
           'phone': 599,
           'shear_revival_clay_pomade': 29,
           'skippy_peanut_butter': 4,
           'sprite': 2,
           'teddy_octopus': 4,
           'watch': 2000,
           'wine': 199}

# Camera object
CAMERA = cv2.VideoCapture(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
predicted_item_list = []

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
            #----- Initialize the window -------
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(width=800, height=600)
        self.geometry("800x600")
        container.pack()
        # the dictionary to keep track of the frame
        # object
        self.frames = {}

        # one page in dictionary
        for p in (WelcomePage, CameraPage, CheckoutPage, PaymentPage, SideCameraPage):
            # get the name of the object page
            f_name = p.__name__
            frame = p(container, self)
            # place the frame in the grid (so that it can be raised)
            frame.grid(row=0,column=0,sticky="nsew")
            self.frames[f_name] = frame
        print("xxx199.frame_dict", self.frames)

        self.show_frame("WelcomePage")
    
    # function to show the page of the page
    # name that is passed into it
    def show_frame(self, f_name):
        # controller will be the key
        # of the frame we want to show
        frame = self.frames[f_name]
        print("xxx222, show_frame", frame)
        frame.tkraise()

    def get_frame(self, f_name):
        return self.frames[f_name]

# the welcome page
class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # create the checkout button
        label = tk.Label(self, text="\n\nWelcome!\n\n", font=LARGE_FONT)
        label.pack()
        button = tk.Button(self, text="Checkout", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("CameraPage"))
        button.pack()

# the camera page
class CameraPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Please checkout", font=LARGE_FONT)
        label.pack()
        # back button
        back_button = tk.Button(self, text="Back", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("WelcomePage"))
        back_button.pack()
        
        # got to checkout
        next_button = tk.Button(self, text="Next", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("CheckoutPage"))
        next_button.pack()
        


        # add additional categories
        side_camera_button = tk.Button(self, text="Add Item", font=LARGE_FONT, width=12, height=2,
                           command=self.show_checkout_page)
        side_camera_button.pack()

        # button to take a picture
        capture_button = tk.Button(self, text="Capture", font=LARGE_FONT, width=12, height=2,
                           command=self.show_checkout_page)
        capture_button.pack()
         # integrating the camera inside the GUI
        self.video = tk.Label(self)
        self.video.pack()

        self.update_frame()

    def update_frame(self):
        # Get a frame from the video stream
        ret, frame = CAMERA.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create a PhotoImage from the frame
            self.img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the video label with the new frame
            self.video.configure(image=self.img)
        # Schedule the update_frame function to be called
        # after a delay of 30ms
        self.after(30, self.update_frame)
    
    def capture_frame(self):
        _, frame = CAMERA.read()
        # cv2.imwrite("captured_frame.jpg", frame)
        self.predicted_item_list = self.predict_item(frame)
        print("xxx1333.", predicted_item_list)

    def show_checkout_page(self):
        self.capture_frame()
        print("xxx5552.item-list", self.predicted_item_list)
        self.controller.frames["CheckoutPage"].update_list(self.predicted_item_list)
        self.controller.show_frame("CheckoutPage")

    def predict_item(self, img):
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

        if not os.path.exists('product_database.pth'):
            print("Making database")
            dataset_path = 'dataset/'
            model.eval()
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

        print("Preprocessing images")
        preprocessing = Preprocessing(img)
        obj_img_list = preprocessing.get_obj_img()
            

        idx2class = {}
        for class_name in class2idx_map:
            idx2class[class2idx_map[class_name]] = class_name
        print("Inference")
        model.eval()
        inference_model = InferenceModel(model, encode_bucket, idx2class)
        predicted_product_list = []
        for i, obj_img in enumerate(obj_img_list):
            predicted_product_list.append(inference_model.product_matching(obj_img))

        predicted_product_list = [idx2class[idx] for idx in predicted_product_list]
        return predicted_product_list


class SideCameraPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Please checkout", font=LARGE_FONT)
        label.pack()
        # back button
        back_button = tk.Button(self, text="Back", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("CameraPage"))
        back_button.pack()

        # item name entry
        self.item, self.price, self.frame = tk.StringVar(), tk.IntVar(), []
        item_text = tk.Label(self, text="Item*", font=SMALL_FONT)
        item_text.pack()
        item_entry = tk.Entry(self, textvariable=self.item)
        item_entry.pack()

        # item price entry
        price_text = tk.Label(self, text="Price", font=LARGE_FONT)
        price_text.pack()
        price_entry = tk.Entry(self, textvariable=self.price)
        price_entry.pack()

        # button to take a picture
        capture_button = tk.Button(self, text="Capture", font=LARGE_FONT, width=12, height=2,
                           command=self.capture_frame_and_info)
        capture_button.pack()
        self.video = tk.Label(self)
        self.video.pack()
        self.get_info()
        self.update_frame()

    def update_frame(self):
        # Get a frame from the video stream
        ret, frame = CAMERA.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create a PhotoImage from the frame
            self.img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            # Update the video label with the new frame
            self.video.configure(image=self.img)
        # Schedule the update_frame function to be called
        # after a delay of 30ms
        self.after(10, self.update_frame)
    
    def capture_frame_and_info(self):
        # initialize database connection
        conn = db.connection()
        cursor = conn.cursor()
        
        # fetch the item from the event
        _, frame = CAMERA.read()
        self.frame = frame
        print("xxx777.frame_info", frame)
        
        self.item = self.item.get()
        print("xxx777.item_info", self.item)
        
        self.price = self.price.get()
        print("xxx788.item_info", self.price)

        #add item and price to the database
        cursor.execute("INSERT INTO Items (item, price) VALUES (?, ?)",
                        (self.item, self.price))
        conn.commit()

        cv2.imwrite("captured_frame.jpg", frame)
        conn.close()
    
    def get_info(self):
        print("xxx799, get new item info", self.item, self.frame)

 # the checkout  page
 # this page will containt the stripe
 # checkout
class CheckoutPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.item_list = []
        #initialize customer information
        self.first_name, self.last_name, self.email = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.controller = controller

    def update_list(self, item_list):
        print("xxx555.item_list for checkout", item_list)
        # label
        label = tk.Label(self, text="\n\nHere are your items", font=LARGE_FONT)
        label.pack()
        
        # item lists
        self.item_list = item_list
        self.sum = 0
        item_text = ""
        if self.item_list:
            for item in self.item_list:
                price = CATALOG[item]
                item_text += item + ": $" + str(price) + "\n"
                self.sum += price

        item_label = tk.Label(self, text=item_text, font=LARGE_FONT)
        item_label.pack()
        price_label = tk.Label(self, text="Your total is $" + str(self.sum) + "\n", font=LARGE_FONT)
        price_label.pack()

        checkout = tk.Button(self, text="Continue To Payment", font=LARGE_FONT, width=20, height=2,
                             command=self.show_next_frame)
        checkout.pack()
    
    def show_next_frame(self):
        self.controller.frames["PaymentPage"].update_payment(self.item_list, self.sum)
        self.controller.show_frame("PaymentPage")


# Payment page
# Will redirect to the Square Payment Page 
class PaymentPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

    def update_payment(self, item_list, sum):
        label = tk.Label(self, text="Please Enter follow the link to checkout", font=LARGE_FONT)
        label.pack()
        self.item_list = item_list
        self.sum = sum
        result = client.checkout.create_payment_link(
                    body = {
                        "quick_pay": {
                        "name": "Your Items",
                        "price_money": {
                            "amount": sum * 100, # since the Square API is in cents
                            "currency": "USD"
                        },
                        "location_id": "LVT2MPPHNKY2X"
                        }
                    }
                )
        url = ""
        if result.is_success():
            print(result.body)
            url = result.body["payment_link"]["url"]
        elif result.is_error():
            print(result.errors)

        link = tk.Label(self, text=url,font=('Helveticabold', 12), fg="blue", cursor="hand2")
        link.pack()
        link.bind("<Button-1>", lambda e:
                  self.redirect(url))


    def redirect(self, url):
        webbrowser.open_new_tab(url)

if __name__ == "__main__":
    client = Client (
                access_token=TOKEN,
                environment='sandbox'
            )
    app = App()
    app.mainloop()
