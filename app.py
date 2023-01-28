import tkinter as tk
import cv2
from PIL import ImageTk, Image
from square.client import Client
from dotenv import load_dotenv
import os

load_dotenv()
ID = os.environ.get("id")
TOKEN = os.environ.get("token")
print("xxx100", TOKEN)

#create



LARGE_FONT = ("Verdana", 12)
SMALL_FONT = ("Verdana", 8)

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
        for p in (WelcomePage, CameraPage, CheckoutPage):
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

# the welcome page
class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # create the checkout button
        label = tk.Label(self, text="Welcome!", font=LARGE_FONT)
        label.pack()
        button = tk.Button(self, text="Checkout", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("CameraPage"))
        button.pack()

# the camera page
class CameraPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Please checkout", font=LARGE_FONT)
        label.pack()
        # back button
        back_button = tk.Button(self, text="Back", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("WelcomePage"))
        back_button.pack()
        # button to take a picture
        submit_button = tk.Button(self, text="Submit", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("CheckoutPage"))
        submit_button.pack()

        capture_button = tk.Button(self, text="Capture", font=LARGE_FONT, width=12, height=2,
                           command=self.capture_frame)
        capture_button.pack()

        # integrating the camera inside the GUI
        self.video = tk.Label(self)
        self.video.pack()

        self.vid = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        # Get a frame from the video stream
        ret, frame = self.vid.read()
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
        _, frame = self.vid.read()
        cv2.imwrite("captured_frame.jpg", frame)

 # the checkout  page
 # this page will containt the stripe
 # checkout
class CheckoutPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Please Enter Your Billing Information", font=LARGE_FONT)
        label.pack()

        #initialize customer information
        self.first_name, self.last_name, self.email = tk.StringVar(), tk.StringVar(), tk.StringVar()


        # input customer information
        # first name
        first_text = tk.Label(self, text="First Name*", font=SMALL_FONT)
        first_text.pack()
        first_entry = tk.Entry(self, textvariable=self.first_name)
        first_entry.pack()


        # last name
        last_text = tk.Label(self, text="Last Name*", font=SMALL_FONT)
        last_text.pack()
        last_entry = tk.Entry(self, textvariable=self.last_name)
        last_entry.pack()

        # email
        email_text = tk.Label(self, text="Email*", font=SMALL_FONT)
        email_text.pack()
        email_entry = tk.Entry(self, textvariable=self.email)
        email_entry.pack()

        # buttons
        # register
        register = tk.Button(self, text="Register", width=12, height=2,
                             command=self.save_info)
        register.pack()

        button = tk.Button(self, text="Back", font=LARGE_FONT, width=12, height=2,
                           command= lambda: controller.show_frame("WelcomePage"))
        button.pack()
    
    # input the user info to register customer
    def save_info(self):
        first_name_info = self.first_name.get()
        print("xxx333.received first name", first_name_info)
        last_name_info = self.last_name.get()
        print("xxx334.received last name", last_name_info)
        email_info = self.email.get()
        print("xxx335.received email", email_info)
        self.create_customer(first_name_info, last_name_info, email_info)
    
    # create customer through square api
    def create_customer(self, first, last, email):
        result = client.customers.create_customer(
            body = {
                "given_name": first,
                "family_name": last,
                "email_address": email
            }
        )

        if result.is_success():
            print("xxx200.result", result.body)
        elif result.is_error():
            print("xxx400.result", result.errors)



if __name__ == "__main__":
    client = Client (
                access_token=TOKEN,
                environment='sandbox'
            )
    app = App()
    app.mainloop()