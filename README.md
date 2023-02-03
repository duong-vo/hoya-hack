[![IMAGE ALT TEXT](https://github.com/duong-vo/hoya-hack/blob/main/banner.png)](https://youtu.be/t6kov8xUKNU "Self-checkout Kiosk App with Camera for Retails")
<a href="https://youtu.be/t6kov8xUKNU"><img src="https://github.com/duong-vo/hoya-hack/blob/main/banner.png" style="width:300px;height:300px;"></a>
# Inspiration
We already have self-checkout with barcodes at most places in the college. However, I believe that we can save so much more time with the checkout process than that, which is using computer vision for checking out.

# What it does
Our checkout system utilizes a Machine Learning model called Siamese Network to do one-shot-learning object identification to identify items from the store's storage. We choose this model because of the ease of adding new items (or "class" in Deep Learning's language) to the store's checkout system. In more detail, we used pre-trained weights from the ResNet-18 model to extract features from images. After training, the model will be used to "compress" (or extract features) products' images into small vectors. These vectors would be stored in the database. 
When customers check out, the computer would take images of their products, crop them and feed them into the Deep Learning model to get the "feature vectors". Those would be compared with all of the features we have in the database, to know which product they would most likely be. Next, those proposed item predictions would be looked up on the store's catalog to get the price and calculate the total bill. Finally, with Square API, clients can use their credit cards to pay for the items.

# Challenges we ran into
We mainly had problems with the Machine Learning model. Training ML model needs lots of time for hyperparameter-tunings.
Version control is also a challenge for us. Branching and merging conflict causes a lot of headac

# What we learned
In the end, version control is the most we learned from this hackathon. We learn a lot about merging branches and resolving conflicts in GitHub. We hope that with this knowledge of version control, we will be more efficient with working on projects in later hackathons as well as in our later careers.

#Techstack: PyTorch, OpenCV, scikit-learn, Tkinter, SQLite, Square API.
