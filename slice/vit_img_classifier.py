import tensorflow as tf
import numpy as np
from transformers import ViTFeatureExtractor, TFViTForImageClassification

class ViTImageClassifier:
    def __init__(self, model_name):
        self.name = "vitp16"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = TFViTForImageClassification.from_pretrained(model_name)

    def preprocess_image(self, images):
        #image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        #inputs = self.feature_extractor(images=images, return_tensors="tf")
        images = images / 255.0

        return images


    def predict_old(self, inputs, verbose=0):
        outputs = self.model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()
        top_probability = tf.reduce_max(probabilities).numpy()
        return predicted_class, top_probability

    def predict(self, images, verbose=0): # this extracts and also predicts
        # Preprocess the batch of images
        images = np.round(images)
        inputs = self.feature_extractor(images=images, return_tensors="tf")

        # Make predictions using the pre-trained model
        outputs = self.model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)

        if verbose > 0:
            predicted_classes = tf.argmax(probabilities, axis=-1).numpy()
            top_probabilities = tf.reduce_max(probabilities, axis=-1).numpy()
            for i in range(len(images)):
                print(
                    f"Image {i + 1}: Predicted class index: {predicted_classes[i]} with probability: {top_probabilities[i]}")

        return probabilities.numpy()

    def model_name(self):
        return self.name

# # Usage example:
# model_name = "google/vit-base-patch16-224"
# model = ViTImageClassifier(model_name)
# image_path = 'C:\\Users\\revotipb\\PycharmProjects\\LimeImageSimple\\images_old\\elephant.jpg'
# #img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
# img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
# img = tf.keras.preprocessing.image.img_to_array(img)
# inputs = model.preprocess_image(img)
# predicted_class, top_probability = model.predict_old(inputs)
# print(f"Predicted class index: {predicted_class} with probability: {top_probability}")
#
# model_name = "google/vit-base-patch16-224"
# vit_classifier = ViTImageClassifier(model_name)
#
# # Load and preprocess a batch of example images (replace 'image_paths' with a list of image file paths)
# import numpy as np
#
# img_path = 'C:\\Users\\revotipb\\PycharmProjects\\LimeImageSimple\\images_old\\elephant.jpg'
# image_paths = ['C:\\Users\\revotipb\\PycharmProjects\\LimeImageSimple\\images_old\\elephant.jpg',
#                'C:\\Users\\revotipb\\PycharmProjects\\LimeImageSimple\\images_old\\elephant.jpg',
#                'C:\\Users\\revotipb\\PycharmProjects\\LimeImageSimple\\images_old\\elephant.jpg',
#                'C:\\Users\\revotipb\\PycharmProjects\\LimeImageSimple\\images_old\\elephant.jpg']
# images = [tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224)) for image_path in image_paths]
# images_array = np.array([tf.keras.preprocessing.image.img_to_array(image) for image in images])
#
# # Make predictions for the batch of images
# probabilities = vit_classifier.predict(images_array, verbose=1)
# print(probabilities)