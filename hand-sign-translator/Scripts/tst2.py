import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)

# Load the labels
with open("Model/labels.txt", "r") as file:
    labels = [line.strip() for line in file.readlines()]
def get_prediction(img):
    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Get prediction
    prediction = model.predict(img)
    index = np.argmax(prediction)
    
    return prediction[0], index