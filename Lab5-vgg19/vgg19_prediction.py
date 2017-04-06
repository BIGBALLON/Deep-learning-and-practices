import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from PIL import Image
from keras.preprocessing import image
import numpy as np

model = VGG19(weights='imagenet')

while True:
	img_path = input('Please input picture file to predict ( input Q to exit ):')
	if img_path == 'Q':
		break
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	results = model.predict(x)
	print('Predicted:', decode_predictions(results, top=5)[0])
