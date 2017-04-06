import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from PIL import Image
from keras.preprocessing import image
import numpy as np
import os.path

model = VGG19(weights='imagenet')

while True:
	img_path = input('Please input picture file to predict ( input Q to exit ):  ')
	if img_path == 'Q':
		break
	if not os.path.exists(img_path):
		print("file not exist!")
		continue
	try:
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x[:, :, 0] = x[:, :, 0] - 103.939
		x[:, :, 1] = x[:, :, 1] - 116.779
		x[:, :, 2] = x[:, :, 2] - 123.680
		x = np.expand_dims(x, axis=0)
		# x = preprocess_input(x)
		results = model.predict(x)
		print('Predicted:', decode_predictions(results, top=5)[0])
	except Exception as e:
		pass


	
