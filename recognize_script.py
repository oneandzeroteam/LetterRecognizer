from LetterRecognizer.model import RecogModel
from Tensorflow_DataConverter.load.img_loader import load_image
from Tensorflow_DataConverter.load.converter_img import convert_image_to_numpy
from Tensorflow_DataConverter.process.normalizer_pillow import normalize_image
from Tensorflow_DataConverter.visualize.visualizer_numpy import show_numpy_image


path = "data/test/0/0 (1).jpg"
img = load_image(path)
img = normalize_image(img)
img = convert_image_to_numpy(img)

model = RecogModel()
print(model.predict(img))
show_numpy_image(img, model.predict(img))
import matplotlib
matplotlib.pyplot.show()