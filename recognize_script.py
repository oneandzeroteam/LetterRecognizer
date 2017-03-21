from LetterRecognizer.model import RecogModel
from Tensorflow_DataConverter.load.dir_loader import load_allpath
from Tensorflow_DataConverter.load.img_loader import load_image
from Tensorflow_DataConverter.load.converter_img import convert_image_to_numpy
from Tensorflow_DataConverter.process.normalizer_pillow import normalize_image
from Tensorflow_DataConverter.visualize.visualizer_numpy import show_numpy_image
import matplotlib


path = "data/test/0/0 (1).jpg"

def recog_img_by_path(model, path):
    img = load_image(path)
    img = normalize_image(img)
    img = convert_image_to_numpy(img)

    print(model.predict(img))
    show_numpy_image(img, model.predict(img))
    matplotlib.pyplot.show()

model = RecogModel()
paths = load_allpath("data/sample")
for path in paths:
    recog_img_by_path(model, path)