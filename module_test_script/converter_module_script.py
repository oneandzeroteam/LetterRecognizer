from Tensorflow_DataConverter import inputdata
from config.config import HYPARMS


data_sets = inputdata.read_image_datasets(HYPARMS.train_data_dir,
                                          HYPARMS.test_data_dir,
                                          reshape=False)