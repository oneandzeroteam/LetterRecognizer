from config.config import FTP_Parameter, HYPARMS
from FTP_Manager.downloadfromftp import downloadfromftp
from Tensorflow_DataConverter import inputdata
from LetterRecognizer.model import Model


# ftp_pwd = ''
# downloadfromftp(FTP_Parameter(ftp_pwd))

data_sets = inputdata.read_image_datasets(HYPARMS.train_data_dir,
                                          HYPARMS.test_data_dir,
                                          reshape=False)
# from Tensorflow_DataConverter.visualize.visualizer_numpy import show_random_nbyn_data
# show_random_nbyn_data(data_sets.train)

model = Model()
model.load_data(data_sets.train, data_sets.test)
model.train()
