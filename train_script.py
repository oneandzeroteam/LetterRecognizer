from config.config import FTP_Parameter, HYPARMS
from FTP_Manager.downloadfromftp import downloadfromftp
from Tensorflow_DataConverter import inputdata



ftp_pwd = ''
downloadfromftp(FTP_Parameter(ftp_pwd))

data_sets = inputdata.read_image_datasets(HYPARMS.train_data_dir,
                                          HYPARMS.test_data_dir,
                                          reshape=False)
