# FTP module parameters
class FTP_Parameter:
    def __init__(self, ftp_pwd):
        self.init()
        self.ftp_pwd = ftp_pwd

    def init(self):
        self.ftp_domain = 'yangyinetwork.asuscomm.com'
        self.ftp_user = 'soyung'
        self.ftp_pwd = ''
        self.ftp_homepath = '/My_Passport/Download2/workspace/data/CarlicenseProject'
        self.ftp_uploadlocaldir = 'data'
        self.ftp_downloadlocaldir = 'data'
        self.ftp_targetpath = 'number'

    def set_ftp_targetpath(self,ftp_targetpath):
        self.ftp_targetpath = ftp_targetpath

class Hyparms():
    def __init__(self):
        pass

# Hyper Parameters
HYPARMS = Hyparms()

HYPARMS.batch_size = 10
HYPARMS.learning_rate = 0.01
HYPARMS.max_steps = 3000
HYPARMS.log_dir = 'logs'
HYPARMS.train_data_dir = 'data/train'
HYPARMS.test_data_dir = 'data/test'
HYPARMS.recog_data_dir = 'data/recog'
HYPARMS.ckpt_dir = 'logs'
HYPARMS.ckpt_name = 'trained_weight'
HYPARMS.dropout_rate = 0.9