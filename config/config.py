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
        self.ftp_downloadlocaldir = 'data'
        self.ftp_targetpath = 'number'
        # self.ftp_uploadlocaldir = 'data/test'

    def set_ftp_targetpath(self,ftp_targetpath):
        self.ftp_targetpath = ftp_targetpath