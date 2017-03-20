from config.config import FTP_Parameter
from FTP_Manager.uploadtoftp import uploadtoftp



ftp_pwd = ''
uploadtoftp(FTP_Parameter(ftp_pwd))
