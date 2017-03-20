from config.config import FTP_Parameter
from FTP_Manager.downloadfromftp import downloadfromftp


ftp_pwd = ''
downloadfromftp(FTP_Parameter(ftp_pwd))