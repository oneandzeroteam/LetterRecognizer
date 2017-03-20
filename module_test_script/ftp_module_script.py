from config.config import FTP_Parameter
from FTP_Manager.downloadfromftp import downloadfromftp


ftp_pwd = '12345'
FTP_PARMS = FTP_Parameter(ftp_pwd)
downloadfromftp(FTP_PARMS)