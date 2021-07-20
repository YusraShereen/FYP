from googleapiclient.http import MediaFileUpload
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def getdata():
    gauth = GoogleAuth()
    #gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1ERyEZJByCjf8UPDECA_C2P48X3CT2tYd')}).GetList()
    # # for file in file_list:
    # # 	print('title: %s, id: %s' % (file['title'], file['id']))

    for i, file in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
        print("---------------------NEW FILE---------------------------")
        print(file['title'])
        file = drive.CreateFile({'id': file['id']})
        txt = file.GetContentString(file['title'])
        print(txt)
        #print('Downloading {} file from GDrive ({}/{})'.format(file['title'], i, len(file_list)))
        #print(file['title'])
        # file.Delete()
        #x = file.GetContentString(file['title'])

    return txt




