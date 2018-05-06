import dropbox
import os


def download_files():
    folders = input("Files and/or folders do download (seperate by ';'): ")
    dbx = start_session()
    for foldername in folders.split(';'):
        if '.' not in foldername:
            recursive = ""
            while recursive != "Y" and recursive != "N":
                recursive = input("Download subfolders recursively? (Y/N): ")
                if recursive == "Y":
                    download_files_recursive(dbx, '/' + foldername, True)
                elif recursive == "N":
                    download_files_recursive(dbx, '/' + foldername, False)
        else:
            download_files_recursive(dbx, '/' + foldername, False)


def download_files_recursive(dbx, foldername, recursive):
    if '.' not in foldername:
        if recursive:
            print("Folder " + foldername)
            if not os.path.exists('.' + foldername):
                print("Creating " + '.' + foldername)
                os.makedirs('.' + foldername)
        for entry in dbx.files_list_folder(foldername).entries:
            if '.' in entry.name or recursive:
                download_files_recursive(dbx, foldername + '/' + entry.name, True)
    else:
        print("File " + foldername)
        if not os.path.exists('.' + foldername):
            print("Downloading " + foldername)
            dbx.files_download_to_file('.' + foldername, foldername)


def start_session():
    APP_KEY = "bdnfbbtz309an28"
    APP_SECRET = "mmrpk3jqkk6zhgu"
    authflow = dropbox.DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET)
    authorizeurl = authflow.start()
    print("1. Go to: " + authorizeurl)
    print("2. Click \"Allow\" (you might have to log in first).")
    print("3. Copy the authorization code.")
    authcode = input("Enter the authorization code here: ").strip()
    oauthresult = authflow.finish(authcode)
    dbx = dropbox.Dropbox(oauthresult.access_token)
    acc = dbx.users_get_current_account()
    print("Account linked: " + acc.name.display_name)
    return dbx


print(start_session().sharing_create_shared_link_with_settings("/convlfw").url)
#download_files()
