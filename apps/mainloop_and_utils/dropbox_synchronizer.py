import dropbox
import os

APP_KEY = "bdnfbbtz309an28"
APP_SECRET = "mmrpk3jqkk6zhgu"


def update_files(foldername):
    if '.' not in foldername:
        print("Folder " + foldername)
        if not os.path.exists('.' + foldername):
            print("Creating " + '.' + foldername)
            os.makedirs('.' + foldername)
        for entry in dbx.files_list_folder(foldername).entries:
            update_files(foldername + '/' + entry.name)
    else:
        print("File " + foldername)
        if not os.path.exists('.' + foldername):
            print("Downloading " + foldername)
            dbx.files_download_to_file('.' + foldername, foldername)


authflow = dropbox.DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET)
authorizeurl = authflow.start()
print("1. Go to: " + authorizeurl)
print("2. Click \"Allow\" (you might have to log in first).")
print("3. Copy the authorization code.")
authcode = input("Enter the authorization code here: ").strip()

oauth_result = authflow.finish(authcode)
dbx = dropbox.Dropbox(oauth_result.access_token)
acc = dbx.users_get_current_account()
print("Account linked: " + acc.name.display_name)

folder = input("Select folder or file to update: ")
update_files('/' + folder)
