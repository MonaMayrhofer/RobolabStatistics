import dropbox
import os

APP_KEY = "bdnfbbtz309an28"
APP_SECRET = "mmrpk3jqkk6zhgu"


def update_files(foldername):
    for entry in dbx.files_list_folder(foldername).entries:
        name = foldername + '/' + entry.name
        if '.' not in entry.name:
            print("Folder " + name)
            if not os.path.exists('.' + name):
                print("Creating " + '.' + name)
                os.makedirs('.' + name)
            update_files(name)
        else:
            print("File " + name)
            if not os.path.exists('.' + name):
                print("Downloading " + name)
                dbx.files_download_to_file('.' + name, name)


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

folder = input("Select folder to update: ")
update_files('/' + folder)
