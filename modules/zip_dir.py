import os
import zipfile

def zip_dir(zip_file, dir_to_zip, zip_handle=None, subpath=""):
    
    if zip_handle is None:
        # first instance
        main_cwd = os.getcwd()
        os.chdir(dir_to_zip)
        with zipfile.ZipFile(zip_file, "w") as zf:
            zip_dir("", dir_to_zip+"/", zf)
        os.chdir(main_cwd)
    else:
        # n-th instance
        # loop through each object in current dir
        for entry in os.listdir(dir_to_zip+subpath):

            obj = dir_to_zip+subpath+entry

            if os.path.isdir(obj):
                zip_handle.write(subpath+entry)
                zip_dir("", dir_to_zip, zip_handle, subpath+entry+"/")
            else:
                zip_handle.write(subpath+entry)