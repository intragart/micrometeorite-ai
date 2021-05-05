# import libraries
import os
import zipfile

def zip_dir(zip_file, dir_to_zip, zip_handle=None, subpath=""):
    """Creates a Zipfile of the given directory and all subdirectories

    Keyword arguments:
    zip_file -- path+filename where the zipfile should be saved
    dir_to_zip -- path to be zipped
    zip_handle -- used by function to pass on the ziphandle into recursive calls
    subpath -- used by function to build the filetree for recursive calls
    """

    # this checks if there's already a ziphandle given. if no handle is given
    # the function hasn't been recursively called by itself
    if zip_handle is None:
        # first instance - not recursively called
        # save the current working dir to be able to change back to it when
        # function is finished
        main_cwd = os.getcwd()

        # set current working dir to the directory that should be zipped
        os.chdir(dir_to_zip)

        # create a ziphandle and call the function recursive using the 
        # filehandle starting with the "root" directory 
        with zipfile.ZipFile(zip_file, "w") as zf:
            zip_dir("", dir_to_zip+"/", zf)

        # change the current working dir back to the original working dir
        os.chdir(main_cwd)
    else:
        # n-th instance of function
        # loop through each object in current dir
        for entry in os.listdir(dir_to_zip+subpath):

            # building a path+file-string for the current object of the
            # directory
            obj = dir_to_zip+subpath+entry

            # If the current object is a directory write it in the zipfile and
            # call the function recursively
            # If the current object is a file write it into the zipfile
            if os.path.isdir(obj):
                zip_handle.write(subpath+entry)
                zip_dir("", dir_to_zip, zip_handle, subpath+entry+"/")
            else:
                zip_handle.write(subpath+entry)