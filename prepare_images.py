import matplotlib.pyplot as plt
import cv2
import os
import time
import numpy as np

# import own modules
from modules.background_removal import background_removal as bgrv
import modules.config as config

def prepare_images(raw_root, target_path, debug_mode=False, subpath=""):
    """Prepares all Pictures in the given directory and all subdirectories

    Keyword arguments:
    raw_root -- path with raw data
    target_path -- path where the prepared images should be saved
    debug_mode -- creates a debug picture for each picture if True
    subpath -- used by function to build the filetree for recursive calls
    """

    pictures_labels = ["original", "contour", "cutout"]

    # this checks if there's already a ziphandle given. if no handle is given
    # the function hasn't been recursively called by itself
    if subpath == "":
        # first instance - not recursively called
        # save the current working dir to be able to change back to it when
        # function is finished
        main_cwd = os.getcwd()

        # set current working dir to the directory that should be zipped
        os.chdir(raw_root)

        # create a ziphandle and call the function recursive using the 
        # filehandle starting with the "root" directory
        prepare_images(raw_root, target_path, debug_mode=debug_mode, subpath="/")

        # change the current working dir back to the original working dir
        os.chdir(main_cwd)
    else:
        # n-th instance of function
        # loop through each object in current dir
        for entry in os.listdir(raw_root+subpath):

            # building a path+file-string for the current object of the
            # directory
            obj = raw_root+subpath+entry

            # If the current object is a directory write it in the zipfile and
            # call the function recursively
            # If the current object is a file write it into the zipfile
            if os.path.isdir(obj):
                print("DIRECTORY: ", raw_root+subpath+entry)
                os.makedirs(target_path+subpath+entry, exist_ok=True)
                if debug_mode:
                    os.makedirs(target_path+"/debug"+subpath+entry, exist_ok=True)
                    #print("DEBUG:", target_path+"/debug"+subpath+entry)
                prepare_images(raw_root, target_path, debug_mode=debug_mode, subpath=subpath+entry+"/")
            else:
                print(raw_root+subpath+entry)
                pictures = []
                try:
                    img = cv2.imread(raw_root+subpath+entry, 1)
                    pictures.append(img)
                except Exception as e:
                    print("ERROR:", e)
                else:
                    try:
                        pictures.extend(bgrv(pictures[0], config.CONTOUR_MODEL))
                    except Exception as e:
                        print("ERROR BGRV:", e)
                    else:
                        try:
                            cv2.imwrite(target_path+subpath+entry+".jpg", pictures[-1])
                        except Exception as e:
                            print("ERROR IMWRITE:", e)

                        if debug_mode:
                            plt.figure(figsize=(24, 8))
                            for i in range(len(pictures_labels)):
                                plt.subplot(1,len(pictures_labels),i+1)
                                plt.rc("font", size=30)
                                plt.xticks([])
                                plt.yticks([])
                                plt.xlabel(pictures_labels[i], color="black")
                                try:
                                    plt.imshow(cv2.cvtColor(pictures[i], cv2.COLOR_BGR2RGB))
                                except Exception as e:
                                    print("ERROR IMSHOW:",e)

                            plt.savefig(target_path+"/debug"+subpath+entry+".jpg", dpi=100, transparent=False,
                            facecolor="w")
                            plt.close()

if __name__ == "__main__":
    prepare_images(config.RAW_DATA, config.TRAINING_DATA, debug_mode=True)

    if False:
        time.sleep(60)
        os.system('shutdown -s')