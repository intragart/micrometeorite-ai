import os
import cv2

def pathCrawler(root_dir, levels=-1, subpath=""):
    """Returns a list of tupels. One for each file/folder.

    Keyword arguments:
    root_dir -- path to be crawled
    levels -- defines, how deeply is crawled (-1 = no limit, 0 = only root_dir, 1 = root_dir +1 in depth, ...)
    subpath -- used by function to build the filetree for recursive calls
    """

    entries = []

    # loop through each object in current dir
    for entry in os.listdir(f"{root_dir}/{subpath}"):

        # building a path+entry-string for the current object of the
        # directory
        obj = f"{root_dir}/{subpath}{entry}"

        # string that defines the current entries type
        obj_type = ""

        if os.path.isdir(obj):

            # the current object is a directory
            obj_type = "dir"

            # call the script recursively with the subpath and
            # keep the returns if there're depth-levels left of no
            # depth-limits
            if levels != 0:
                subcontents = pathCrawler(root_dir, levels-1, f"{subpath}{entry}/")

                # append subcontents to current entries-list if
                # there're any
                if len(subcontents) > 0:
                    entries.extend(subcontents)

        else:

            # the current object is a file
            obj_type = "file"

        # append the current object to the list
        # obj = complete path
        # obj_type = defines the objects type (e.g. "dir")
        entries.append((obj, obj_type))

    return entries

if __name__ == "__main__":
    
    # set the test path, don't end with '/'
    test_path = "C:/abc"
    dest_path = "C:/def/64x64"

    # get the length of the test path
    len_test_path = len(test_path)
    
    # save contents to variable
    contents = pathCrawler(test_path, -1)

    # new dimensions
    dim = (64, 64) # width, height

    # create all subdirs
    for i in range(len(contents)):

        if contents[i][1] == "dir":

            os.makedirs(f"{dest_path}{contents[i][0][len_test_path:]}", exist_ok=True)

    # print all entries
    for i in range(len(contents)):
        
        if contents[i][1] == "file":
            
            img = cv2.imread(contents[i][0], cv2.IMREAD_UNCHANGED)

            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imwrite(f"{dest_path}{contents[i][0][len_test_path:]}", resized)
