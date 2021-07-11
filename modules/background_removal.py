import numpy as np
import cv2
import math

def background_removal(src, model):
    """Returns the given picture without the background. Additionally returns a debug picture

    Parameters
    ----------
    Keyword arguments:
    src -- picture to be modified
    model -- path to the pre-trained structured forest ML model

    code source: https://gist.github.com/Munawwar/ee371009a6ddb58aa96cff9e0391646f
    see also: https://www.codepasta.com/computer-vision/2019/04/26/background-segmentation-removal-with-opencv-take-2.html
    """

    blurred = cv2.GaussianBlur(src, (5, 5), 0)

    blurred_float = blurred.astype(np.float32) / 255.0
    # download model from https://github.com/opencv/opencv_extra/blob/5e3a56880fb115e757855f8e01e744c154791144/testdata/cv/ximgproc/model.yml.gz
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection(model)
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    #cv2.imwrite('edge-raw.jpg', edges)


    def filterOutSaltPepperNoise(edgeImg):
        # Get rid of salt & pepper noise.
        count = 0
        lastMedian = edgeImg
        median = cv2.medianBlur(edgeImg, 3)
        while not np.array_equal(lastMedian, median):
            # get those pixels that gets zeroed out
            zeroed = np.invert(np.logical_and(median, edgeImg))
            edgeImg[zeroed] = 0

            count = count + 1
            if count > 50:
                break
            lastMedian = median
            median = cv2.medianBlur(edgeImg, 3)

    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)
    #cv2.imwrite('edge.jpg', edges_8u)

    def cropToContent(pic, padding=5):

        x_min = len(pic)
        x_max = 0
        y_min = len(pic[0])
        y_max = 0

        for y in range(len(pic)):
            for x in range(len(pic[0])):
                if pic[y][x][0] == 255 and \
                pic[y][x][1] == 255 and \
                pic[y][x][2] == 255:
                    continue
                else:
                    if x < x_min:
                        x_min = x 

                    if x > x_max:
                        x_max = x 

                    if y < y_min:
                        y_min = y 

                    if y > y_max:
                        y_max = y

        content_height = y_max - y_min
        content_width = x_max - x_min

        space_top = 0
        space_left = 0
        square_pixels = 0

        if content_width > content_height:

            # additional padding top
            space_top = math.floor((content_width - content_height) / 2)
            square_pixels = x_max - x_min

        elif content_width < content_height:

            # additional padding left
            space_left = math.floor((content_height - content_width) / 2)
            square_pixels = y_max - y_min

        # calculate padding pixels
        pad_pixels = round(square_pixels * (padding / 100))

        new_pic = []
        new_pic_size = 2 * pad_pixels + square_pixels
        y_padzone = space_top + pad_pixels
        x_padzone = space_left + pad_pixels
        y_max_content = space_top + pad_pixels + content_height
        x_max_content = space_left+pad_pixels+content_width
        y_offset = y_min-space_top-pad_pixels
        x_offset = x_min-space_left-pad_pixels

        for y in range(new_pic_size):

            line_of_pixels = []

            for x in range(new_pic_size):

                if y >= y_padzone and x >= x_padzone and\
                    y <= y_max_content and x <= x_max_content:

                    # add current pixel to line of pixels
                    line_of_pixels.append([pic[y+y_offset][x+x_offset][0],\
                    pic[y+y_offset][x+x_offset][1],\
                    pic[y+y_offset][x+x_offset][2]])

                else:

                    # add white pixel
                    line_of_pixels.append([255,255,255])

            new_pic.append(line_of_pixels)
        
        return np.array(new_pic)

    def findLargestContour(edgeImg):
        contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # From among them, find the contours with large surface area.
        contoursWithArea = []
        for contour in contours:
            area = cv2.contourArea(contour)
            contoursWithArea.append([contour, area])
            
        contoursWithArea.sort(key=lambda tupl: tupl[1], reverse=True)
        largestContour = contoursWithArea[0][0]
        return largestContour

    contour = findLargestContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    #cv2.imwrite('contour.jpg', contourImg)

    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    #cv2.imwrite('trimap.png', trimap_print)

    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')
    #cv2.imwrite('mask2.jpg', mask2)


    contour2 = findLargestContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255.0
    alpha[alpha > 255] = 255.0

    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255.0

    #cv2.imwrite('foreground.png', foreground)
    #cv2.imwrite('background.png', background)
    #cv2.imwrite('alpha.png', alpha)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    cutout = cv2.add(foreground, background)

    picture_content = cropToContent(cutout, 5)

    #cv2.imwrite('cutout.jpg', cutout)
    return [contourImg, np.array(picture_content, dtype="uint8"),picture_content]