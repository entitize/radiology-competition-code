import cv2 as cv
import sys
import skimage
from skimage import measure
import numpy as np
#can try separating filters into different RGB channels?

def preprocess(img, diaphragm = True, filters=True, borders=True, mask=True):
    if diaphragm:
        img = remove_diaphragm(img)
    if filters:
        img = filter_image_from_cv(img)
    if borders:
        img = preprocess_border(img)
    if mask:
        img = preprocess_mask(img)
    return img

def preprocess_double(img1, img2, avg=True, crop=True):
    if avg:
        img1 = preprocess_avg(img1, img2)
    if crop:
        img1 = preprocess_crop(img1, img2)

def remove_diaphragm(img, kernel_size=3, threshold=0.9):
    #diaphragm
    #define threshold of pixel value (diaphragm is close to white - 255)
    t = np.min(img) + threshold * (np.max(img) - np.min(img))
    binary = np.where(img < t, 0, 255)

    #morphological closing to connect pieces together
    #kernel size determines how close the pieces should be for connecting
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    closing = cv.morphologyEx(np.uint8(binary), cv.MORPH_CLOSE, kernel)

    #find the largest white area and mask it
    labels_mask = measure.label(closing)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    mask = labels_mask

        #remove masked area from the original image
    final = np.where(mask == 0, img, 0)

    # imgplot = plt.imshow(final, cmap='gray')
    # plt.show()
    return final

    # R, G, B = cv.split(bilateral)
    # contours, hierarchy= cv.findContours(R.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # sorted_contours= sorted(contours, key=cv.contourArea, reverse= True)
    # largest_item= sorted_contours[0]
    # cv.drawContours(closing, largest_item, -1, color=(0, 0, 0), thickness=cv.FILLED)
    # img2 = cv.merge((R, R, R))
    # return closing



# input image should be 3 channel RGB image with pixel values between 0 and 1
def filter_image_from_generator(img):
    # img = cv.imread(img_path,0)
    bilateral = cv.bilateralFilter(img, 9, 75, 75)
#     img = img * 255
#     bilateral = cv.cvtColor(bilateral, cv.COLOR_BGR2GRAY)
#     bilateral = np.uint8(bilateral)
    R, G, B = cv.split(bilateral)
    output1_R = cv.equalizeHist(np.uint8(R*255))
    output1_G = cv.equalizeHist(np.uint8(G*255))
    output1_B = cv.equalizeHist(np.uint8(B*255))
    equ = cv.merge((output1_R, output1_G, output1_B))
#     equ = equ / 255
#     equ = cv.equalizeHist(bilateral)
    return equ / 255

def filter_image_from_cv(img):
    # img = cv.imread(img_path,0)
    bilateral = cv.bilateralFilter(img, 9, 75, 75)
#     img = img * 255
#     bilateral = cv.cvtColor(bilateral, cv.COLOR_BGR2GRAY)
#     bilateral = np.uint8(bilateral)
    R, G, B = cv.split(bilateral)
    output1_R = cv.equalizeHist(np.uint8(R))
    output1_G = cv.equalizeHist(np.uint8(G))
    output1_B = cv.equalizeHist(np.uint8(B))
    equ = cv.merge((output1_R, output1_G, output1_B))
#     equ = equ / 255
#     equ = cv.equalizeHist(bilateral)
    return equ


def preprocess_border(img):
        (img_height, img_length, channels) = img.shape
        border_chance = 0.05
        x = np.random.uniform(0, 1)
        ret = np.copy(img)
        if (x > border_chance):
            return ret
        # bottom_border = np.random.randint((img_height // 2) + 2, img_height - 2)
        # top_border = np.random.randint(2, (img_height // 2) - 2)
        # right_border = np.random.randint((img_length // 2) + 2, img_length - 2)
        # left_border = np.random.randint(2, (img_length // 2) - 2)

        bottom_border = img_height - 10
        top_border = 10
        right_border = img_length - 10
        left_border = 10

        for i in range(img_height):
            for j in range(img_length):
                for k in range(channels):
                    if (i > bottom_border or i < top_border) or (j > right_border or j < left_border):
                        ret[i][j][k] = 1

        return ret

def preprocess_mask(img1):
    (img1_height, img1_length, channels) = img1.shape
    mask_chance = 0.05
    x = np.random.uniform(0, 1)
    ret = np.copy(img1)
    if (x > mask_chance):
        return ret

    mask_width = np.random.randint(img1_height / 3, img1_height / 2)
    mask_length = np.random.randint(img1_height / 3, img1_length / 2)
    init_location_x = np.random.randint(2, img1_length - mask_width - 2)
    init_location_y = np.random.randint(2, img1_height - mask_length - 2)

    for i in range(img1_height):
        for j in range(img1_length):
            for k in range(channels):
                if (i >= init_location_y and i < init_location_y + mask_width) and (j >= init_location_x and j < init_location_x + mask_length):
                    b = ret[i][j][k]
                    ret[i][j][k] = np.random.uniform(0, b)

    return ret

def preprocess_crop(img1, img2):
    (img1_height, img1_length, channels) = img1.shape
    crop_chance = 0.05
    x = np.random.uniform(0, 1)
    ret = np.copy(img1)
    if (x > crop_chance):
        return ret

    crop_width = np.random.randint(img1_height / 3, img1_height / 2)
    crop_length = np.random.randint(img1_height / 3, img1_length / 2)
    init_location_x = np.random.randint(2, img1_length - crop_width - 2)
    init_location_y = np.random.randint(2, img1_height - crop_length - 2)

    for i in range(img1_height):
        for j in range(img1_length):
            for k in range(channels):
                if (i >= init_location_y and i < init_location_y + crop_width) and (j >= init_location_x and j < init_location_x + crop_length):
                    ret[i][j][k] = img2[i][j][k]

    return ret

def preprocess_avg(img1, img2):
    (img1_height, img1_length, channels) = img1.shape
    avg_chance = 0.05
    x = np.random.uniform(0, 1)
    ret = np.copy(img1)
    if (x > avg_chance):
        return ret

    for i in range(img1_height):
        for j in range(img1_length):
            for k in range(channels):
                ret[i][j][k] = (img2[i][j][k] + img1[i][j][k]) / 2

    return ret





def get_default_augmentation_image_generator_args():
    return {
        "shear_range": 0.2,
        "horizontal_flip": True,
        "width_shift_range": 0.05,
        "height_shift_range": 0.05,
        "rotation_range": 10
    }
