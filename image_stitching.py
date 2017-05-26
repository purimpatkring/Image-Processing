# USAGE
# python stitch.py -f img/

from matplotlib import pyplot as plt
import argparse
import imutils
import cv2
import os
import numpy as np

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to the image folder")
args = vars(ap.parse_args())

# Function to load all images in specific folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Function to find Keypoint and match between 2 images
def detect_and_match_keypoint(image_1,image_2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_1,None)
    kp2, des2 = sift.detectAndCompute(image_2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    match_details = []
    image_1_keypoint = []
    image_2_keypoint = []
    for m,n in matches:
        if m.distance < 0.70*n.distance:
            match_details.append([m])
            image_1_keypoint.append(kp1[m.queryIdx])
            image_2_keypoint.append(kp2[m.trainIdx])
    return match_details,np.float32([kp.pt for kp in image_1_keypoint]),np.float32([kp.pt for kp in image_2_keypoint])


# Function to draw matching keypoint
def draw_match_keypoint(image_1_keypoint,image_2_keypoint,image_1,image_2):
    image_1_rows,image_1_cols,n_chan = image_1.shape
    image_2_rows,image_2_cols,n_chan = image_2.shape
    match_image_rows = image_1_rows if image_1_rows > image_2_rows else image_2_rows
    match_image_cols = image_1_cols + image_2_cols
    match_image = np.zeros((match_image_rows, match_image_cols,n_chan),dtype="uint8")
    match_image[0:image_1_rows,0:image_1_cols] = image_1
    match_image[0:image_2_rows,image_1_cols:] = image_2
    for i in range(0,len(image_1_keypoint)):
        point_image_1 = (int(image_1_keypoint[i][0]), int(image_1_keypoint[i][1]))
        point_image_2 = (int(image_2_keypoint[i][0]) + image_1_cols, int(image_2_keypoint[i][1]))
        cv2.line(match_image, point_image_1, point_image_2,(200,150,100), 1)
    return match_image

# Function to use RANSAC to find transform matrix
def merge_image(image_1_keypoint,image_2_keypoint,image_1,image_2):
    #find transform matrix by using RANSAC
    h, status = cv2.findHomography(image_2_keypoint, image_1_keypoint , cv2.RANSAC , 4.0)
    print "H is a transform matrix"
    print h
    #transform coordinate image
    if image_2.shape[0] != image_1.shape[0]:
    	panorama_image = cv2.warpPerspective(image_2, h,(image_2.shape[1] + image_1.shape[1], image_1.shape[0]))
    else:
		panorama_image = cv2.warpPerspective(image_2, h,(image_2.shape[1] + image_1.shape[1], image_2.shape[0]))
    panorama_image[0:image_1.shape[0], 0:image_1.shape[1]] = image_1
    return h,panorama_image


#===============================================
# 	MAIN FUNCTION
#=============================================== 

# Read images from folder
images = load_images_from_folder(args["folder"])

# Initial imageA
image_1_color = images[0]
image_1_color = imutils.resize(image_1_color,height = 800 ,width=400)

# Loop all image
for num_img in range(len(images)-1):
	# Define imageB
	image_2_color = images[num_img+1]
	image_2_color = imutils.resize(image_2_color, height = 800 ,width=400)

	# Convert image to grayscale and find its keypoint
	gray_image1 = cv2.cvtColor(image_1_color, cv2.COLOR_BGR2GRAY)
	gray_image2 = cv2.cvtColor(image_2_color, cv2.COLOR_BGR2GRAY)
	match_details,image_1_keypoint,image_2_keypoint = detect_and_match_keypoint(gray_image1,gray_image2)

	# Merge two image by using RANSAC
	h,panorama_image = merge_image(image_1_keypoint,image_2_keypoint,image_1_color,image_2_color)

	# Get row and column of result image 
	(row,col,channel) = panorama_image.shape

	count_black = 0
	blackC = 0
	# Loop to find black area
	for cols in range(col):
		count_black = 0
		for rows in range(row):
			if np.any(panorama_image[rows,cols] == 0):
				count_black = count_black+1
				# print "row = "+str(rows)+" : " + str(panorama_image[rows,cols])
		if count_black == row:
			blackC = cols 
			# print "col = "+str(cols)+"count_black = "+str(count_black)
			# print "blackC = "+str(blackC)
			break


	# Init new image
	result_new = np.zeros((row,blackC-1,3), np.uint8)

	# Copy result image to new image except black area
	for rows in range(row):
		for cols in range(blackC-1):
			result_new[rows, cols] = panorama_image[rows,cols]

	# Draw matching keypoint
	plt.figure(figsize=(20,20))
	plt.imshow(cv2.cvtColor(draw_match_keypoint(image_1_keypoint,image_2_keypoint,image_1_color,image_2_color), cv2.COLOR_BGR2RGB))
	plt.show()

	# Draw panorama image
	plt.figure(figsize=(20,20))
	plt.imshow(cv2.cvtColor(result_new, cv2.COLOR_BGR2RGB))
	plt.show()

	# Set result image to imageA for combine with next image
	image_1_color = result_new

