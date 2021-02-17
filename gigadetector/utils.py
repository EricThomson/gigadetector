#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:46:24 2019

@author: eric

Part of gigadetector repo:
https://github.com/EricThomson/gigadetector
"""
import os
from glob import glob
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import logging
logging.basicConfig(level = logging.INFO)

#from imgaug.augmentables.segmaps import SegmentationMapOnImage

def get_bg_aug(aug_random_state):
    lr_flip = iaa.Fliplr(0.5);
    ud_flip = iaa.Flipud(0.5);
    rotate_90 = iaa.Rot90([1, 3], keep_size = False)
    rotate_90_st = iaa.Sometimes(0.5, rotate_90)
    brightness = iaa.Add((-30, 80), random_state = aug_random_state)
    scale = iaa.Affine(scale = (0.75, 1.5), mode = "edge");
    scale_st = iaa.Sometimes(0.9, scale)
    # sharpen/blur
    sharpen = iaa.Sharpen(alpha = (0.1, 0.3), lightness = 1.0)
    gauss_blur = iaa.GaussianBlur(sigma = (1, 2.5));
    no_filt = iaa.Noop()
    blurring = iaa.OneOf([sharpen, gauss_blur, no_filt, no_filt])
    motion_blur = iaa.MotionBlur(k = (10,20), angle = (-90, 90), direction = 0, order = 1 );
    motion_blur_st = iaa.Sometimes(0.15, motion_blur)
    rotate = iaa.Affine(rotate = (-5, 5), mode = "edge");
    rotate_st = iaa.Sometimes(0.9, rotate)
    #Bring together (note don't use 90- degree turn it can really mess with things as it maintains shape)
    bg_augseq = iaa.Sequential([rotate_90_st,
                                lr_flip, ud_flip,
                                brightness,
                                scale_st,
                                blurring,
                                motion_blur_st,
                                rotate_st], random_order = False)
    return bg_augseq
        
    
def get_fg_aug(aug_random_state):
    """
    Don't set brightness that is set manually to be same as bg_image
    Scale is set a little higher because erosion operation tends to make it
    smaller on average: this is worth playing with.
    """
    lr_flip = iaa.Fliplr(0.5);
    ud_flip = iaa.Flipud(0.5);
    rotate_90 = iaa.Rot90([1, 3], keep_size = False)
    rotate_90_st = iaa.Sometimes(0.5, rotate_90)
    #brightness = iaa.Add((-30, 80), random_state = aug_random_state)
    # make fg scale larger because they get eroded
    fg_scale = iaa.Affine(scale = (0.75, 1.5), mode = "edge");
    fg_scale_st = iaa.Sometimes(0.95, fg_scale)
    # sharpen/blur
    sharpen = iaa.Sharpen(alpha = (0.1, 0.3), lightness = 1.0)
    gauss_blur = iaa.GaussianBlur(sigma = (1, 2.5));
    no_filt = iaa.Noop()
    blurring = iaa.OneOf([sharpen, gauss_blur, no_filt, no_filt])
    #motion_blur = iaa.MotionBlur(k = (5, 10), angle = (-90, 90), direction = 0, order = 1 );  #k 10,20
    #motion_blur_st = iaa.Sometimes(0.1, motion_blur)  #was 0.15
    rotate = iaa.Affine(rotate = (-5, 5), mode = "edge");
    rotate_st = iaa.Sometimes(0.9, rotate)
    fg_augseq = iaa.Sequential([rotate_90_st,
                                lr_flip, ud_flip,
                                #brightness,
                                fg_scale_st,
                                blurring,
                                #motion_blur_st,
                                rotate_st], random_order = False)
    return fg_augseq

def cv_imshow(image, name = 'image', shape = (500, 500), xy = (10, 10)):
    """
    helper function for standard cv imshow, resizable window
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, shape[0], shape[1])
    cv2.moveWindow(name, xy[0], xy[1])  #x y pos on screen
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plot_bbs(image, bboxes, size = 2, name = 'image', shape = (500, 500), xy = (10, 10)):
    """
    plot image copy with bboxes (boundingboxesonimage object from imgaug) at xy with shape
    uses cv_imshow
    """
    mn_val = np.mean(image)
    if mn_val >= 127.5:
        color = (0, 0, 0)
    else:
        color = (255, 255, 255)
    if bboxes:
        im_copy = bboxes.draw_on_image(image, color = color, size = size, copy = True);
    else:
        im_copy = image.copy()
    cv_imshow(im_copy, shape = shape, xy = xy, name = name)

    
def cv_loopshow(image, name = 'image', shape = (500, 500), xy = (10, 10)):
    """
    for showing in a for loop: don't include waitkey/destroy all windows here
    as you will put it after the loop
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, shape[0], shape[1])
    cv2.moveWindow(name, xy[0], xy[1])  #x y pos on screen
    cv2.imshow(name, image)
    return


def cv_rotate(image, angle):
    """
    Rotate image by angle in degrees (positive ccw, negative cw)
    Code adapted from https://cristianpb.github.io/blog/image-rotation-opencv
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos_theta = np.abs(M[0, 0])
    sin_theta = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin_theta) + (w * cos_theta))
    nH = int((h * cos_theta) + (w * sin_theta))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def cv_alpha(image, thresh_val = 254):
    """
    Takes in rgba returns rgb with masked values set to 1.
    Handing rgba image: masking out the stuff you actually want masked:
        OpenCV does not handle this well so you need to convert it to rgb
        with a mask.
    returns image with alpha values actually respected (well, white)
    can then use as a mult or some such when combining for augmentation.
    """
    # Pull the fourth channel
    alpha_channel = image[:, :, 3]
    # Set to binary mask (0 when below threshold, 1 above)
    _, mask = cv2.threshold(alpha_channel, thresh_val, 255, cv2.THRESH_BINARY)  # binarize mask
    # New image is going to be just the first three channels (not including a)
    color = image[:, :, :3]
    # Apply mask to color image
    new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    return new_img

def extract_bbs(annotation_path):
    """
    extract and return list of bboxes (numpy arrays xs, ys, xe, ye) given
    path to annotation file in Pascal VOC format.
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    # Use xpath to find all bounding boxes
    all_bndboxes = root.findall('./object/bndbox')
    bboxes = []
    for bndbox in all_bndboxes:
        bbox_coords = []
        for coords in bndbox:  #all children of bbox are coordinates xmin, ymin, xmax, ymax
            bbox_coords.append(int(coords.text))
        bboxes.append(np.array(bbox_coords))
    return bboxes

def extract_imaug_bboxes(annotation_path, shape):
    """ Annotate function here """
    bboxes = extract_bbs(annotation_path)
    bboxes_ia = std2ia(bboxes)
    bboxes_on_image = BoundingBoxesOnImage(bboxes_ia, shape = shape)
    return bboxes_on_image

def std2ia(bboxes):
    """
    create augmenter BoundingBox list from list of standard (numpy) bounding boxes.
    xs, ys, xe, ye -> ia.boundingbox
    """
    num_bboxes = len(bboxes)
    bboxes_ia = []
    for box_ind in range(num_bboxes):
        bbox = bboxes[box_ind]
        bboxes_ia.append(BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3]))
    return bboxes_ia

def cv_bbox(image, bbox, color = (255, 255, 255), line_width = 2):
    """
    use opencv to add bbox to an image using opencv: shortcut for usual cv2 function
    assumes bbox is in standard form xs ys xe ye"""
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_width)
    return

def cv_bboxes(image, bboxes, color = (255, 255, 255), line_width = 2):
    """
    takes in a list of bounding boxes and draws them.
    assumes standard form for each one: xs, ys, xe, ye
    """
    num_bboxes = len(bboxes)
    for bb_ind in range(num_bboxes):
        bbox = bboxes[bb_ind]
        cv_bbox(image, bbox, color = color, line_width = line_width)
    return
   
def cv_bboxes_colors(image, bboxes, colors, line_width = 2):
    """ 
    Draws bboxes with different colors.
    takes in list of bboxes (xs, ys, xe, ye) format, 
    and list of colors, and draws bboxes in those colors.
    """
    for color, bbox in zip(colors, bboxes):
        cv_bbox(image, bbox, color = color, line_width = line_width)
    return

def cv_bboxes_subset(image, bboxes, num_boxes, color = (255, 255, 255), line_width = 2):
    """
    takes in image, and list of std (xs, ys, xe, ye) bboxes, and returns image with 
    bounding box draw from 0 to index-1. Useful when doing image annotation -- see cv_idboxes().
    """
    for bb_ind in range(num_boxes):
        bbox = bboxes[bb_ind]
        cv_bbox(image, bbox, color = color, line_width = line_width)
    return

def cv_circles(image, centers, color = (255, 255, 255), radius = 5, thickness = -1):
    """ 
    Draw filled circle at center positions (list of coordinates)
    in image in given color and size. Thickness of line is -1 for filled circle.
    
    Recall for opencv it uses bgr not rgb for some reason.
    """
    for center in centers:
        cv2.circle(image, tuple(center), radius, color, thickness)
        


def cv_circles_colors(image, centers, colors, radius = 5, thickness = -1):
    """ 
    Draw filled circle at center positions (list of coordinates)
    in image in corresponding color. Thickness of line is -1 for filled circle.
    
    Recall for opencv it uses bgr not rgb for some reason.
    """
    for color, center in zip(colors, centers):
        cv2.circle(image, tuple(center), radius, color, thickness)
        
def cv_text(image, labels, bboxes, xy_rel_pos = (10, -30), 
                color = (255, 255, 255), scale = 2, thickness = 4):
    """
    Take in image, list of labels, list of bounding boxes (and postiions/color) and
    draw the labels on each bbox in the image. Useful for confidence  estimates.
    """
    for ind, label in enumerate(labels):
        bbox = bboxes[ind]
        text_y_pos = bbox[1] + xy_rel_pos[1]
        text_x_pos = bbox[0] + xy_rel_pos[0]
        cv2.putText(image, label, (text_x_pos, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness) 
    return

def cv_text_colors(image, labels, bboxes, colors, xy_rel_pos = (10, -30), 
                   scale = 2, thickness = 4):
    """
    Take in image, list of labels, list of bounding boxes, list of colors, and
    draw the labels on each bbox in the image. Useful for id labels for individual objects
    """
    for color, bbox, label in zip(colors, bboxes, labels):
        text_y_pos = bbox[1] + xy_rel_pos[1]
        text_x_pos = bbox[0] + xy_rel_pos[0]
        cv2.putText(image, str(label), (text_x_pos, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                    scale, color, thickness) 
    return

def copy_to_corrected(small_image, small_mask, large_image, small_image_position):
    """
    Copy small_image (masked using small_mask) into the large_image at position
    small_image_position(row, col).

    small_image_position can contain negative numbers
    if you want to insert above or left of start of large_image (cropping the top left
    part of the small_image accordingly). Similar operations occur if small_image_position
    ends up moving the edges of small_image outside of the boundaries of large_image.
    """
    import warnings
    (large_image_h, large_image_w) = large_image[:,:,1].shape
    (small_image_h, small_image_w) = small_image[:,:,1].shape
    if small_image_position[0] >= large_image_h or small_image_position[1] >= large_image_w:
        raise ValueError(f"Paste position of small image must be within shape of large image {large_image[:,:,1].shape}")
    elif small_image_position[0]+small_image_h < 0 or small_image_position[1]+small_image_w < 0:
        raise ValueError(f"Negative paste position of small image must be within shape of small image {small_image[:,:,1].shape}")

    row_inds = [small_image_position[0], small_image_position[0] + small_image_h]
    col_inds = [small_image_position[1], small_image_position[1] + small_image_w]
    #Correct for negative image position
    if small_image_position[0] < 0:
        warnings.warn("Truncating small image rows: is shifted negative y (rows). Make sure it looks ok.")
        row_inds[1] = small_image_h + small_image_position[0]
        row_inds[0] = 0
        small_mask = small_mask[-small_image_position[0]:, :]
        small_image = small_image[-small_image_position[0]:,:,:]

    if small_image_position[1] < 0:
        warnings.warn("Truncating small image columns: is shifted negative x (cols). Make sure it looks ok.")
        col_inds[1] = small_image_w + small_image_position[1]
        col_inds[0] = 0
        small_mask = small_mask[:, -small_image_position[1]:]
        small_image = small_image[:,-small_image_position[1]:,:]

    #Correct for cases where the shifted small image will go outside the bounds of the large image
    if row_inds[1] > large_image_h:
        warnings.warn("Truncating small image rows: was shifted outside of the large image (height). Make sure it looks ok.")
        row_inds[1] = large_image_h
        num_rows = row_inds[1] - row_inds[0]
        small_mask = small_mask[:num_rows, :]
        small_image = small_image[:num_rows, :, :]

    if col_inds[1] > large_image_w:
        warnings.warn("Truncating small image columns: was shifted outside of the large image (width). Make sure it looks ok.")
        col_inds[1] = large_image_w
        num_cols = col_inds[1] - col_inds[0]
        small_mask = small_mask[:, :num_cols]
        small_image = small_image[:, :num_cols, :]

    cv2.copyTo(small_image,
               small_mask,
               large_image[row_inds[0]: row_inds[1], col_inds[0]: col_inds[1]]);
    return


def cv_annotate(image, annotation_path, color = (255, 255, 255), line_width = 2):
    """
    takes in path to image file, and to file with annotation info (bbox, not label) 
    (Pascal VOC format xml) and displays the annotation on the image
    """
    bbs = extract_bbs(annotation_path)
    cv_bboxes(image, bbs, color = color, line_width = line_width)
    return


def cv_extract_and_annotate(image_path, annotation_path, color = (255, 255, 255),
                         name = 'image',  shape = (500, 500), xy = (0,0), line_width = 2):
    """
    extracts image at image_path, and bboxes at annotation_path, and plots them using opencv
    """
    image = cv2.imread(image_path)
    cv_annotate(image, annotation_path, color = color, line_width = line_width)
    cv_loopshow(image, name = name, shape = shape, xy = xy)
    return

def display_root(root):
    readable_doc = ET.tostring(root, encoding = 'utf8').decode('utf8')
    print(readable_doc)

def display_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    display_root(root)


def make_fish_object(bbox):
    object_elt = ET.Element('object')
    name = ET.SubElement(object_elt, 'name').text = 'fish'
    pose = ET.SubElement(object_elt, 'pose').text = 'Unspecified'
    truncated = ET.SubElement(object_elt, 'truncated').text = str(0)
    difficult = ET.SubElement(object_elt, 'difficult').text = str(0)
    bndbox = ET.SubElement(object_elt, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin').text = str(bbox.x1_int)
    ymin = ET.SubElement(bndbox, 'ymin').text = str(bbox.y1_int)
    xmax = ET.SubElement(bndbox, 'xmax').text = str(bbox.x2_int)
    ymax = ET.SubElement(bndbox, 'ymax').text = str(bbox.y2_int)
    return object_elt


def update_tree(xml_path, aug_dir_path, aug_image_name, aug_image_path, aug_bbs):
    """
    update tree with augmented data: note this uploads the raw image to calculate 
    width and height because the h/w attributes often change during augmentation.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    folder_xml = root.findall('./folder')[0]
    filename_xml = root.findall('./filename')[0]
    path_xml = root.findall('./path')[0]
    all_bndboxes_xml = root.findall('./object/bndbox')
    
    #update xml properties
    folder_xml.text = aug_dir_path
    filename_xml.text = aug_image_name
    path_xml.text = aug_image_path
    
    #Update width/height
    img = cv2.imread(aug_image_path)
    (height, width) = img[:,:,1].shape
    height_node = root.findall('./size/height')[0]
    width_node = root.findall('./size/width')[0]
    #print(height_node[0].tag, height_node[0].attrib, height_node[0].text)
    height_node.text = str(height)
    width_node.text =str(width)

    # update bounding boxes
    if aug_bbs:
        all_bndboxes_aug = aug_bbs.bounding_boxes  #list of BoundingBox classes
        num_bboxes = len(all_bndboxes_aug)
        for bbox_ind in range(num_bboxes):
            bbox_xml = all_bndboxes_xml[bbox_ind]
            bbx_aug = all_bndboxes_aug[bbox_ind]
            for coords in bbox_xml:  #all children of bbox are coordinates xmin, ymin, xmax, ymax
                #print(coords.tag, coords.text)
                if coords.tag == 'xmin':
                    coords.text = str(bbx_aug.x1_int)
                elif coords.tag == 'ymin':
                    coords.text = str(bbx_aug.y1_int)
                elif coords.tag == 'xmax':
                    coords.text = str(bbx_aug.x2_int)
                elif coords.tag == 'ymax':
                    coords.text = str(bbx_aug.y2_int)      
    return tree

def generate_square_bb(bb, scale_factor = 1):
    """
    takes in bb and generates square bb with scale factor depending on long axis,
    with same centroid. bb is BoundingBox type from imgaug
    """
    xc = bb.center_x
    yc = bb.center_y
    width = bb.width
    height = bb.height
    if width > height:
        dist = scale_factor*(width//2)
    elif height > width:
        dist = scale_factor*(height//2)
    else:
        dist = scale_factor*(height//2)
    x1 = xc - dist
    x2 = xc + dist
    y1 = yc - dist
    y2 = yc + dist
    return BoundingBox(x1, y1, x2, y2)

  
   
    
def central_bb(bboxes, shape):
    """
    takes in boundingboxesonimage (from imgaug)
    returns index of bb with minimum distance from center of image
    """
    bounding_boxes = bboxes.bounding_boxes; #list
    num_bboxes = len(bounding_boxes)
    if num_bboxes == 0:
        return 0
    else:
        distances_from_center = []
        image_center = np.array([shape[0]//2, shape[1]//2])
        for bb_ind in range(num_bboxes):
            bb = bounding_boxes[bb_ind]
            bb_center = np.array([bb.center_y, bb.center_x])
            bb_dist = np.linalg.norm(bb_center - image_center)
            distances_from_center.append(bb_dist)
    #return index with minimum value
    return np.where(distances_from_center == np.amin(distances_from_center))[0][0]

    
def get_valid_aug(img_orig, bbs_orig, augmenter):
    """ 
    returns augmented image/bbs ensuring augmented bbs are in the augmented image.
    bbs_orig are BoundingBoxesOnImage type
    
    If there are no bounding boxes, 
    """
    bboxes = bbs_orig.bounding_boxes
    num_bboxes = len(bboxes)       
    if num_bboxes == 0:
        img_aug = augmenter(image = img_orig)
        return img_aug, []
    bbs_accepted = 0
    while bbs_accepted < num_bboxes:
        img_aug, bbs_aug = augmenter(image = img_orig, bounding_boxes = bbs_orig)
        bboxes_aug = bbs_aug.bounding_boxes  #list
        for bb_aug in bboxes_aug:
            if bb_aug.is_fully_within_image(img_aug):
                bbs_accepted += 1
            else:
                bbs_accepted = 0
                break
    return img_aug, bbs_aug
    

def erode_and_smooth(image_mask, erode_kernel, threshold, gauss_width, bb_buffer):
    """
    does erode and smooth operation on image mask, and
    return bounding box of new mask with bb_buffer (expand bounding box around
    the contour by bb_buffer pixels)
    """
    #First erode the mask
    eroded_mask = cv2.erode(image_mask, erode_kernel, iterations = 1)
    mask_blurred = cv2.GaussianBlur(eroded_mask, (gauss_width, gauss_width), 0)
    _, smooth_mask_eroded = cv2.threshold(mask_blurred, threshold, 255, cv2.THRESH_BINARY)

    #Get bounding box of contour of new mask
    contours, _ = cv2.findContours(smooth_mask_eroded, 1, 2)
    
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    mask_contour = contours[max_index]

    x, y, w, h = cv2.boundingRect(mask_contour)
    xs = x - bb_buffer
    ys = y - bb_buffer
    w = w + bb_buffer
    h = h + bb_buffer
    xe = x + w
    ye = y + h
    cv_bb = (xs, ys, xe, ye)
    bounding_box = BoundingBox(cv_bb[0], cv_bb[1], cv_bb[2], cv_bb[3])
    return smooth_mask_eroded, bounding_box


def image_center(image, bbox, center_size):
    """
    Calculate the mean intensity of image at the center_size square in athe middle of bbox
    """
    center =  np.array([int(bbox.center_x), int(bbox.center_y)])
    center_init_coords = center - center_size//2
    center_rect =  [center_init_coords[0], 
                    center_init_coords[1], 
                    center_init_coords[0] + center_size, 
                    center_init_coords[1] + center_size]  #x1, y1, x2, y2
    center_image =  image[center_rect[1]: center_rect[3], center_rect[0]: center_rect[2], :] 
    center_mn = np.mean(center_image)
    return center_image, center_mn, center_rect


def fg_brightness_shift(fg_image, bg_image, fg_mask, fg_bbox, fg_center_size, bg_bbox, bg_center_size):
    """
    get mean difference in brightness between fg and bg images: use to shift brightness to match later.
    """
    bg_center_image, mean_bg_center, _ = image_center(bg_image, bg_bbox, bg_center_size)
    fg_center_image, _, fg_center_rect = image_center(fg_image, fg_bbox, fg_center_size)
    
    # Mask the fg center and get mean of that
    smooth_binary_mask = fg_mask//255 
    smooth_mask_bbox = smooth_binary_mask[fg_center_rect[1]:fg_center_rect[3], fg_center_rect[0]: fg_center_rect[2]]
    mask_3d_init = smooth_mask_bbox[:,:,None]*np.ones(3, dtype = np.uint8)[None, None, :]
    mask_3d = np.abs((mask_3d_init - 1))
    
    masked_fg_center_image = np.ma.masked_array(fg_center_image, mask = mask_3d)
    mean_fg_center = np.ma.mean(masked_fg_center_image)
    fg_bg_diff = mean_bg_center - mean_fg_center
    
    return fg_bg_diff, fg_center_image, bg_center_image, mean_fg_center


def calibrate_fg_bg(fg_image, fg_mask, bg_occluded, fg_image_position, fg_bg_diff):
    """ 
    change brightness of fg image so it matches bg image
    """
    #Shift to be in line
    fg_img_calibrated = np.int16(fg_image + int(fg_bg_diff)) #int16 because there can be negative nums
    fg_img_calibrated[fg_img_calibrated < 0] = 0
    fg_img_calibrated[fg_img_calibrated > 255] = 255
    fg_img_calibrated = np.uint8(fg_img_calibrated)
    
    #copy over to bg_image
    copy_to_corrected(fg_img_calibrated, 
                      fg_mask, 
                      bg_occluded, 
                      fg_image_position);  #position is row, column
            
    return bg_occluded


def extract_paths(directory_name, extension = 'bmp', prefix = '*'):
    """
    Extract list of all paths to files of type extension in directory_name
    """
    file_pattern = prefix + r'*.' + extension
    search_expression = os.path.join(directory_name, file_pattern)  
    return np.sort(glob(search_expression))



def remove_files_in_dir(dir_name, extension = 'bmp', prefix = '*'):
    """
    removes all files in directory. Careful this is not reversible.
    Also, does not remove directories in directory, or hidden files.
    Used for cleaning out test/train folders for data generation.
    """
    all_files = extract_paths(dir_name, extension = extension, prefix = prefix)
    #print(all_files) #for debugging
    for filename in all_files:
        os.remove(filename)
    return True


def bbox_image2std(bbox):
    """
    convert bbox from [ys ye xs xe] to [xs ys xe ye]
    """
    return np.array([bbox[2], bbox[0], bbox[3], bbox[1]])
 
def bbox_std2image(bbox):
    """
    convert bbox from [xs ys xe ye] to [ys ye xs xe]
    """
    return np.array([bbox[1], bbox[3], bbox[0], bbox[2]])

def bbox_std2ocv(bboxes):
    """
    Either takes in single bbox (list) or list of bboxes.
    Converts each box from [xs ys xe ye] to [xs ys w h]
    This is used for opencv bboxes (e.g., non-max suppression (nms) algorithm, 
    which also requires ints)
    """
    #print(bboxes)
    if not isinstance(bboxes[0], list) and not isinstance(bboxes[0], np.ndarray):
        bbox = bboxes
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return [int(bbox[0]), int(bbox[1]), int(w), int(h)]
    else:
        bboxes_converted = []
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bboxes_converted.append([int(bbox[0]), int(bbox[1]), int(w), int(h)])
        return bboxes_converted   
            
def bbox_ocv2std(bbox):
    """
    convert from [xs ys w h] to [xs ys xe ye] 
    The h/w version is used for opencv non-max suppression (nms) algorithm
    """
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def sliding_window(image, stepSize, windowSize):
    """
    Sliding window iterator generates subimages for object detector.
    One wrinkle: all windows must be windowSize or else faster-rcnn
    will give bad performance on the smaller windows. So when it reaches
    an edge (right hand or bottom) it doesn't go past it, but jumps back and
    grabs a full window_size x window_size subimage that buts the edge.
    
    image: height x width pixels
    stepSize: stride (jump size in pixels both in x and then y when it reaches end)
    windowSize : x, y in pixels (width x height) of subimage to grab
    
    yields xstart, ystart, and subimage
    
    This is adapted from code at:
        https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    """
    (H, W) = image.shape
	# slide a window across the image
    for y in range(0, H, stepSize):
        for x in range(0, W, stepSize):
			# yield the current window
            if y + windowSize[1] < H and x + windowSize[0] < W:
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
                
            elif y + windowSize[1] >= H and x + windowSize[0] < W:
                logging.debug("Yield case 2: y out of bounds, x in bounds")
                yield (x, H-windowSize[1], image[H-windowSize[1]:, x:x + windowSize[0]])
                
            elif y + windowSize[1] < H and x + windowSize[0] >= W:
                logging.debug("Yield case 3: y in bounds x out of bounds")
                yield (W-windowSize[0], y, image[y: y+windowSize[1], W-windowSize[0]:])
                
            else:
                logging.debug("Yield final else: both out of bounds")
                yield (W-windowSize[0],  H-windowSize[1], image[-windowSize[1]:, -windowSize[0]:])


def bb_centroid_calc(data, return_type = 'int'):
    """
    data assumed to be 1x4 numpy array with x1, y1, x2, y2
    returns centroid as x,y (note this is reverse of many image coordinates)
    
    Note returns as int for images
    """
    if return_type == 'int':
        x_cent = (data[0]+data[2])//2
        y_cent = (data[1]+data[3])//2
    else:
        x_cent = (data[0]+data[2])/2
        y_cent = (data[1]+data[3])/2
    return (x_cent, y_cent)
 
def bb_intersect(bbox1, bbox2):
    """
    return x,y coordinates of intersection rectangle between bbox1 and bbox2
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    http://ronny.rest/tutorials/module/localization_001/iou/
    bboxes are assumed to be xs, ys, xe, ye
    """
    # Lower bounds
    xs = np.max([bbox1[0], bbox2[0]])
    ys = np.max([bbox1[1], bbox2[1]])
    # Upper bounds
    xe = np.min([bbox1[2], bbox2[2]])
    ye = np.min([bbox1[3], bbox2[3]])
    coord = np.array([xs, ys, xe, ye])
    width = xe - xs
    height = ye - ys
    if width > 0 and height > 0:
        return coord
    else:
        return np.array([])
 
def bb_union(bbox1, bbox2):
    """
    combine bbox1 and bbox2 into a union bbox
    assumes xs, ys, xe, ye (s < e)
    """
    # Lower bounds
    xs = np.min([bbox1[0], bbox2[0]])
    ys = np.min([bbox1[1], bbox2[1]])
    # Upper bounds
    xe = np.max([bbox1[2], bbox2[2]])
    ye = np.max([bbox1[3], bbox2[3]])
    coord = np.array([xs, ys, xe, ye])
    return coord
 
     
def bb_area(bbox):
    """
    returns area: bbox in format xs, ys, xe, ye
    if bbox is empty returns 0
    """
    if bbox.size == 0:
        area = 0
    else:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width*height
    return area
            
def iou(bbox1, bbox2):
    """intersection over union aka the jaccard index"""
    union = bb_union(bbox1, bbox2)
    if union.size == 0:  #check for empty intersect
        int_over_un = 0
    else:
        intersection = bb_intersect(bbox1, bbox2)
        union_area = bb_area(union)
        intersect_area = bb_area(intersection)
        int_over_un = intersect_area/union_area
    return int_over_un 
 
def overlap_to_dist(area, overlap_threshold = 0.1):
    """
    This is a provisional, largely untested, function.
    overlap to distance
        would be nice to add bbox centroids, and have it dedpend on those as well
        when they are not overlapping? Also should take into account the areas
        of the two bounding boxes, and normalize to that (basically intersection over union)
    """
    if area > overlap_threshold:
        dist = 1/np.sqrt(area)
    else:
        dist = 1/overlap_threshold
         
    return dist
 

def bb_filter_initial(bboxes, scores):
   """
   Filter out elts of list that do not contain bbox
   """
   bboxes_initial = []
   scores_initial = []
   num_rois = len(scores)
   for roi_ind in range(num_rois):
       bboxes_roi = bboxes[roi_ind]
       scores_roi = scores[roi_ind]
       if bboxes_roi:       
           for score, bbox in zip(scores_roi, bboxes_roi):
               scores_initial.append(float(score))
               bboxes_initial.append(bbox)
   return bboxes_initial, scores_initial

def bb_filter_initial_features(bboxes, scores, features):
    """
    Filter out elts of list that do not contain bbox
    """
    bboxes_initial = []
    scores_initial = []
    features_initial = []
    num_rois = len(scores)
    for roi_ind in range(num_rois):
        bboxes_roi = bboxes[roi_ind]
        if bboxes_roi:       
            scores_roi = scores[roi_ind]
            features_roi = features[roi_ind]
            for score, bbox, feature in zip(scores_roi, bboxes_roi, features_roi):
                scores_initial.append(float(score))
                bboxes_initial.append(bbox)
                features_initial.append(feature)
    assert len(bboxes_initial) == len(scores_initial), "num boxes not == num scores"
    assert len(bboxes_initial) == len(features_initial), "num boxes not == num features"                
    return bboxes_initial, scores_initial, features_initial

def draw_bboxes_scores(image, bboxes, scores, bb_color = (255, 255, 255),
                       name = 'boxology', xy = (2300, 50), 
                       shape = (800, 800), line_width = 5, 
                       text_thickness = 3):
    """
    Given a list of bounding boxes and scores, and an image, draw them on it, 
    and show image.
    """
    H, W =  image.shape[:2]
    for ind, bbox in enumerate(bboxes):
        score = scores[ind]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bb_color, line_width)
        text_y_pos = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
        text_x_pos = bbox[0] if bbox[0] - 20 < W else bbox[0] - 20
        label = f"{score:0.6f}"
        cv2.putText(image, 
                    label, 
                    (text_x_pos, text_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    bb_color, 
                    text_thickness)
    cv_imshow(image, name = name, xy = xy, shape = shape)
    return

def put_bboxes_scores(image, bboxes, scores, bb_color = (255, 255, 255),
                       line_width = 5, text_thickness = 3):
    """
    Given a list of bounding boxes and scores, and an image, draw them on it,
    but do not show image, just return it.
    """
    H, W =  image.shape[:2]
    for ind, bbox in enumerate(bboxes):
        score = scores[ind]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bb_color, line_width)
        text_y_pos = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
        text_x_pos = bbox[0] if bbox[0] - 20 < W else bbox[0] - 20
        label = f"{score:0.6f}"
        cv2.putText(image, 
                    label, 
                    (text_x_pos, text_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    bb_color, 
                    text_thickness)
    return image

def pairwise_overlap(bboxes):
    """
    Calculate percent overlap matrix: overlap(AB) is percentage of A that is inside B.
    It is asymmetric and can be used to filter out some BBs
    """
    num_boxes = len(bboxes)
    overlap_matrix = np.zeros((num_boxes, num_boxes), dtype = np.float32)
    for row_ind in range(num_boxes):
        for col_ind in range(num_boxes):
            bbox1 = bboxes[row_ind]
            bbox2 = bboxes[col_ind]
            intersect_bbox = bb_intersect(bbox1, bbox2)
            if intersect_bbox.size == 0:
                overlap_matrix[row_ind, col_ind] = 0
                overlap_matrix[col_ind, row_ind] = 0
            else:
                overlap_matrix[row_ind, col_ind] = iou(bbox1, intersect_bbox)
                overlap_matrix[col_ind, row_ind] = iou(bbox2, intersect_bbox)
    return overlap_matrix*100


def pairwise_iou(bboxes):
    """
    Calculate percent overlap matrix: overlap(AB) is percentage of A that is inside B.
    It is symmetric and can be used to filter out some BBs
    """
    num_boxes = len(bboxes)
    iou_matrix = np.zeros((num_boxes, num_boxes), dtype = np.float32)
    for row_ind in range(num_boxes):
        for col_ind in range(num_boxes):
            bbox1 = bboxes[row_ind]
            bbox2 = bboxes[col_ind]
            iou_matrix[row_ind, col_ind] = iou(bbox1, bbox2)
            iou_matrix[col_ind, row_ind] = iou(bbox2, bbox1)
    return iou_matrix*100
  
def overlap_suppression(overlap_matrix, overlap_thresh):
    """
    Return indices of rows with overlap above overlap_thresh with other bbs.
    This will probably be removed.
    """
    indices_to_remove = []
    num_rows, W = overlap_matrix.shape
    for row_ind in range(num_rows):
        row = overlap_matrix[row_ind,:]
        #print(row)
        exceeders = np.where(row > overlap_thresh)[0]
        #print(f"{row_ind}: {exceeders}, {len(exceeders)}")
        if len(exceeders) > 1:
            indices_to_remove.append(row_ind)
    return indices_to_remove


def cv_idboxes(image, bboxes, name):
    """
    Label boxes with correct label. Shows image/bboxes, and enter label in command line.
    To do:
        Make rect green (or whatever) when waiting for number to be entered.
        Zoom in on the area of the image around the bbox
    """
    # sort by first column (xe)
    object_ids = []
    im_shape = image.shape; print(im_shape)
    print('With new red bbox, click n to enter data, and escape or q to quit.')
    #image_copy = image.copy()
    for ind, bbox in enumerate(bboxes):
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 8) #red
        cv_loopshow(image, shape = (950, 950), xy = (2880, 50), name = name)
        k = cv2.waitKey()   
        if k == 27 or k == ord('q'):
            print("Leaving loop")
            break
        elif k == ord('n'):
            cv2.destroyWindow(name)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 9) #red
            cv_loopshow(image, shape = (950, 950), xy = (2880, 50), name = name)
            cv2.waitKey(0)
            object_label = input(f"Enter class for bbox {bbox}: ")
            object_ids.append(object_label)
            text_y_pos = bbox[3] - 30 
            text_x_pos = bbox[0] + 30 
            label = f"{object_label}"
            cv2.putText(image, label, (text_x_pos, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 7)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 10) #black after selection
            cv2.destroyWindow(name)


    cv2.destroyAllWindows()
    print("Done annotating objects")
    return object_ids, image

def cv_idboxes_working(image, bboxes, name = "image", xy = (2000, 50), shape = (950, 950)):
    """
    Click 'n' to trigger keyboard prompt for next input. 
    Active bbox is in green, unlabeled white, labeled black. 
    Click 'p' to go to previous
    
    Repeat for all bboxes.
    """
    num_bboxes = len(bboxes)
    print(f"You have {num_bboxes} boxes. Click n for next, p to correct previous.")
    object_ids = []
    image_working = image.copy()
    cv_bboxes(image_working, bboxes, color = (255, 255, 255), line_width = 9)
    ind = 0
    while ind+1 <= num_bboxes:
        print(f"\tOn box {ind+1} out of {num_bboxes}.")
        bbox = bboxes[ind]
        # Color working one green
        cv2.rectangle(image_working, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 9) #green
        cv_loopshow(image_working, name = name, xy = xy, shape = shape)
        k = cv2.waitKey(0)   
        if k == 27 or k == ord('q'):
            print("Leaving loop")
            break
        elif  k == ord('n'):
            #Get label, and paint bbox black
            object_label = input(f"Enter class for bbox {bbox}: ")
            object_ids.append(object_label)
            cv2.destroyWindow(name)
            text_y_pos = bbox[3] - 30 
            text_x_pos = bbox[0] + 10 
            cv2.putText(image_working, object_label, (text_x_pos, text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            cv2.rectangle(image_working, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 9) #black after selection   
            ind += 1
        elif k == ord('p'):
            # Go back to previous image
            if ind == 0:
                print("Cannot go back from index 0 ya numbskull")
            else:
                ind -= 1
                num_boxes = ind+1
                image_working = image.copy()
                cv_bboxes(image_working, bboxes, color = (255, 255, 255), line_width = 9) #all in white
                cv_bboxes_subset(image_working, bboxes, num_boxes, color = (0, 0, 0), line_width = 9)  #done already in black
                # to do: fill in object labels
                object_ids = object_ids[:ind]
                # cv_text?
                #cv_add_text(image_working, object_ids, bboxes[:ind]) #previous line
                cv_text(image_working, object_ids, bboxes[:ind])
            
    cv_loopshow(image_working, name = name, xy = xy, shape = shape)
    k = cv2.waitKey(0)
    print("DONE ANNOTATING IMAGE!")
    return object_ids, image_working
