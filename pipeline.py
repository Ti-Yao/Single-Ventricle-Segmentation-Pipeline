import pydicom as dicom
import numpy as np
# import os
import time
import pandas as pd
import glob
# from tqdm import tqdm
import cv2
# import json
import math
# import re
import matplotlib.pyplot as plt
import matplotlib.tight_bbox as tight_bbox
import matplotlib.animation as animation
import seaborn as sns
# from IPython.display import HTML
from PIL import Image
# from pathlib import Path
from scipy import stats
import os
# from keras import backend as K
# from keras.layers import Input
# from keras.models import Model
# import pickle
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
from shapely.geometry import Polygon, box, Point, shape, MultiPolygon
from rasterio import features, Affine
from skimage.measure import label
import pdb

min_timesteps = 10
min_slices = 6
min_images = 100
cropper_image_size = 256 # for cropper
segger_image_size = 128
sax_id_image_size = 128

segger_num = 212
blobber_num = 80
sax_id_num = 67

PATH = os.path.dirname(os.path.abspath(__file__))
# sax_id_path = PATH + '/models/SAX-{sax_id_num}.h5'
# blobber_path = PATH + '/models/BLOB-{blobber_num}.h5'
# segger_path = PATH + '/models/SEG-{segger_num}.h5'
sax_id_path = PATH + '/models/SAX-67.h5'
blobber_path = PATH + '/models/BLOB-80.h5'
segger_path = PATH + '/models/SEG-212.h5'


def resize_crop_box(crop_box, crop_factor):
    '''
    rescale square crop box by crop_factor
    '''
    x_min, y_min, x_max, y_max = crop_box
    if y_max - y_min > x_min - x_max:
        largest_side = y_max - y_min
    else:
        largest_side = x_max - x_min
    x_mid = (x_min + x_max)/2
    y_mid = (y_min + y_max)/2
    largest_side = largest_side * crop_factor

    x_max, x_min = round(x_mid + largest_side/2), round(x_mid - largest_side/2)
    y_max, y_min = round(y_mid + largest_side/2), round(y_mid - largest_side/2)
    return x_min, y_min, x_max, y_max

def keep_largest_component(segmentation):
    '''
    keep largest connected component of a mask
    '''
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))* 1
    return largestCC

def get_one_hot(targets, nb_classes):
    '''
    One hot encode segmentation mask
    '''
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def mask_to_polygons_layer(mask):
    '''
    convert mask to polygons
    '''
    all_polygons = []
    for poly, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform= Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shape(poly))

    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.geom_type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def get_five_slices(stack_df):
    '''
    get the centre five slices of a stack
    '''
    unique_slices = sorted(list(stack_df.slicelocation.unique()))
    mid_slice_id = round(len(unique_slices)/2)
    five_slices = []
    for uni_slice_idx in range(mid_slice_id - 2, mid_slice_id + 3):
        five_slices.append(stack_df.loc[stack_df['slicelocation'] == unique_slices[uni_slice_idx]])
    stack_df= pd.concat(five_slices)
    return stack_df


def get_first_phase(stack_df):
    '''
    get the first phase of a stack
    '''
    stack_df = stack_df.sort_values('triggertime')
    unique_slices = sorted(list(stack_df.slicelocation.unique()))
    first_phases = []
    for uni_slice in unique_slices:
        first_phases.append(stack_df.loc[stack_df['slicelocation'] == uni_slice].iloc[0])
    stack_df= pd.concat(first_phases,axis=1).T
    return stack_df


def remove_unconnected_segmentations(masks):
    '''
    postprocessing step, clean up any segmentations outside the heart (the main connected layer)
    '''

    keep_seg_masks = []
    mid_slice_idx = round(masks.shape[2] / 2)
    for channel in range(1, 3):
        keep_seg_masks_time = []
        for time in range(masks.shape[3]):
            heart_mask = np.sum(np.sum(np.sum(masks[..., time, -2:], -1), 0), 0)
            max_heart_idx = np.argmax(heart_mask)
            y_sum = np.sum(masks[..., max_heart_idx, time, 1:], -1)
            sum_polygon = mask_to_polygons_layer(y_sum)
            keep_seg_masks_slice = []
            for pos in range(masks.shape[2]):
                pred_polygons = mask_to_polygons_layer(masks[..., pos, time, channel])
                keep_pred_polygons = []
                for pred_poly in list(pred_polygons.geoms):
                    current_poly = pred_poly
                    criteria = current_poly.intersects(sum_polygon)
                    if criteria:
                        keep_pred_polygons.append(pred_poly)
                if len(keep_pred_polygons) > 0:
                    keep_seg_masks_slice.append(features.rasterize(keep_pred_polygons, out_shape=y_sum.shape))
                else:
                    keep_seg_masks_slice.append(np.zeros_like(masks[..., pos, time, channel]))
            keep_seg_masks_slice = np.stack(keep_seg_masks_slice, -1)
            keep_seg_masks_time.append(keep_seg_masks_slice)
        keep_seg_masks_time = np.stack(keep_seg_masks_time, -1)
        keep_seg_masks.append(keep_seg_masks_time)

    keep_seg_masks = np.stack(keep_seg_masks, -1)
    bkg_mask_array = np.ones(keep_seg_masks.shape[:4]) - np.sum(keep_seg_masks[..., 1:], axis=-1)
    masks = np.concatenate([bkg_mask_array[..., np.newaxis], keep_seg_masks], axis=-1)
    return masks

def remove_basal_apical(masks):
    '''
    gets the minimum from each half and gets the contiguous slices
    '''
    new_masks = []
    for time in range(masks.shape[3]):
        has_segs = np.sum(np.sum(np.sum(masks[...,time,1:],-1),0),0)
        has_segs[has_segs > 0] = 1
        peaks = np.where(has_segs==1)[0]
        ventricle_slices = np.array(sorted(max(np.split(peaks, np.where(np.diff(peaks) != 1)[0]+1), key=len).tolist()))
        if len(ventricle_slices)> 0:
            non_ventricle_slices = np.array(list(set(np.arange(masks.shape[2])).difference(ventricle_slices)))
            if len(non_ventricle_slices)> 0:
                masks[:,:,non_ventricle_slices,time,1:] = 0
                masks[:,:,non_ventricle_slices,time,0] = 1
        new_masks.append(masks[:,:,:,time,:])
    new_masks = np.stack(new_masks,3)
    return new_masks


def find_crop_box(mask, crop_factor):
    '''
    Calculated a bounding box that contains the masks inside
    '''
    x = np.sum(mask, axis=1)
    y = np.sum(mask, axis=0)

    top = np.min(np.nonzero(x)) - 1
    bottom = np.max(np.nonzero(x)) + 1

    left = np.min(np.nonzero(y)) - 1
    right = np.max(np.nonzero(y)) + 1
    if abs(right - left) > abs(top - bottom):
        largest_side = abs(right - left)
    else:
        largest_side = abs(top - bottom)
    x_mid = round((left + right) / 2)
    y_mid = round((top + bottom) / 2)
    half_largest_side = round(largest_side * crop_factor / 2)
    x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side)
    y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side)
    if x_min < 0:
        x_max -= x_min
        x_min = 0

    if y_min < 0:
        y_max -= y_min
        y_min = 0

    return [x_min, y_min, x_max, y_max]


def color_mask_image(img, mask, channels='all'):
    '''
    Adds blood pool and myocardium masks to image
    '''

    if img.ndim == 3:
        img = img[:, :, 0]
    if np.max(img) <= 1:
        img = img * 255
    img = Image.fromarray(img).convert('RGB')
    img = np.array(img)

    num_channels = mask.shape[-1]

    if num_channels == 3:
        bkg = mask[:, :, 0].astype('uint8')
        epi = mask[:, :, 1].astype('uint8')
        endo = mask[:, :, 2].astype('uint8')
        redImg = np.zeros(img.shape, img.dtype)
        redImg[:, :] = (160, 20, 0)
        redMask = cv2.bitwise_and(redImg, redImg, mask=endo)
        img = cv2.addWeighted(redMask, 0.7, img, 1, 0)
        blueImg = np.zeros(img.shape, img.dtype)
        blueImg[:, :] = (0, 20, 160)
        blueMask = cv2.bitwise_and(blueImg, blueImg, mask=epi)
        img = cv2.addWeighted(blueMask, 0.7, img, 1, 0)
    elif num_channels == 2:
        bkg = mask[:, :, 0].astype('uint8')
        epi = mask[:, :, 1].astype('uint8')
        color = (160, 20, 0) if channels == 'endo' else (0, 20, 160)
        colorImg = np.zeros(img.shape, img.dtype)
        colorImg[:, :] = color
        colorMask = cv2.bitwise_and(colorImg, colorImg, mask=epi)
        img = cv2.addWeighted(colorMask, 0.7, img, 1, 0)

    return img


def convert_to_4d(im,position):
    '''
    convert the output of the segmentation model into the form of (h,w,slice,time,channel)
    '''
    image_4d = []
    for i in range(position):
        image_4d.append(np.array(list(im[i::position])))
    image_4d = np.moveaxis(image_4d, 0,-2)
    image_4d = np.moveaxis(image_4d, 0,-2)
    return image_4d


def normalize(image):
    '''
    standardise image such that the mean is 0 and the variance is 1
    '''
    mean = np.mean(image)
    std = np.std(image)
    if std != 0:
        norm = (image - mean) / std
    else:
        norm = np.zeros_like(image)
    return norm


def reverse_through_timesteps(y, func, **kwargs):
    '''
    applies a function through each timestep in an image (for converting the segmentation mask to original size)
    '''
    new_y = []
    timesteps = y.shape[3]
    for time in range(timesteps):
        new_y.append(func(y[:,:,:,time,:].copy(), **kwargs))
    new_y = np.stack(new_y, axis = 3)
    return new_y

def reverse_pad(y, original_shape, scale_shape):
    '''
    reverses the padding used in the crop function (for converting the segmentation mask to original size)
    '''
    # scale_shape = (p.image.shape[:2] /np.max(p.image.shape[:2]))* 400
    if scale_shape[0] != cropper_image_size:
        unpad_amount = round((y.shape[0] - scale_shape[0])/2)
        y = y[unpad_amount:unpad_amount + scale_shape[0],:,:,:]
    if scale_shape[1] != cropper_image_size:
        unpad_amount = round((y.shape[1] - scale_shape[1])/2)
        y = y[:,unpad_amount:round(unpad_amount + scale_shape[1]),:,:]
    return y

def reverse_resize(y, old_shape):
    '''
    reverses the resizing for the segmentation step (for converting the segmentation mask to original size)
    '''
    if y.ndim == 2:
        y = np.array(
                    tf.image.resize(y[:,:,np.newaxis],
                    old_shape,
                    method = 'nearest'
                    ))[:,:,0]
    elif y.ndim == 3:
        y = np.array(
                tf.image.resize(y,
                old_shape,
                    method = 'nearest'
                ))
    elif y.ndim == 4:
        y = np.moveaxis(y, 2,0)
        y = np.array(
                tf.image.resize(y,
                old_shape,
                    method = 'nearest'
                ))
        y = np.moveaxis(y, 0,2)
    return y

def reverse_crop(y_crop, crop_box, original_shape):
    '''
    reverses the crop function (for converting the segmentation mask to original size)
    '''
    x_min,y_min , x_max, y_max = crop_box
    if y_crop.ndim == 2:
        y = np.zeros(original_shape[:2])
        y [y_min:y_max, x_min:x_max] = y_crop
    elif y_crop.ndim == 3:
        x_min, x_max, y_min, y_max = crop_box
        bkg = np.ones(original_shape[:2] + (1,))
        myo_endo = np.zeros(original_shape[:2] + (2,))
        y = np.concatenate([bkg, myo_endo], axis = -1)
        y [y_min:y_max, x_min:x_max, 0] = y_crop[:,:,0] # bkg
        y [y_min:y_max, x_min:x_max, 1] = y_crop[:,:,1] # myo
        if y.shape[-1] == 3:
            y [y_min:y_max, x_min:x_max, 2] = y_crop[:,:,2] # endo
    elif y_crop.ndim == 4:
        bkg = np.ones(original_shape[:3] + (1,))
        if y_crop.shape[-1] == 3:
            myo_endo = np.zeros(original_shape[:3] + (2,))
        elif y_crop.shape[-1] == 2:
            myo_endo = np.zeros(original_shape[:3] + (1,))
        y = np.concatenate([bkg, myo_endo], axis = -1)
        y [y_min:y_max, x_min:x_max, :, 0] = y_crop[:,:, :,0] # bkg
        y [y_min:y_max, x_min:x_max, :, 1] = y_crop[:,:, :,1] # myo
        if y.shape[-1] == 3:
            y [y_min:y_max, x_min:x_max, :, 2] = y_crop[:,:, :,2] # endo
    return y


class Pipeline:
    def __init__(self, patient, data_path, results_path='results'):
        self.patient = patient
        self.data_path = data_path
        self.path = data_path + '/' + patient + '/'#f'{data_path}/{patient}/'
        self.results_path = results_path

        self.get_dicoms() # get all the dicom files for each patient
        self.dicom_info = self.get_dicom_info() # read dicom headers for each file into a dataframe called dicom_info

        self.start_time = time.time()

        self.stack_df_list = self.get_stack_df_list(self.dicom_info) # find the cine stacks
        self.sax_df = self.get_sax_df(self.stack_df_list) # using classifier find which cine stack is sax
        self.sax_df = self.clean_sax_df(self.sax_df) # clean the sax data e.g. remove repeated scans
        self.single_dicom = dicom.dcmread(self.sax_df.reset_index().iloc[0].dicom) # get patient information from single dicom
        self.voxel_size = self.get_voxel_size() # calculate size of the voxel
        self.image = self.get_sax_image()# create the sax image
        self.cropped_image, self.crop_box = self.blob_image(self.image)  # crop heart out of image
        self.segged_image, self.masks = self.seg_image(self.cropped_image) # segment the heart
        self.volume, self.mass, self.esv, self.edv, self.sv, self.ef = self.get_metrics(self.masks)

        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time
        
        self.make_video(self.segged_image,'segs') # create gif of segmented image
        self.wgif_end_time = time.time()
        self.wgif
        self.save_sys_dia_image(self.segged_image) # save the systolic and diastolic segmented image

    def get_dicoms(self):
        '''
        find all the dicom files in the folders
        '''
        series_list = [series for series in os.listdir(self.path) if
                       os.path.isdir(os.path.join(self.path, series)) and 'wsx' not in series]

        dicom_files = {}
        new_series_list = []
        for series in series_list:
            series_path = f'{self.path}/{series}/'
            dicoms_in_series = sorted(glob.glob(f"{series_path}/*.dcm") + glob.glob(f"{self.path}/*.dcm"))
            dicom_files[series] = dicoms_in_series
            new_series_list.append(series)
        if len(dicom_files) == 0:
            self.status = 'no_dicoms'
            raise ValueError(f'{self.patient} has no  dicom files, please check')

        self.series_list = sorted(new_series_list)
        self.dicom_files = dicom_files

    def get_dicom_info(self):
        '''
        puts all the dicom header information for ALL dicoms into a dataframe
        '''

        dicom_info = {}
        remove_series = []
        for series in self.series_list:
            dicoms_in_series = self.read_dicom_header(series)
            if len(dicoms_in_series) > min_timesteps:  # number in series has to be greater than minimum timesteps
                dicom_info.update(dicoms_in_series)
            else:
                remove_series.append(series)
        self.series_list = sorted(list(set(self.series_list).difference(
            set(remove_series))))  # remove any series that doesn't have minimum number of timesteps
        dicom_info = pd.DataFrame.from_dict(dicom_info, orient='index').reset_index().rename(
            columns={'index': 'dicom'})  # put dicom info for all images into a dataframe
        dicom_info['image_shape'] = dicom_info.image.apply(lambda x: x.shape)
        dicom_info = dicom_info.loc[dicom_info['phase'] == 0]  # remove flow images
        dicom_info = dicom_info[dicom_info['triggertime'].notna()]  # remove scans with no triggertimes
        if dicom_info.slicelocation.isnull().any():
            main_axis = np.argmax(
                np.cross(dicom_info['orientation'].iloc[0][:3], dicom_info['orientation'].iloc[0][3:]))
            dicom_info['slicelocation'] = dicom_info['position'].apply(lambda x: x[main_axis])
        return dicom_info

    def read_dicom_header(self,series):
        '''
        read the information we want from the header and assert that the series has to have pixelarray data
        '''
        dicoms_in_series = self.dicom_files[series]
        dicom_info = {}
        for dicom_path in dicoms_in_series: # go through dicom in each series
            dcm = dicom.dcmread(dicom_path) # read dicom
            if 'Image Storage' in dcm.SOPClassUID.name:
                image_exists = False
                try:
                    image = dcm.pixel_array
                    if image.ndim == 3: 
                        image_exists = False
                    else:
                        image_exists = True
                except Exception as e:
                    try:
                        image_bytes = tf.io.read_file(dicom_path)
                        image = tfio.image.decode_dicom_image(image_bytes,dtype=tf.uint16).numpy()[0]
                        image_exists = True
                    except Exception as e:
                        print(e)
                        image_exists = False
                try:
                    if dcm.MRAcquisitionType == '3D': # ignore dicom if 3d
                        image_exists = False
                except Exception as e:
                    image_exists = False
            else:
                image_exists = False
            if image_exists: # if image exists and is not 3d read all other information
                dicom_info[dicom_path] = {}
                dicom_info[dicom_path]['image'] = image
                dicom_info[dicom_path]['uid'] = dcm.SOPInstanceUID

                # have to use try and excepts, if the dicom doesn't the information stored use nan
                try:
                    dicom_info[dicom_path]['slicelocation'] = round(dcm.SliceLocation,3)
                except:
                    dicom_info[dicom_path]['slicelocation'] = np.nan
            
                try:
                    dicom_info[dicom_path]['thickness'] = round(dcm.SpacingBetweenSlices,3)
                except:
                    try:
                        dicom_info[dicom_path]['thickness'] = round(dcm.SliceThickness,3)
                    except:
                        dicom_info[dicom_path]['thickness'] = np.nan
                try:
                    dicom_info[dicom_path]['series_uid'] = dcm.SeriesInstanceUID
                except:
                    dicom_info[dicom_path]['series_uid'] = np.nan
                try:
                    dicom_info[dicom_path]['seriesnumber'] = dcm.SeriesInstanceUID
                except:
                    dicom_info[dicom_path]['seriesnumber'] = np.nan
                try:
                    dicom_info[dicom_path]['triggertime'] = round(dcm.TriggerTime)
                except:
                    dicom_info[dicom_path]['triggertime'] = np.nan
                try:
                    dicom_info[dicom_path]['N_timesteps'] = int(dcm.CardiacNumberOfImages)
                except:
                    dicom_info[dicom_path]['N_timesteps'] = 0
                try:
                    dicom_info[dicom_path]['orientation'] = [round(val,3) for val in dcm.ImageOrientationPatient]
                except:
                    dicom_info[dicom_path]['orientation'] = np.nan
                try:
                    dicom_info[dicom_path]['position'] = [round(val,3) for val in dcm.ImagePositionPatient]
                except:
                    dicom_info[dicom_path]['position'] = np.nan
                try:
                    dicom_info[dicom_path]['pixelspacing'] = round(dcm.PixelSpacing[0],3)
                except:
                    dicom_info[dicom_path]['pixelspacing'] = np.nan
                try:
                    dicom_info[dicom_path]['phase'] = dcm[0x0028, 0x1052].value
                except:
                    try:
                        dicom_info[dicom_path]['phase'] = list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueIntercept 
                    except:
                        dicom_info[dicom_path]['phase'] = 0
        
        return dicom_info

    def get_stack_df_list(self, dicom_info):
        '''
        using header information find the stacks.
        images from the same stack are images that have the same orientation, pixelspacing, thickness and image shape
        and have more than the minimum timesteps number of timesteps
        '''
        if len(dicom_info) < min_slices * min_timesteps:
            self.status = 'no stacks'
            raise ValueError('no stacks')

        stack_df_list = []
        unique_orientations = np.unique(dicom_info.orientation.dropna().values)
        for unique_or in unique_orientations:
            unique_df = dicom_info.loc[dicom_info['orientation'].apply(lambda x: x == unique_or)]  # choose orientation
            unique_df = unique_df.set_index(
                ['pixelspacing', 'thickness', 'N_timesteps']).sort_index()  # sort pixel spacing and thickness
            for unique_idx in unique_df.index.unique():  # going through all the unique orientation+pixelspacing+thickness
                stack_df = unique_df.loc[unique_idx]
                if isinstance(stack_df, pd.DataFrame):  # checking if there is more than one result
                    unique_image_shapes = np.unique(stack_df.image_shape.values)
                    for unique_imshape in unique_image_shapes:  # making sure all the images are of the same shape
                        stack_unique_image_shape_df = stack_df.loc[
                            stack_df['image_shape'] == unique_imshape].reset_index()
                        if isinstance(stack_unique_image_shape_df, pd.DataFrame) and len(
                                stack_unique_image_shape_df) > 1:
                            separated = False
                            for uni_series in stack_unique_image_shape_df.seriesnumber.unique():  # going through each series
                                series_df = stack_unique_image_shape_df.loc[
                                    stack_unique_image_shape_df['seriesnumber'] == uni_series]
                                if len(series_df) > 50:  # separate large series, sometimes long axis and short-axis will have the same orientation
                                    if self.is_sax_valid(series_df):
                                        stack_df_list.append(series_df)
                                        separated = True
                            if not separated:
                                if self.is_sax_valid(stack_unique_image_shape_df) and len(
                                        stack_unique_image_shape_df) > min_timesteps * min_slices:
                                    stack_df_list.append(stack_unique_image_shape_df)
        # print(f'num stack = {len(stack_df_list)}')
        if len(stack_df_list) == 0:
            self.status = 'no_stacks'
            raise ValueError('no stacks')
        else:
            self.status = 'has stacks'
        return stack_df_list

    def get_sax_df(self, stack_df_list):
        '''
        uses the sax id classifier model to find which of the stacks is sax.
        if there is more than one sax stack, we choose the one with more images.
        '''
        sax_id_model = tf.keras.models.load_model(sax_id_path, compile=False)

        sax_max_probs = []
        sax_mean_probs = []
        
        for stack_idx in range(len(stack_df_list)):  # going through each stack
            stack_df = stack_df_list[stack_idx]
            stack_df = stack_df.sort_values(['slicelocation','triggertime'])
            stack_df = self.cine_series(stack_df)  # make sure each stack has minimum number of timesteps
            stack_df = stack_df.sort_values(['slicelocation', 'triggertime'])
            if len(stack_df) > 0 and self.is_sax_valid(stack_df):  # make sure the stack is valid
                N_slices = stack_df.slicelocation.nunique()
                if N_slices > min_slices:
                    stack_df = get_five_slices(stack_df)
                    stack_df = get_first_phase(stack_df)
                    stack_df.loc[:, 'image'] = stack_df.loc[:, 'image'].map(lambda x: normalize(np.array(
                        tf.image.resize(x[..., np.newaxis], [sax_id_image_size, sax_id_image_size],
                                        method='nearest'))))
                    images_to_classify = np.stack(stack_df.image.values, axis=0)
                    probs = sax_id_model.predict(images_to_classify[..., np.newaxis])
                    mean_probs = probs.mean()  # get the mean probability of all the images classified that it is a sax
                    max_probs = round(probs.max(), 2)
                    sax_mean_probs.append(mean_probs)
                    sax_max_probs.append(max_probs)
                else:
                    sax_mean_probs.append(0)
                    sax_max_probs.append(0)
            else:
                sax_mean_probs.append(0)
                sax_max_probs.append(0)
                
        if len(sax_mean_probs) > 0:
            if np.max(sax_mean_probs) == 0:  # if all the stacks return 0 probability that it is a sax, there is no sax
                self.status = 'no SAX'
                raise ValueError('no sax')
                return None
            else:
                self.status = 'has SAX'
                self.sax_max_probs = sax_max_probs
                self.sax_mean_probs = sax_mean_probs
                if len(np.where(sax_max_probs == np.max(sax_max_probs))[0]) > 1:
                    sax_probs = np.array(sax_mean_probs)
                    sax_probs[sax_max_probs != np.max(sax_max_probs)] = 0
                else:
                    sax_probs = np.array(sax_max_probs)
                self.sax_probs = sax_probs
                threshold = 0.3 # minimum sax probability
                filtered_indices = np.where(sax_probs >= threshold)[0]
                sax_id = filtered_indices[np.argsort(sax_probs[filtered_indices])[::-1]]
                valid = False
                sax_id_cnt = 0
                sax_df = None
                while valid == False and sax_id_cnt < len(sax_id): # from the highest to lowest probability, get the first valid stack
                    sax_df = stack_df_list[sax_id[sax_id_cnt]] 
                    valid = self.is_sax_valid(sax_df)
                    sax_id_cnt += 1
                    if sax_id_cnt == len(sax_id):
                        break
                if isinstance(sax_df, pd.DataFrame):
                    return sax_df.reset_index().sort_values(['slicelocation','triggertime'])
                else:
                    raise ValueError('no SAX')
                    


    def is_sax_valid(self, sax_df):
        '''
        checks if a stack is valid as a sax by saying that it has the greater than the minimum number of timesteps and slices
        '''
        N_timesteps = int(self.calc_N_timesteps(sax_df))  # calculate the number of timesteps
        self.N_timesteps = N_timesteps
        N_slices = sax_df.slicelocation.nunique()
        self.N_slices = N_slices
        if N_slices >= min_slices and N_timesteps >= min_timesteps and len(sax_df) >= min_images and len(
                sax_df) % N_timesteps == 0:
            sax_valid = True
        else:
            sax_valid = False
        return sax_valid

    def clean_sax_df(self, sax_df):
        '''
        After obtaining the sax, might need to do some cleaning, making sure each series has the same number of timesteps
        Removing any repeated scans, e.g. they may scan the same slice more than once, if it's not very good quality.
        '''
        sax_df = sax_df.drop_duplicates(subset='uid')
        # total number of images should be equal to the number of timesteps multiplied by number of slices
        if len(sax_df) != self.N_timesteps * sax_df.slicelocation.nunique():
            sax_df = self.remove_repeated_scans(sax_df)
        if len(sax_df) != self.N_timesteps * sax_df.slicelocation.nunique():
            sax_df = self.same_timesteps(sax_df)
        N_slices = sax_df.slicelocation.nunique()
        # recalculate the number of slices and timesteps after cleaning
        self.N_slices = N_slices
        N_timesteps = sax_df.N_timesteps.unique()[0]
        if np.isnan(N_timesteps):
            N_timesteps = int(self.calc_N_timesteps(sax_df))
        else:
            N_timesteps = int(N_timesteps)
        self.N_timesteps = N_timesteps
        if self.is_sax_valid(sax_df):
            return sax_df.sort_values(['slicelocation', 'triggertime'])
        else:
            raise ValueError('Mismatched timesteps')

    def cine_series(self, stack_df):
        '''make sure each series has at least minimum number of timesteps'''
        series_df_list = []
        if not all(stack_df.seriesnumber.isna()):
            for uni_series in stack_df.seriesnumber.unique():
                series_df = stack_df.loc[stack_df['seriesnumber'] == uni_series]
                if len(series_df) > min_timesteps:
                    series_df_list.append(series_df)
            if len(series_df_list) >= 1:
                stack_df = pd.concat(series_df_list)
            else:
                stack_df = pd.DataFrame()
        return stack_df

    def same_timesteps(self, sax_df):
        '''make sure each series has same number of timesteps'''
        if sax_df.seriesnumber.nunique() > 1:
            series_df_list = []
            for uni_series in sax_df.seriesnumber.unique():
                series_df = sax_df.loc[sax_df['seriesnumber'] == uni_series]
                if series_df.triggertime.nunique() == self.N_timesteps:
                    series_df_list.append(series_df)
            if len(series_df_list) > 1:
                sax_df = pd.concat(series_df_list)
            else:
                sax_df = pd.DataFrame()
        return sax_df

    def calc_N_timesteps(self, sax_df):
        '''
        N timesteps is given in the dicom header as number cardiac images, but it's not always there.
        This calculates the number of timesteps there should be in a series by taking the modal value of the
        number of trigger times for each series.
        '''
        unique_slices = sax_df.slicelocation.unique()
        possible_N_timesteps = []
        for uni_slice in unique_slices:
            possible_N_timesteps.append(len(sax_df.loc[sax_df['slicelocation'] == uni_slice]))
        # N_timesteps = stats.mode(possible_N_timesteps, keepdims=False)[0]
        N_timesteps = stats.mode(possible_N_timesteps)[0]
        return N_timesteps

    def remove_repeated_scans(self,sax_df):
        '''
        If one slice location has more images than others, it means there has been a repeated scan.
        For the slices with repeated scans, take the later triggertime
        '''
        new_sax_df_list = []
        unique_slices = sax_df.slicelocation.unique()
        sax_df = sax_df.drop_duplicates(subset = 'uid')
        for uni_slice in sax_df.slicelocation.unique():
            slice_df = sax_df.loc[sax_df['slicelocation'] == uni_slice]
            if slice_df.seriesnumber.nunique() > 1:
                slice_df = slice_df.loc[slice_df['seriesnumber'] == np.max(slice_df.seriesnumber.unique())]
                slice_df = slice_df.drop_duplicates(subset = 'triggertime').sort_values('triggertime')
            new_sax_df_list.append(slice_df)
        if len(new_sax_df_list) > 0:
            sax_df = pd.concat(new_sax_df_list).sort_values(['slicelocation','triggertime'])
            return sax_df
        else:
            return sax_df
            
            
    def get_voxel_size(self):
        dcm = self.single_dicom
        pixel_size = dcm.PixelSpacing[0] ** 2
        try:
            thickness = dcm.SpacingBetweenSlices
        except:
            thickness = dcm.SliceThickness
        voxel_size = pixel_size * thickness
        return voxel_size / 1000

    def get_sax_image(self):
        '''
        makes the 4D sax image image[height, width, slice, time]
        '''
        try:
            image_4D = []
            images = self.sax_df.sort_values(['slicelocation', 'triggertime']).image.values
            for uni_slice in self.sax_df.slicelocation.unique():
                image_4D.append(
                    np.stack(self.sax_df.loc[self.sax_df['slicelocation'] == uni_slice].image.values, axis=-1))
            image_4D = np.stack(image_4D, axis=-2)
        except:
            self.status = 'Mismatched timesteps'
            raise ValueError('Mismatched timesteps')
        return image_4D

    def make_video(self, image_4D, video_name='segs'):
        '''
        plots a gif of the input image. can be the sax image, the cropped image or the segmented image.
        '''
        position = image_4D.shape[2]
        timesteps = image_4D.shape[3]
        num_tiles = position

        # Now find rows and columns for this aspect ratio.
        grid_rows = int(np.sqrt(num_tiles) + 0.5)  # Round.
        grid_cols = (num_tiles + grid_rows - 1) // grid_rows  # Ceil.

        row_cols = math.ceil(np.sqrt(position))
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))

        frames = []
        Path(f"{self.results_path}/{video_name}").mkdir(parents=True, exist_ok=True)
        plt.gcf().set_facecolor("white")
        for time in range(timesteps):
            ttl = plt.text(0.5, 1.01, f'timestep = {time + 1}/{timesteps}', horizontalalignment='center',
                           verticalalignment='bottom', transform=axes[0, 0].transAxes, fontsize="large")
            artists = [ttl]
            for row, col in np.ndindex(grid_rows, grid_cols):
                axes[row, col].axis('off')
                axes[row, col].patch.set_facecolor('white')
                pos = row * grid_cols + col
                if pos < position:
                    if image_4D.ndim == 5:
                        artist = axes[row, col].imshow(image_4D[:, :, pos, time, :])
                    else:
                        artist = axes[row, col].imshow(image_4D[:, :, pos, time], cmap='gray')
                    artists.append(artist)
            frames.append(artists)
        bbox_inches = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox_inches = bbox_inches.padded(0.1)
        tight_bbox.adjust_bbox(fig, bbox_inches)
        fig.set_size_inches(bbox_inches.width, bbox_inches.height)
        ani = animation.ArtistAnimation(fig, frames)
        ani.save(f"{self.results_path}/{video_name}/{self.patient}.gif", fps=round(timesteps / 2))
        plt.close()

    def save_sys_dia_image(self, image_4D):
        position = image_4D.shape[2]
        num_tiles = position

        for i, time in enumerate([self.dia_idx, self.sys_idx]):

            # Now find rows and columns for this aspect ratio.
            grid_rows = int(np.sqrt(num_tiles) + 0.5)  # Round.
            grid_cols = (num_tiles + grid_rows - 1) // grid_rows  # Ceil.

            row_cols = math.ceil(np.sqrt(position))
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
            for row, col in np.ndindex(grid_rows, grid_cols):
                axes[row, col].axis('off')
                axes[row, col].patch.set_facecolor('white')
                pos = row * grid_cols + col
                if pos < position:
                    if image_4D.ndim == 5:
                        artist = axes[row, col].imshow(image_4D[:, :, pos, time, :])
                    else:
                        artist = axes[row, col].imshow(image_4D[:, :, pos, time], cmap='gray')
            Path(f"{self.results_path}/sys_dia_plot").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{self.results_path}/sys_dia_plot/{self.patient}_{'sys' if i == 1 else 'dia'}.png",
                        bbox_inches='tight')
            plt.close()

    def blob_image(self, image_4D, blobber_num=blobber_num):
        blobber = tf.keras.models.load_model(blobber_path, compile=False)

        position = image_4D.shape[2]
        mid_slice_idx = int(position / 2)
        timesteps = image_4D.shape[3]
        images_to_crop = []
        for pos in range(position):  # take the quarter
            time = 0
            im = image_4D[..., pos, time].copy()
            im = im[..., np.newaxis]
            scale_shape = (im.shape[:2] / np.max(im.shape[:2])) * cropper_image_size
            self.scale_shape = [round(shape) for shape in scale_shape]
            scale_shape = np.array([round(s) for s in scale_shape])
            im = np.array(tf.image.resize(im, scale_shape, method='nearest'))
            im = tf.image.resize_with_crop_or_pad(im, cropper_image_size,
                                                  cropper_image_size)  # pad image to make sure it can go in the model without stretching it
            im = np.array(im)
            im = normalize(im[..., 0])
            images_to_crop.append(im[..., np.newaxis])
        images_to_crop = np.stack(images_to_crop, 0)
        blobs = blobber.predict(images_to_crop)
        if type(blobs) == list:
            blobs = blobs[-1]
        blobs = get_one_hot(np.argmax(blobs, axis=-1), 2)  # binarise each mask
        blobs = blobs[..., -1]  # get the mask not bkg
        y_sum = np.sum(blobs, 0)  # flatten masks together
        y_sum[y_sum < y_sum.max()] = 0
        y_sum[y_sum == y_sum.max()] = 1
        sum_polygon = mask_to_polygons_layer(keep_largest_component(y_sum))
        keep_blobs = np.zeros_like(y_sum)
        for i in range(len(blobs)):
            pred_polygons = mask_to_polygons_layer(blobs[i])
            keep_pred_polygons = []
            for pred_poly in list(pred_polygons.geoms):
                if pred_poly.intersects(sum_polygon):
                    keep_pred_polygons.append(pred_poly)
            if len(keep_pred_polygons) > 0:
                keep_blobs += features.rasterize(keep_pred_polygons, out_shape=y_sum.shape)
        keep_blobs[keep_blobs >= 1] = 1
        keep_blobs[keep_blobs < 1] = 0
        keep_blobs = keep_largest_component(keep_blobs)
        try:
            # plt.imshow(image_4D[...,mid_slice_idx,0], cmap = 'gray')
            # plt.imshow(keep_blobs, alpha = (keep_blobs/np.max(keep_blobs)) * 0.6)
            # Path(f"{self.results_path}/blobs").mkdir(parents=True, exist_ok=True)
            # plt.savefig(f"{self.results_path}/blobs/{self.patient}.jpg")
            # plt.close()
            x_min, y_min, x_max, y_max = find_crop_box(keep_blobs, crop_factor=1.5)
            if np.argmin(scale_shape) == 1 and x_min < (
                    cropper_image_size - np.min(scale_shape)) / 2:  # means x is overflowing
                x_min, x_max = round(x_min + ((cropper_image_size - np.min(scale_shape)) / 2 - x_min)), round(
                    x_max + ((cropper_image_size - np.min(scale_shape)) / 2 - x_min))
            elif np.argmin(scale_shape) == 0 and y_min < (cropper_image_size - np.min(scale_shape)) / 2:
                y_min, y_max = round(y_min + ((cropper_image_size - np.min(scale_shape)) / 2 - y_min)), round(
                    y_max + ((cropper_image_size - np.min(scale_shape)) / 2 - y_min))

            pad_images = []  # pad sax image
            for time in range(image_4D.shape[-1]):
                im = image_4D[..., time].copy()
                scale_shape = (im.shape[:2] / np.max(im.shape[:2])) * cropper_image_size
                im = np.array(tf.image.resize(im, scale_shape, method='nearest'))
                pad_images.append(tf.image.resize_with_crop_or_pad(im, cropper_image_size, cropper_image_size))
            pad_images = np.stack(pad_images, -1)
            cropped_image = pad_images[y_min:y_max, x_min:x_max, ...]  # crop sax image using bounding box
            crop_box = (x_min, y_min, x_max, y_max)
            # self.status = 'crop success'
            return cropped_image, crop_box
        except:  # if the blobber fails to find the heart return the image uncropped
            print('Could not find heart, check image')
            self.status = 'crop fail'
            cropped_image = image_4D.copy()
            crop_box = (0, 0, image_4D.shape[1], image_4D.shape[0])
            return cropped_image, crop_box

    def seg_image(self, image_4D, segger_num=segger_num):
        '''
        input the cropped image and segment the ventricles.
        '''
        segger = tf.keras.models.load_model(segger_path, compile=False)
        images = []
        position = image_4D.shape[2]
        timesteps = image_4D.shape[3]
        images_to_seg = []
        # loop through all images, make sure they are the right size and normalise
        for time in range(timesteps):
            for pos in range(position):
                image = image_4D[..., pos, time][..., np.newaxis]
                image = np.array(tf.image.resize(image, [segger_image_size, segger_image_size], method='nearest'))
                images.append(image / np.max(image))
                image = normalize(image)
                images_to_seg.append(image)
        images_to_seg = np.stack(images_to_seg, 0)
        masks = segger.predict(images_to_seg)
        if type(masks) == list:
            masks = masks[-1]
        masks = convert_to_4d(masks, position)  # change the output into the form h,w,position,time, channel
        images = convert_to_4d(images, position)
        masks = get_one_hot(np.argmax(masks, axis=-1), 3)  # binarise each mask
        masks = remove_basal_apical(masks)
        masks = remove_unconnected_segmentations(masks)
        # combine masks and images into color masks for plotting
        segged_image = np.empty_like(masks)
        for time in range(timesteps):
            for pos in range(position):
                segged_image[..., pos, time, :] = color_mask_image(images[..., pos, time, :],
                                                                   masks[..., pos, time, :]) / 255

        masks = reverse_through_timesteps(masks, reverse_resize,
                                          old_shape=self.cropped_image.shape[:2])  # turns mask from 128,128 to 400,400
        masks = reverse_through_timesteps(masks, reverse_crop, crop_box=self.crop_box, original_shape=(
        cropper_image_size, cropper_image_size, self.image.shape[2]))  # turns image back into original shape
        masks = reverse_through_timesteps(masks, reverse_pad, original_shape=self.image.shape,
                                          scale_shape=self.scale_shape)  # removes padding
        masks = reverse_through_timesteps(masks, reverse_resize,
                                          old_shape=self.image.shape[:2])  # turns image back into original shape

        return segged_image, masks

    def get_metrics(self, mask_4D):
        '''
        calculates cardiac metrics such as volume and mass
        '''

        # find the diastolic and systolic frames using the centre five segmented slices as they are the most reliable
        slice_segs = np.sum(np.sum(np.sum(np.sum(mask_4D[..., -2:], -1), 0), 0), -1)
        with_segs = np.where(slice_segs > 0)[0]
        mid_slice_idx = with_segs[round(len(with_segs) / 2)]
        min_heart_idx = mid_slice_idx - 2
        max_heart_idx = mid_slice_idx + 3
        if min_heart_idx < 0:
            max_heart_idx -= min_heart_idx
            min_heart_idx = 0
        sum_masks = np.sum(np.sum(np.sum(np.sum(mask_4D[..., -2:], 0), 0), -1)[min_heart_idx:max_heart_idx, :],
                           0)  # take the 5 largest slices
        dia_idx = np.argmax(sum_masks)
        sys_idx = np.argmin(sum_masks)
        self.dia_idx = dia_idx
        self.sys_idx = sys_idx

        # get volume and mass by summing the voxels and mulitplying by voxel size
        sum_height = np.sum(mask_4D, axis=0)
        sum_width = np.sum(sum_height, axis=0)
        sum_slices = np.sum(sum_width, axis=0)

        volume = np.sum(sum_slices[..., -1][..., np.newaxis], axis=-1) * self.voxel_size
        esv = round(volume[sys_idx], 5)
        edv = round(volume[dia_idx], 5)

        mass = np.sum(sum_slices[..., -2][..., np.newaxis], axis=-1) * self.voxel_size
        mass = round(mass[dia_idx], 5)
        mass = mass * 1.05

        sv = edv - esv
        ef = sv * 100 / edv

        fig = plt.figure()
        sns.set_palette('Set2')
        sns.lineplot(data=volume, linewidth=3)
        plt.title(f'Volume Curve for {self.patient}')
        plt.xlabel('Phases')
        plt.ylabel('Volume (ml)')
        Path(f"{self.results_path}/volume_curves").mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{self.results_path}/volume_curves/{self.patient}.jpg")
        plt.close()
        return volume, mass, esv, edv, sv, ef

    def reorder_diastole(self, image_4D):
        '''
        reorders an image such that the diastolic frame is first
        '''
        dia_idx = self.dia_idx
        if image_4D.ndim == 5:
            image_4D = np.concatenate([image_4D[:, :, :, dia_idx:, :], image_4D[:, :, :, :dia_idx, :]], axis=-2)
        elif image_4D.ndim == 4:
            image_4D = np.concatenate([image_4D[:, :, :, dia_idx:], image_4D[:, :, :, :dia_idx]], axis=-1)
        return image_4D

