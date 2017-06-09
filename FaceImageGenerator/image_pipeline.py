import dlib
import os
import numpy as np
import glob
import math
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread
from skimage import transform, exposure, filters
from .image_augmentation import rotate, zoom, width_shift, height_shift
from .image_processing import bbox_extractor, save_image, add_landmarks_to_img 
import threading


class FACE_pipeline():

    def __init__(self,
            histogram_normalization=False,
            grayscale = False,
            standardisation = False,
            output_size = [160,240],
            face_size = 160,
            resize = False,
            rotation_range = 10,
            width_shift_range = 0.05,
            height_shift_range = 0.05,
            zoom_range = 0.05,
            gaussian_range = 0,
            fill_mode = 'edge',
            random_flip = True,
            path_to_shape_model = None,
            allignment_type = 'similarity',
            ):
        '''
        '''
        self.output_size = output_size 
        self.face_size = face_size 
        self.allignment_type = allignment_type 
        # {'similarity','affine','projective'}

        self.histogram_normalization = histogram_normalization
        self.grayscale = grayscale

        # Standardisation (zero-mean, unit-variance)
        self.standardisation = standardisation 
        self.resize = resize 

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.gaussian_range = gaussian_range 
        self.fill_mode = fill_mode
        self.random_flip = random_flip

        self.shape_predictor = None
        self.path_to_shape_model = path_to_shape_model


    def transform(self, 
            img, 
            face_detect = False, 
            preprocessing = False, 
            augmentation = False, 
            ):

        pts = None
        img = gray2rgb(img)

        if face_detect:

            # load detector
            if self.shape_predictor==None:
                self.detector = dlib.get_frontal_face_detector()

                if self.path_to_shape_model!=None:
                    pwd = os.path.abspath(self.path_to_shape_model)
                else:
                    pwd = os.path.dirname(os.path.abspath(__file__))
                    pwd = pwd+'/data/shape_predictor_68_face_landmarks.dat'
                self.shape_predictor = dlib.shape_predictor( pwd )

            img, pts, pts_raw  = bbox_extractor(
                    self.detector, 
                    self.shape_predictor, 
                    img,
                    self.output_size,
                    self.face_size,
                    self.allignment_type,
                    self.fill_mode,
                    )
        else:
            pts = None 
            pts_raw = None


        # normalize input to float32 with values between 0 and 1
        img = np.float32(img)
        if np.all(img==img.item(0)):
            s_0 = 0
            s_1 = 1
            img = np.zeros_like(img)
        else:
            s_0 = np.min(img)
            s_1 = np.max(img)
            img = (img-s_0)/(s_1-s_0)
            # img -= np.apply_over_axes(np.min, img, [0,1])
            # img /= np.apply_over_axes(np.max, img, [0,1])

        if preprocessing:
            if self.grayscale and img.shape[-1]==3:
                img = rgb2gray(img)
                img = img[:,:,None]

            if self.histogram_normalization:
                img = np.float32(img)
                img = exposure.equalize_hist(img)

            if self.resize:
                img = transform.resize(img, self.resize)

            if self.standardisation:
                img -= np.apply_over_axes(np.mean, img, [0,1])

                if np.all(img==img.item(0)):
                    # if all pixels values are identical, variance is zerso -> no scaling
                    pass
                else:
                    img /= np.apply_over_axes(np.std, img, [0,1])



        if augmentation:

            ############################################################### 
            # add noise trough various image filter 
            ############################################################### 

            # add gaussian noise trough smoothing
            if self.gaussian_range:
                sigma = np.random.uniform(0, self.gaussian_range)
                img = filters.gaussian(img, sigma)
            ############################################################### 



            ############################################################### 
            # add noise trough transformation 
            ############################################################### 

            # compute random rotation
            deg = np.random.uniform(-self.rotation_range, self.rotation_range)
            trans = rotate(img.shape, deg)

            # # compute random zooming
            z = np.random.uniform(-self.zoom_range, self.zoom_range)
            trans += zoom(img.shape, z)

            # # compute random shifting
            shift = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            trans += width_shift(img.shape, shift)

            # compute random shifting
            shift = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            trans += height_shift(img.shape, shift)

            if self.random_flip:
                if np.random.rand()>0.5:
                    img=img[:,::-1]

            img_tr = transform.warp(img, trans, mode=self.fill_mode)

            # if transformation is valid, update image
            if not (np.all(img==0) or np.any(np.isnan(img))):
                img = img_tr 
                if pts!=None:
                    pts = trans.inverse(pts)
            ############################################################### 

        img = img * (s_1-s_0) + s_0

        return img, pts, pts_raw


    def batch_transform(self, batch, *args, **kwargs):

        # parallel
        ################################################################ 
        # threads = [None] * len(batch)
        # out_img = [None] * len(batch)
        # out_pts = [None] * len(batch)
        # out_pts_raw = [None] * len(batch)

        # def _target(i, sample, *args, **kwargs):
            # out = self.transform(sample, *args, **kwargs)
            # out_img[i] = out[0]
            # out_pts[i] = out[1]
            # out_pts_raw[i] = out[2]


        # for i, sample in enumerate(batch):
            # threads[i] = threading.Thread(
                # target=_target,
                # args=(i, sample, *args),
                # kwargs = kwargs,
                # )
            # threads[i].start()

        # for t in threads:t.join()

        # out_img = np.stack(out_img)
        # out_pts = np.stack(out_pts)
        # out_pts_raw = np.stack(out_pts_raw)

        # return out_img, out_pts, out_pts_raw

        # ToDo! find bug: some images show weird artefacts when using multi-threading
        # multi-threading is deactivated and batch_transform works in serial mode
        IMG, PTS, PTS_RAW = [], [], []
        for img in batch:
            img, pts, pts_raw = self.transform(img, *args, **kwargs)
            IMG.append(img)
            PTS.append(pts)
            PTS_RAW.append(pts_raw)

        return np.array(IMG), np.array(PTS), np.array(PTS_RAW)



