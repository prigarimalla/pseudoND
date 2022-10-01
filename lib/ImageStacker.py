from pickletools import uint8
from re import L
import cv2
import numpy as np
import logging
from collections import namedtuple

class _ImageLoader():

    VALID_FILES_TYPES = {'bmp', 'pbm', 'pgm', 'ppm', 'jpg', 'jpeg', 'tiff',
                         'tif'}

    class ImageLoadException(Exception):
        pass

    ImageData = namedtuple('ImageData', ['image', 'path'])

    def _load_images(image_paths: list[str]) -> list[ImageData]:
        def _can_load(image_path: str) -> bool:
            return image_path.split('.')[-1] in _ImageLoader.VALID_FILES_TYPES

        def _load(image_path: str) -> cv2.Mat:
            try: 
                # TODO: Expand to more filetypes
                return cv2.imread(image_path)
            except: 
                raise _ImageLoader.ImageLoadException()

        # Sort input image paths to ensure stable reference image selection
        image_paths = sorted(image_paths)

        image_datas = []
        skipped = []
        logging.info("Loading Images")
        for path in image_paths:
            try:
                if _can_load(path):
                    image_datas.append(
                        _ImageLoader.ImageData(image=_load(path), path=path))
                    logging.info(f'Loaded {path}')
                else:
                    skipped.append(path)

            except _ImageLoader.ImageLoadException:
                skipped.append(path)

        logging.info(f'Loaded {len(image_datas)} images')
        logging.info(f'Skipped {len(skipped)} files')
        if len(skipped):
            logging.debug(f'Skipped: {skipped}')

        return image_datas


class _ImageStacker():
    def __init__(self, reference_image: cv2.Mat, overlap_threshold=0.9) -> None:
        self.ref_img_color = reference_image
        self.overlap_threshold = overlap_threshold
        self.ref_img_gray = cv2.cvtColor(self.ref_img_color, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.ref_img_gray.shape

        self.orb_detector = cv2.ORB_create(5000)
        self.ref_kp, self.ref_descriptors = self.orb_detector.detectAndCompute(
            self.ref_img_gray, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        self.ref_bounding_contour = np.array(
            [[[0, 0],
              [0, self.height-1],
              [self.width-1, self.height-1],
              [self.width-1, 0]]],
            dtype=np.float32).reshape((-1,1,2))
        self.ref_area = cv2.contourArea(self.ref_bounding_contour)

        self.contours = []
        self.warped_images = []

    def _findHomography(self, image: cv2.Mat) -> np.array:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, descriptors = self.orb_detector.detectAndCompute(image_gray, None)

        # Sort matches and only keep top 90%
        unsorted_matches = self.matcher.match(descriptors, self.ref_descriptors)
        sorted_matches = sorted(unsorted_matches, key=lambda x: x.distance)
        matches = sorted_matches[:int(len(sorted_matches)*0.9)]
        
        # ---BEGIN MAGIC CODE---
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp[matches[i].queryIdx].pt
            p2[i, :] = self.ref_kp[matches[i].trainIdx].pt

        # Find the homography matrix.
        homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
        # ---END MAGIC CODE---

        return homography

    def addImageToStack(self, image: cv2.Mat, file_name: str = 'file') -> bool:
        try:
            homography = self._findHomography(image)
        except cv2.error as e:
            logging.info(f'Error adding {file_name} (homography calculation)')
            return False

        corners = self._calculateHomogenousCorners(homography)
        bounding_contour = np.array(
            [corners[0],corners[2],corners[3],corners[1]],
            np.int32).reshape((-1,1,2))

        intersecting_contour = _ImageStacker._calculateIntersectingContour([
            bounding_contour, self.ref_bounding_contour])
        intersecting_area = cv2.contourArea(intersecting_contour)
        
        if (intersecting_area/self.ref_area) >= self.overlap_threshold:
            self.contours.append(intersecting_contour)
            self.warped_images.append(
                cv2.warpPerspective(image, homography,
                                    (self.width, self.height)))
            
            logging.info(f'Added {file_name} to stack')
            return True

        logging.info(f'Error adding {file_name} (below overlap threshold)')
        return False

    def getAlignedImage(self):
        logging.info(f'Aligning Stack (size: {len(self.warped_images)})')
        opacity = 1/len(self.warped_images)
        return sum(map(lambda img: img*opacity, self.warped_images))

    def getAlignedImageLowMem(self):
        logging.info(f'Aligning Stack Low Mem(size: {len(self.warped_images)})')
        opacity = 1/len(self.warped_images)
        res = np.zeros_like(self.ref_img_color)

        # Calculate result image row by row to avoid instantiating massive 
        # intermediate array of float64's (OOMs in WASM). 
        for r in range(self.height):
            resRow = np.zeros((1, self.width, 3))
            for img in self.warped_images:
                imgRow = img[r:r+1, 0:self.width, 0:4]
                resRow += imgRow*opacity
            res[r] = resRow
        return res

    def getAlignedCroppedImage(self):
        aligned = self.getAlignedImageLowMem()

        logging.info(f'Cropping Stack (size: {len(self.contours)})')
        bound_tl, bound_br = self._calculateBoundingBox()
        aligned_and_cropped = aligned[bound_tl[1]:bound_br[1],
                                      bound_tl[0]:bound_br[0]]
        
        return aligned_and_cropped

    def _calculateHomogenousCorners(self, homography: np.array):
        src_pts = np.array([[[0, 0],
                             [self.width, 0],
                             [0, self.height],
                             [self.width, self.height]]],
                            dtype=np.float32)
        dst_pts = cv2.perspectiveTransform(src_pts, homography)
        
        res = []
        for c in dst_pts[0]:
            res.append((int(c[0]), int(c[1])))

        return res

    def _calculateBoundingBox(self):
        tl, bl, br, tr = _ImageStacker._calculateIntersectingContour(
                            self.contours)

        bound_tl = [max(tl[0][0], bl[0][0]),
                    max(tl[0][1], tr[0][1])]

        bound_br = [min(br[0][0], tr[0][0]),
                    min(br[0][1], bl[0][1])]
        
        return (bound_tl, bound_br)

    @staticmethod
    def _calculateIntersectingContour(contours):
        corners = {'tr': [], 'tl': [], 'br': [], 'bl': []}
        for contour in contours:
            corners['tl'].append(contour[0])
            corners['bl'].append(contour[1])
            corners['br'].append(contour[2])
            corners['tr'].append(contour[3])

        corner_tl = [float('-inf'), float('-inf')]
        for c in corners['tl']:
            x = c[0][0]
            y = c[0][1]
            corner_tl[0] = max(x, corner_tl[0])
            corner_tl[1] = max(y, corner_tl[1])

        corner_tr = [float('inf'), float('-inf')]
        for c in corners['tr']:
            x = c[0][0]
            y = c[0][1]
            corner_tr[0] = min(x, corner_tr[0])
            corner_tr[1] = max(y, corner_tr[1])

        corner_bl = [float('-inf'), float('inf')]
        for c in corners['bl']:
            x = c[0][0]
            y = c[0][1]
            corner_bl[0] = max(x, corner_bl[0])
            corner_bl[1] = min(y, corner_bl[1])

        corner_br = [float('inf'), float('inf')]
        for c in corners['br']:
            x = c[0][0]
            y = c[0][1]
            corner_br[0] = min(x, corner_br[0])
            corner_br[1] = min(y, corner_br[1])

        contour_order = [corner_tl, corner_bl, corner_br, corner_tr]
        return np.array(contour_order, np.int32).reshape((-1,1,2))

class UnableToAlignAndStackImagesException(Exception):
    pass

def align_and_stack_images(image_paths: list[str],
                           dst_path: str,
                           should_crop: bool = True) -> cv2.Mat:
    """
    align_and_stack_images() does the following in order:
        1. Takes a list of file system paths to images
        2. Loads them into memory
        3. Calculates features in each one
        4. Uses matching features to align the images
        5. Discards any images that don't align well enough
        6. Optionally calculates a maximal crop where all input images overlap
        7. Writes the resulting image to the file system

    Args:
        image_paths: List of paths to the input images
        dst_path: Path to write the output image to
        should_crop: Whether or not to crop the output to the overlapping region 
    """
    image_datas = _ImageLoader._load_images(image_paths)
    if len(image_datas) < 2:
        logging.error(
            f'Too few images to proceed (min 2, supplied {len(image_datas)}')
        raise UnableToAlignAndStackImagesException()

    # Use first image as reference image to align others to
    ref_image_data = image_datas[0]
    stacker = _ImageStacker(ref_image_data.image)

    logging.info('Adding Images to Stack')
    errors = []
    for image, path in image_datas:
        success = stacker.addImageToStack(image, path.split('/')[-1])
        if not success:
            errors.append(path)
        
    if len(errors):
        logging.info(f'Unable to calculate {len(errors)} homographies: {errors}')

    output = stacker.getAlignedCroppedImage() if should_crop else stacker.getAlignedImage()

    logging.info(f'Writing output to {dst_path}')
    cv2.imwrite(dst_path, output)
