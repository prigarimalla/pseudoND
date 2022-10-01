from ImageStacker import _ImageStacker
import cv2
import numpy as np
import base64
import sys

def _dataurl_to_cv2Mat(uri) -> cv2.Mat:
   encoded_data = uri.split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def _cv2Mat_to_dataurl(img_bytes) -> str:
    _, buffer = cv2.imencode(".jpg", img_bytes)
    b64_encoded = base64.b64encode(buffer).decode("ascii")
    return f'data:image/jpg;base64,{b64_encoded}'

_STACKER = None
def load_and_stack_from_data_url_event(dataurl, file_name):
    global _STACKER

    img  = _dataurl_to_cv2Mat(dataurl)
    print(f'Size({file_name})={sys.getsizeof(img)}')
    # Init stacker with first access
    if _STACKER == None:
        _STACKER = _ImageStacker(img)

    _STACKER.addImageToStack(img, file_name)

def get_final_image():
    global _STACKER
    res_img = _STACKER.getAlignedCroppedImage()
    return _cv2Mat_to_dataurl(res_img)