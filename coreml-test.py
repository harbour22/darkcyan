import coremltools as ct
import cv2
import PIL.Image

import numpy as np

def load_image_as_numpy_array(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.BICUBIC)
    img_np = np.array(img).astype(np.float32) # shape of this numpy array is (Height, Width, 3)
    return img_np

# Load the model
model = ct.models.MLModel('/Users/chris/developer/darkcyan_data/engines/yolov8_4.9_large-det.mlpackage')

Height = 640  # use the correct input image height
Width = 640  # use the correct input image width
model_expected_input_shape = (1, 3, Height, Width) # depending on the model description, this could be (3, Height, Width)


img_as_np_array = load_image_as_numpy_array("/Users/chris/developer/github_projects/darkcyan/image.jpg",resize_to=(Width, Height))


# Add the batch dimension if the model description has it.
img_as_np_array = np.reshape(img_as_np_array, model_expected_input_shape)

#destRGB = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
#pil_img = PIL.Image.fromarray(resized_image)

open_cv_image = np.array(img_as_np_array)


# Load the image and resize using PIL utilities.
#_, img = load_image('', resize_to=(Width, Height))
out_dict = model.predict({'image': img_as_np_array})
print(out_dict)
for coord in out_dict['coordinates']:
    print(coord)
    x1 = int(coord[3]*640)
    y1 = int(coord[1]*640)
    x2 = x1 + int(coord[2]*640)
    y2 = y1 - int(coord[3]*640)
    print(x1,y1,x2,y2)
    resized_image = cv2.rectangle(open_cv_image, (x1,y1), (x2,y2),color=(255,0,0), thickness=2)


cv2.imshow('image',open_cv_image)
cv2.waitKey(0)
