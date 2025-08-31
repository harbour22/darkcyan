from pathlib import Path
import cv2
import yaml
import numpy as np

from darkcyan.constants import DEFAULT_RUNTIME_CONFIG_FILE, DEFAULT_CONFIG_DIR

from darkcyan.config import Config

import blessed
term = blessed.Terminal()

def show_image_with_coords(image_file, camera_name):
    print(f'Loading {image_file}')

    # read image 
    image = cv2.imread(image_file.as_posix())

    y_size, x_size, depth = image.shape

    with open((Path(DEFAULT_CONFIG_DIR) / DEFAULT_RUNTIME_CONFIG_FILE ), "r", encoding='utf8') as f:
        app_defaults = yaml.full_load(f.read())


    app_config = {**app_defaults}

    # create zone list
    for camera in app_config['camera_zones']:
        for zone in app_config['camera_zones'][camera]:
            zone_array = []
            for coord in app_config['camera_zones'][camera][zone]['coords']:
                x = coord[0]*x_size
                y = coord[1]*y_size
                zone_array.append([int(x),int(y)])
            contours = np.array(zone_array)
            
            app_config['camera_zones'][camera][zone]['coords_adj'] = contours
            print(f'Adjusted camera {camera} for zone {zone}')



    for zone in app_config['camera_zones'][camera_name]:
        overlay = image.copy()
        overlay_colour = app_config['zones'][zone]['colour']
        contours = app_config['camera_zones'][camera_name][zone]['coords_adj']        

        M = cv2.moments(contours)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(image, zone, (cx - 20, cy - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.fillPoly(overlay, [contours], color=overlay_colour )
        cv2.addWeighted(overlay, 0.3, image, 1 - 0.3,0, image)
        for point in contours:
            cv2.circle(image, tuple(point), 5, (255, 0, 255), -1)
            cv2.putText(image, f'[{point[0]/x_size:.3f}, {point[1]/y_size:.3f}]', tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    x_loc = y_loc = 0
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # draw circle here (etc...)
            print(f'- [{x/x_size:.3f}, {y/y_size:.3f}]')

        if event == cv2.EVENT_MOUSEMOVE:
            x_loc = x
            y_loc = y
            with term.location():
                print(term.move_xy(0,0) + term.clear_eol)
                print(term.move_xy(0,1) + f'x: {x_loc/x_size:.3f}, y: {y_loc/y_size:.3f}' + term.clear_eol)
                print(term.move_xy(0,2) + term.clear_eol)
                                
            
            



    # show the image, provide window name first
    cv2.imshow('image', image)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',onMouse)

    # add wait key. window waits until user presses a key

    # and finally destroy/close all open windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    show_image_with_coords( Path(Config.get_value('local_data_repository')) / 'test_data' / 'Reolink4k-Courtyard.jpg', 'reolink4k-courtyard')


if __name__ == "__main__":
    main()
