"""
Separates an equirectangular image into surfaces for a cube map

This creates files named cube-{up,down,front,back,left,right}.png in the current working directory
with a square image resolution with sides equal to 1/4 the equirectangular image width.

Usage: python eqr_to_cube.py <equirectangular image>
"""

import cv2
import numpy as np
import os
import sys

def eqr_to_rectilinear(im_eqr, ang, aspect_ratio=1, cam_height=None, eqr_overlay=False):
    lat1, long1 = ang
    eqr_height, eqr_width = im_eqr.shape[:2]
    
    if not cam_height:
        cam_height = int(eqr_width/4)
        
    cam_width = int(cam_height*aspect_ratio)
    
    xx, yy = np.meshgrid(np.linspace(-aspect_ratio,aspect_ratio, cam_width), np.linspace(-1,1, cam_height))
    
    rho = np.sqrt(xx**2 + yy**2)
    c = np.arctan(rho)
    
    cos_c = np.cos(c)
    sin_c = np.sin(c)

    lat = np.arcsin(cos_c*np.sin(lat1) + yy*sin_c*np.cos(lat1)/rho)
    long = long1 + np.arctan2(xx*sin_c, (rho*np.cos(lat1)*cos_c - yy*np.sin(lat1)*sin_c))
    
    lat = lat / (np.pi*0.5)
    long = long / np.pi
    
    lat = (lat + 1.0) * eqr_height/2
    long = (long + 1.0) * eqr_width/2
    
    if len(im_eqr.shape) == 3:
        im_win = np.zeros((cam_height, cam_width, 3), dtype=im_eqr.dtype)
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_NEAREST
        im_win = np.zeros((cam_height, cam_width), dtype=im_eqr.dtype)
        
    cv2.remap(im_eqr, long.astype(np.float32), lat.astype(np.float32), interp, im_win, cv2.BORDER_WRAP)
    
    return im_win

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python eqr_to_cube.py <equirectangular image>')
        exit()

    im_eqr = cv2.imread(sys.argv[1])

    if im_eqr is None:
        print('Unable to open image file')
        exit()

    print('Input image: %s' % sys.argv[1])
    print('Input size: %d x %d' % (im_eqr.shape[1], im_eqr.shape[0]))
    print('Face size: %d x %d\n' % (im_eqr.shape[1]//4, im_eqr.shape[1]//4))

    angles = {
        'up':    (-np.pi/2, 0),
        'down':  (np.pi/2, 0),
        'front': (0, 0),
        'back':  (0, np.pi),
        'left':  (0, -np.pi/2),
        'right': (0, np.pi/2)
    }

    cwd = os.getcwd()
    for k, v in angles.items():
        print('Generating %s' % k)
        im_win = eqr_to_rectilinear(im_eqr, v, 1)
        cv2.imwrite(os.path.join(cwd, 'cube-' + k + '.png'), im_win)

    print('Done')