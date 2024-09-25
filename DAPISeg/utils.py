import numpy as np
import tifffile as tif
import matplotlib.image as image
from cloudvolume import CloudVolume

# Read image
def read_image(img_path):

	if path[-3:] == "png":

		img = image.imread(img_path)
		img = (img*255).astype("uint8")

	elif path[-3:] == "tif":

		img = tif.imread(img_path)

	elif path[:3] == "gs:":

		vol = CloudVolume(img_path, parallel=True, progress=False)
		img = vol[:,:,:][...,0]

	return img


def save_image(img_path, img):

	if path[-3:] == "png":

		image.imsave(img_path, img)

	elif path[-3:] == "tif":

		tif.imwrite(img_path, img)