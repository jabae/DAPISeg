import numpy as np
import tifffile as tif
import matplotlib.image as image
from cloudvolume import CloudVolume

from model import Model
from nets.unet import UNet

# Read image
def read_image(img_path):

	if img_path[-3:] == "png":

		img = image.imread(img_path)
		img = (img*255).astype("uint8")

		if len(img.shape)==3:
			img = img[:,:,2]

	elif img_path[-3:] == "tif":

		img = tif.imread(img_path)

		if len(img.shape)==3:
			img = img[:,:,2]

	elif img_path[:3] == "gs:":

		vol = CloudVolume(img_path, parallel=True, progress=False)
		img = vol[:,:,:][...,0]

	return img


def save_image(img_path, img):

	if img_path[-3:] == "png":

		image.imsave(img_path, img)

	elif img_path[-3:] == "tif":

		tif.imwrite(img_path, img)


def normalize_image(img):

	if img.max()>10:
		img = img/255

	return img.astype("float64")


def load_model(model_path):

	net = UNet()
	net.cuda()
	model = Model(net)
	
	model = load_chkpt(model, model_path)

	return model.eval()


def load_chkpt(model, path):

  model.load(path)

  return model