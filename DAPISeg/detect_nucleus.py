"""
Script to predict nucleus.
"""

import argparse
import torch
import numpy as np

from utils import read_image, save_image, load_model



def preprocess(img):

	if img.max() > 10:
		img = img/255

	img = np.reshape(img, (1,)+img.shape)

	return img
	

# Define a function to run inference
def run_inference(model, image_tensor, device='cpu'):

	image_tensor = image_tensor.to(device)
	model = model.to(device)

	# Run the inference
	with torch.no_grad():
		pred = model(image_tensor)

	pred = torch.sigmoid(pred)

	return pred

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--image", required=True, type=str,
		help="Path to image")
	parser.add_argument("--model", required=True, type=str,
	  help="Model path in chkpt format")
	parser.add_argument("--output", required=False, type=str, default="nucleus_pred.tif",
		help="Path to save cell segmentation image in cloudvolume, png, or tif format")

	args = parser.parse_args()

	img_path = args.image
	model_path = args.model
	output_path = args.output

	# Load model
	model = load_model(model_path)

	# Load image
	image = read_image(img_path)
	image = preprocess(image)
	image_tensor = torch.from_numpy(image)

	# Run inference
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	nucleus_pred = run_inference(model, image_tensor, device)