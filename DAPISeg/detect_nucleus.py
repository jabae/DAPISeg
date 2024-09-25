"""
Script to predict nucleus.
"""

import argparse
import torch
import numpy as np

from utils import read_image, save_image, load_model, chunk_bboxes



def preprocess(img):

	if img.max() > 10:
		img = img/255

	img = np.reshape(img, (1,1,)+img.shape)

	return img.astype("float32")
	

# Define a function to run inference
def run_inference(model, image, device='cpu'):

	image_size = image.shape[2:]
	patch_size = (128, 128)
	overlap_size = (32, 32)

	model = model.to(device)

	bbox_list = chunk_bboxes(image_size, patch_size, overlap_size)

	pred = np.zeros(image_size)
	# Run the inference
	with torch.no_grad():

		for b in bbox_list:

			patch = image[:,:,b[0][0]:b[1][0],b[0][1]:b[1][1]]
			patch_tensor = torch.from_numpy(patch)
			patch_tensor = patch_tensor.to(device)
			
			pred_patch = model(patch_tensor)
			pred_patch = pred_patch.cpu().numpy()
			pred_patch = pred_patch[0, 0,
															overlap_size[0]//2:patch_size[0]-overlap_size[0]//2,
															overlap_size[1]//2:patch_size[1]-overlap_size[1]//2]
			pred[b[0][0]+overlap_size[0]//2:b[1][0]-overlap_size[0]//2,
					 b[0][1]+overlap_size[1]//2:b[1][1]-overlap_size[1]//2] = pred_patch

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

	# Run inference
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	nucleus_pred = run_inference(model, image, device)
	save_image(output_path, nucleus_pred)