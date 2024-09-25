"""
Script to predict nucleus.
"""

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from utils import read_image, save_image

from nets.unet import UNet


class Model(nn.Module):
  """
  Model wrapper for training.
  """
  
  def __init__(self, model):

    super(Model, self).__init__()
    self.model = model

  def forward(self, sample):

    mask = self.model(sample['image'])
    
    preds = {}
    preds["mask"] = torch.sigmoid(mask)
    
    return preds

  def save(self, fpath):

    torch.save(self.model.state_dict(), fpath)

  def load(self, fpath):

    state_dict = torch.load(fpath)
    
    self.model.load_state_dict(state_dict, strict=False)


def load_chkpt(model, path):

  model.load(path)

  return model


# Define a function to load the model in chkpt file
def load_model(model_path):

	net = UNet()
	net.cuda()
	model = Model(net)
	
	model = load_chkpt(model, model_path)

	return model.eval()

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
	image_tensor = torch.from_numpy(image)

	# Run inference
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	nucleus_pred = run_inference(model, image_tensor, device)