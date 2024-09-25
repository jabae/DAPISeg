"""
Script to segment cell from predicted nuclei.
"""

import numpy as np
import matplotlib.pyplot as plt
import cc3d
from scipy.spatial import cKDTree
import argparse


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--nucleus_pred", required=True, type=str,
		help="Path to nuecleus prediction image in cloudvolume, png, or tif format")
	parser.add_argument("--image", required=True, type=str,)