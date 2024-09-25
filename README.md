# DAPISeg
DAPI image segmentor

## Installation
`pip install -r requirements.txt`

## Nucleus prediction
`python DAPISeg/detect_nucleus.py --image test_data/test_image.png --model models/nucleus_model.chkpt`

## Cell segmentation
`python DAPISeg/segment_cell.py --image test_data/test_image.png --nucleus test_data/test_nucleus.tif`