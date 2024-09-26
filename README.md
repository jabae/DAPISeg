# DAPISeg
DAPISeg is a tool that approximates cell area from DAPI stained images.

## Installation
```
git clone https://github.com/jabae/DAPISeg.git
cd DAPISeg

pip install -r requirements.txt
```
## Pipeline
DAPISeg consists of two steps: 1. nucleus prediction and 2. cell segmentation. 

### Nucleus prediction
```
python DAPISeg/detect_nucleus.py --image [test_data/test_image.png] --model [models/nucleus_model.chkpt] --output [output_path.tif]
```

### Cell segmentation
```
python DAPISeg/segment_cell.py --image [test_data/test_image.png] --nucleus [test_data/test_nucleus.tif] --output [output_path.tif]
```

*[ ] are variables users can specify. Currently, the variables point to the test data.
