import torch
from torchvision import transforms
from PIL import Image


# Define a function to load the model in chkpt file
def load_model(checkpoint_path):

    model = torch.load(checkpoint_path)  # Load the checkpoint
    model.eval()  # Set model to evaluation mode

    return model

# Define a function to run inference
def run_inference(model, image_tensor, device='cpu'):

    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    # Run the inference
    with torch.no_grad():
        output = model(image_tensor)
    
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True, type=str,
        help="Model path in chkpt format")

    # Load the checkpoint and model
    checkpoint_path = 'path/to/your/model_checkpoint.chkpt'
    model = load_model(checkpoint_path)

    # Preprocess the image
    image_path = 'path/to/your/image.jpg'
    image_tensor = preprocess_image(image_path)

    # Run inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = run_inference(model, image_tensor, device)