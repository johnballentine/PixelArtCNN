import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image, ImageOps
import os
import glob
import shutil

class PixelLoss(nn.Module):
    # Penalizes intermediate transparency values that are neither 0 nor 1
    def __init__(self):
        super(PixelLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Basic MSE loss
        base_loss = self.mse(pred, target)

        # Extract the alpha channel from the predictions
        pred_alpha = pred[:, 3, :, :]

        # Create a mask for alpha values that are neither 0 nor 1
        mask = (pred_alpha > 0) & (pred_alpha < 1)

        # Calculate the penalty term
        alpha_penalty = torch.sum(mask.float())

        # Hyperparameter to tune penalty for intermediate transparency values
        penalty_weight = args.alpha_penalty

        # Combine the base loss and the penalty
        total_loss = base_loss + penalty_weight * alpha_penalty

        return total_loss

class NNDownscale(nn.Module):
    def __init__(self):
        super(NNDownscale, self).__init__()
        
        # Input shape: [batch, 4, 256, 256] (RGBA channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 64, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 64, 128, 128]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 128, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 128, 64, 64]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 256, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 256, 32, 32]
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Output shape: [batch, 512, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output shape: [batch, 512, 16, 16]
            
            nn.Conv2d(512, 4, kernel_size=1, stride=1, padding=0),  # Output shape: [batch, 4, 16, 16] (RGBA channels)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    

def train_model(directory, epochs, model_save_path="model.pth", batch_size=16):
    print(f"Training model from directory: \033[36m{directory}\033[0m")  # Debug print
    transform = transforms.ToTensor()
    label_files = sorted(glob.glob(os.path.join(directory, '*_label.png')))
    input_files = sorted(glob.glob(os.path.join(directory, '*_input.png')))
    
    # Initialize lists to hold label and input data
    labels = []
    input = []
    
    # Check properties for label images
    for f in label_files:
        img = Image.open(f)
        if img.size != (16, 16):
            print(f"\033[91mError:\033[0m The label image {f} must be 16x16.")
            return
        if 'A' not in img.getbands():
            print(f"\033[91mError:\033[0m The label image {f} must have an alpha channel.")
            return
        labels.append(transform(img))
    
    # Check properties for input images
    for f in input_files:
        img = Image.open(f)
        if img.size != (256, 256):
            print(f"\033[91Error:\033[0m The input image {f} must be 256x256.")
            return
        if 'A' not in img.getbands():
            # Check if there's an alpha channel. If not, add one with all values 255.
            # 4 channels are expected for training.
            img = ImageOps.exif_transpose(img.convert("RGBA"))
        input.append(transform(img))

    # Convert lists to torch tensors
    labels = torch.stack(labels)
    input = torch.stack(input)
    
    print("Creating DataLoader.")
    dataset = TensorDataset(input, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Initializing loss and optimizer.")

    if args.cpu != True:
        # Set the device to GPU if available, otherwise, CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        print(f"Using GPU: {device}")
    else: 
        print("Using CPU")

    model = NNDownscale().to(device)
    criterion = PixelLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)

    print("----")
    print("\033[92mTraining started.\033[0m")
    print("----")
    temp_model_save_path = f"{model_save_path}_temp.pth"
    for epoch in range(epochs):
        epoch_loss = 0.0  # Initialize epoch_loss to accumulate loss over the epoch

        for batch_idx, (batch_processed, batch_labels) in enumerate(dataloader):

            # Move the batch tensors to the same device as the model
            batch_processed, batch_labels = batch_processed.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_processed)

            loss = criterion(outputs, batch_labels)
            epoch_loss += loss.item()  # Accumulate batch loss into epoch loss
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            print(f"\033[34mEpoch:\033[0m {epoch+1}/{epochs}, \033[34mBatch:\033[0m {batch_idx+1}/{len(dataloader)}")

            # Print the mean, median, min, and max for all channels
            all_channel_values = outputs.view(-1)  # Reshape tensor to combine all channel values

            mean_value = torch.mean(all_channel_values)
            median_value = torch.median(all_channel_values)
            min_value = torch.min(all_channel_values)
            max_value = torch.max(all_channel_values)

            print()
            print(f"    \033[92mMean:\033[0m {mean_value}")
            print(f"  \033[92mMedian:\033[0m {median_value}")
            print(f"     \033[92mMin:\033[0m {min_value}")
            print(f"     \033[92mMax:\033[0m {max_value}")
            print()
            print(f"\033[93mLoss:\033[0m {loss.item()}")
            print("----")

        avg_epoch_loss = epoch_loss / len(dataloader)  # Calculate average epoch loss
        scheduler.step(avg_epoch_loss)  # Step the scheduler based on the average epoch loss

        # Save the model to temp after epoch to enable early stopping with lower risk of corruption
        torch.save(model.state_dict(), temp_model_save_path)
        os.rename(temp_model_save_path, model_save_path)

        # Copy the model file at every 10th epoch with a special name
        if (epoch + 1) % 20 == 0:
            filename_without_extension, _ = os.path.splitext(model_save_path)
            special_model_save_path = f"{filename_without_extension}_epoch{epoch+1}.pth"
            shutil.copy(model_save_path, special_model_save_path)

def use_inference(model_path, input_image_path, output_image_path):
    print("Starting inference.")

    input_image = Image.open(input_image_path)

    # Check if the image is 256x256
    if input_image.size != (256, 256):
        print("Error: The input image must be 256x256.")
        return
    
     # Check if the image has an alpha channel
    if 'A' not in input_image.getbands():
        print("Adding alpha channel.")
        input_image = ImageOps.exif_transpose(input_image.convert("RGBA"))

    channels = len(input_image.getbands())
    
    model = NNDownscale()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.ToTensor()
    input_tensor = transform(input_image).unsqueeze(0)

    if args.cpu != True:
        # Set the device to GPU if available, otherwise, CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        print(f"Using GPU: {device}")
    else: 
        print("Using CPU")
    
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Generate unique filename
    counter = 0
    unique_output_path = output_image_path
    while os.path.exists(unique_output_path):
        counter += 1
        filename, file_extension = os.path.splitext(output_image_path)
        unique_output_path = f"{filename}_{counter}{file_extension}"

    # Rescale to 0-255 and convert to byte tensor
    output_tensor = (output_tensor.squeeze() * 255).byte()

    # Convert to PIL Image
    output_image = Image.fromarray(output_tensor.permute(1, 2, 0).numpy(), 'RGBA')
    output_image.save(unique_output_path)

    print(f"Output image saved at {unique_output_path}")

def main(args):
    print("Parsing arguments.")  # Debug print
    if args.train:
        train_model(args.data, args.epochs)
    elif args.infer:
        use_inference(args.model_path, args.input_image, args.output_image)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Art CNN: Restore pixel art to pixel perfection.")

    parser.add_argument('--data', type=str, default='', help='Directory containing training data. Images must be appended with _input.png or _label.png')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Use model to clean pixel art.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for training. Default: 0.005')
    parser.add_argument('--alpha_penalty', type=float, default=0.8, help='Discourages transparency values that are not either 0 or 255 during training. Increases sharp edges, but may corrupt output. Default: 0.8')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to the trained model, or path to save the model.')
    parser.add_argument('--input_image', type=str, default='', help='Path to the input image for inference')
    parser.add_argument('--output_image', type=str, default='', help='Path to save the output image')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    
    args = parser.parse_args()

    if args.train:
        train_model(args.data, args.epochs, args.model_path)
    elif args.infer:
        use_inference(args.model_path, args.input_image, args.output_image)
    else:
        print("Invalid choice. Exiting.")

