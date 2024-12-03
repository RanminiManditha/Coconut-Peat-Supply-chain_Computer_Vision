import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import transforms
from PIL import Image
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a custom collate function to handle variable-sized targets
def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)  # Return images and list of targets


if __name__ == '__main__':  # Ensure all main code is within this block

    # Load the pre-trained SSD model (pre-trained on COCO)
    model = ssd300_vgg16(pretrained=True)
    model.head.classification_head = nn.Conv2d(in_channels=1024, out_channels=4 * 4, kernel_size=3,
                                               padding=1)  # Adjust for 4 classes
    model.to(device)  # Move model to the appropriate device (GPU/CPU)

    # Load and preprocess the image (for testing after training)
    image_path = "D:/SLIIT/Research/test 2/car.jpg"  # Replace with the path to your test image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # SSD expects 300x300 input images
        transforms.ToTensor()  # Convert the image to PyTorch tensor
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Load your custom dataset (COCO format)
    dataset = CocoDetection(
        root='/SLIIT/Research/husk images',
        annFile='/SLIIT/Research/dataset annotations for SSD/annotations/instances_default.json',
        transform=transforms.Compose([
            transforms.Resize((300, 300)),  # Resize all images to 300x300 (SSD expects 300x300)
            transforms.ToTensor()  # Convert the images to tensors
        ])
    )

    # Create a DataLoader for your dataset with a custom collate function
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Define loss functions
    localization_loss_fn = torch.nn.SmoothL1Loss()  # For bounding box regression
    confidence_loss_fn = torch.nn.CrossEntropyLoss()  # For classification

    # Set the model to training mode
    model.train()

    # Training loop
    num_epochs = 10  # You can adjust the number of epochs
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = images.to(device)

            # Check if targets are None or empty
            if targets is None or len(targets) == 0:
                raise ValueError("Targets should not be None or empty during training")

            # Ensure that targets contain bounding boxes and labels
            for i in range(len(targets)):
                for j in range(len(targets[i])):
                    if "bbox" not in targets[i][j]:
                        raise ValueError(f"Missing 'bbox' in target for image {i}")
                    if "category_id" not in targets[i][j]:
                        raise ValueError(f"Missing 'category_id' in target for image {i}")

                    # Convert to tensors and move to the correct device
                    targets[i][j]["bbox"] = torch.tensor(targets[i][j]["bbox"]).to(device)
                    targets[i][j]["category_id"] = torch.tensor(targets[i][j]["category_id"]).to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images, targets)  # Forward pass with targets

            # Extract predicted boxes and classes from outputs
            predicted_boxes = outputs['boxes']
            predicted_classes = outputs['scores']

            # Ground truth (targets)
            ground_truth_boxes = [torch.stack([torch.tensor(t["bbox"]) for t in target]) for target in targets]
            ground_truth_classes = [torch.cat([torch.tensor(t["category_id"]) for t in target]) for target in targets]

            # Calculate the localization loss (bounding box regression)
            loc_loss = localization_loss_fn(predicted_boxes, ground_truth_boxes)

            # Calculate the confidence loss (classification)
            conf_loss = confidence_loss_fn(predicted_classes, torch.cat(ground_truth_classes))

            # Combine losses
            total_loss = loc_loss + conf_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")
