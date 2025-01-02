import torch
import logging
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from training.FoulDataPreprocessor import FoulDataPreprocessor
from training.Decoder import Decoder

class MultiTaskModel(nn.Module):
    def __init__(self, input_size, action_classes, bodypart_classes, offence_classes, touchball_classes, trytoplay_classes, severity_classes):
        super(MultiTaskModel, self).__init__()

        # Shared input layers with Batch Normalization
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        # Separate output layers for each task
        self.fc_actionclass = nn.Linear(128, action_classes)
        self.fc_bodypart = nn.Linear(128, bodypart_classes)
        self.fc_offence = nn.Linear(128, offence_classes)
        self.fc_touchball = nn.Linear(128, touchball_classes)
        self.fc_trytoplay = nn.Linear(128, trytoplay_classes)
        self.fc_severity = nn.Linear(128, severity_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Custom initialization (He initialization)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Forward pass through shared layers with ReLU and Batch Normalization
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        # Separate outputs for each task
        actionclass_output = self.fc_actionclass(x)
        bodypart_output = self.fc_bodypart(x)
        offence_output = self.fc_offence(x)
        touchball_output = self.fc_touchball(x)
        trytoplay_output = self.fc_trytoplay(x)
        severity_output = self.fc_severity(x)
        
        return actionclass_output, bodypart_output, offence_output, touchball_output, trytoplay_output, severity_output

def train_model(X_train, y_train, class_weights, severity_classes, epochs=20, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert inputs and labels to tensors and move them to device
    X_train = X_train.to(device)
    y_train = {key: value.to(device) for key, value in y_train.items()}
    
    # Initialize the model
    model = MultiTaskModel(
        input_size=X_train.shape[1], 
        action_classes=len(class_weights['actionclass']),
        bodypart_classes=len(class_weights['bodypart']),
        offence_classes=len(class_weights['offence']),
        touchball_classes=len(class_weights['touchball']),
        trytoplay_classes=len(class_weights['trytoplay']),
        severity_classes=severity_classes  # Adding severity classes
    ).to(device)

    # Define loss functions and optimizer
    criterion_actionclass = nn.CrossEntropyLoss(weight=class_weights['actionclass'].to(device))
    criterion_bodypart = nn.CrossEntropyLoss(weight=class_weights['bodypart'].to(device))
    criterion_offence = nn.CrossEntropyLoss(weight=class_weights['offence'].to(device))
    criterion_touchball = nn.CrossEntropyLoss(weight=class_weights['touchball'].to(device))
    criterion_trytoplay = nn.CrossEntropyLoss(weight=class_weights['trytoplay'].to(device))
    criterion_severity = nn.CrossEntropyLoss(weight=class_weights['severity'].to(device))  # Add loss for severity

    # AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler with ReduceLROnPlateau for better adaptive learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Add gradient clipping to avoid exploding gradients
    max_grad_norm = 1.0  # Set this to a suitable value

    model.train()
    
    # Create DataLoader for batching
    dataset = TensorDataset(X_train, y_train['actionclass'], y_train['bodypart'], y_train['offence'], y_train['touchball'], y_train['trytoplay'], y_train['severity'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize lists to store losses for plotting
    total_losses = []
    actionclass_losses = []
    bodypart_losses = []
    offence_losses = []
    touchball_losses = []
    trytoplay_losses = []
    severity_losses = []  # Add list for severity losses

    for epoch in range(epochs):
        total_loss = 0
        total_loss_actionclass = 0
        total_loss_bodypart = 0
        total_loss_offence = 0
        total_loss_touchball = 0
        total_loss_trytoplay = 0
        total_loss_severity = 0  # Add variable for severity loss
        
        for inputs, actionclass_labels, bodypart_labels, offence_labels, touchball_labels, trytoplay_labels, severity_labels in data_loader:
            inputs = inputs.to(device)
            actionclass_labels = actionclass_labels.to(device)
            bodypart_labels = bodypart_labels.to(device)
            offence_labels = offence_labels.to(device)
            touchball_labels = touchball_labels.to(device)
            trytoplay_labels = trytoplay_labels.to(device)
            severity_labels = severity_labels.to(device)  # Add severity labels
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss for each task
            loss_actionclass = criterion_actionclass(outputs[0], actionclass_labels)
            loss_bodypart = criterion_bodypart(outputs[1], bodypart_labels)
            loss_offence = criterion_offence(outputs[2], offence_labels)
            loss_touchball = criterion_touchball(outputs[3], touchball_labels)
            loss_trytoplay = criterion_trytoplay(outputs[4], trytoplay_labels)
            loss_severity = criterion_severity(outputs[5], severity_labels)  # Calculate loss for severity
            
            # Total loss (with optional weighting for tasks)
            total_loss_batch = loss_actionclass + loss_bodypart + loss_offence + loss_touchball + loss_trytoplay + loss_severity
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_loss_actionclass += loss_actionclass.item()
            total_loss_bodypart += loss_bodypart.item()
            total_loss_offence += loss_offence.item()
            total_loss_touchball += loss_touchball.item()
            total_loss_trytoplay += loss_trytoplay.item()
            total_loss_severity += loss_severity.item()  # Add severity loss

        # Step learning rate scheduler
        scheduler.step(total_loss)

        # Append loss values to lists for plotting
        total_losses.append(total_loss)
        actionclass_losses.append(total_loss_actionclass)
        bodypart_losses.append(total_loss_bodypart)
        offence_losses.append(total_loss_offence)
        touchball_losses.append(total_loss_touchball)
        trytoplay_losses.append(total_loss_trytoplay)
        severity_losses.append(total_loss_severity)  # Append severity loss

        logging.info(f"Epoch [{epoch + 1}/{epochs}], "
                    f"Total Loss: {total_loss:.4f}, "
                    f"Action Class Loss: {total_loss_actionclass:.4f}, "
                    f"Body Part Loss: {total_loss_bodypart:.4f}, "
                    f"Offence Loss: {total_loss_offence:.4f}, "
                    f"Touchball Loss: {total_loss_touchball:.4f}, "
                    f"Try To Play Loss: {total_loss_trytoplay:.4f}, "
                    f"Severity Loss: {total_loss_severity:.4f}")  # Log severity loss
    
    # After training, plot the loss curves
    plot_losses(total_losses, actionclass_losses, bodypart_losses, offence_losses, touchball_losses, trytoplay_losses, severity_losses)
    
    return model



def plot_losses(total_losses, actionclass_losses, bodypart_losses, offence_losses, touchball_losses, trytoplay_losses, severity_losses):
    """
    Plot the training loss curves for each task.
    """
    epochs = range(1, len(total_losses) + 1)

    plt.figure(figsize=(12, 8))
    
    # Plot total loss
    plt.subplot(2, 4, 1)
    plt.plot(epochs, total_losses, label='Total Loss', color='blue')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot action class loss
    plt.subplot(2, 4, 2)
    plt.plot(epochs, actionclass_losses, label='Action Class Loss', color='red')
    plt.title('Action Class Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot body part loss
    plt.subplot(2, 4, 3)
    plt.plot(epochs, bodypart_losses, label='Body Part Loss', color='green')
    plt.title('Body Part Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot offence loss
    plt.subplot(2, 4, 4)
    plt.plot(epochs, offence_losses, label='Offence Loss', color='purple')
    plt.title('Offence Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot touchball loss
    plt.subplot(2, 4, 5)
    plt.plot(epochs, touchball_losses, label='Touchball Loss', color='orange')
    plt.title('Touchball Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot try-to-play loss
    plt.subplot(2, 4, 6)
    plt.plot(epochs, trytoplay_losses, label='Try To Play Loss', color='brown')
    plt.title('Try To Play Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot severity loss
    plt.subplot(2, 4, 7)
    plt.plot(epochs, severity_losses, label='Severity Loss', color='cyan')
    plt.title('Severity Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()


# Save model function
def save_model(model, file_path, metadata):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata  # Save metadata (e.g., input size, class sizes)
    }
    torch.save(checkpoint, file_path)
    print(f"Model and metadata saved to {file_path}")

# Load model function
def load_model(file_path):
    checkpoint = torch.load(file_path)
    metadata = checkpoint['metadata']

    # Rebuild model architecture using metadata
    model = MultiTaskModel(
        input_size=metadata['input_size'],
        action_classes=metadata['action_classes'],
        bodypart_classes=metadata['bodypart_classes'],
        offence_classes=metadata['offence_classes'],
        touchball_classes=metadata['touchball_classes'],
        trytoplay_classes=metadata['trytoplay_classes'],
        severity_classes=metadata['severity_classes'] 
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {file_path} with metadata: {metadata}")
    return model


def main():
    logging.basicConfig(level=logging.INFO)
    preprocessor = FoulDataPreprocessor()
    input_file = 'data/dataset/train/train_features.h5'
    
    X_train, y_train = preprocessor.process_data(input_file)
    
    if X_train is not None:
        # Calculate class weights for each task, including severity
        class_weights = {
            'actionclass': preprocessor.get_class_weights(y_train['actionclass'], len(preprocessor.action_class_map)),
            'bodypart': preprocessor.get_class_weights(y_train['bodypart'], len(preprocessor.bodypart_map)),
            'offence': preprocessor.get_class_weights(y_train['offence'], len(preprocessor.offence_map)),
            'touchball': preprocessor.get_class_weights(y_train['touchball'], len(preprocessor.touchball_map)),
            'trytoplay': preprocessor.get_class_weights(y_train['trytoplay'], len(preprocessor.trytoplay_map)),
            'severity': preprocessor.get_class_weights(y_train['severity'], len(preprocessor.severity_map))  # Add severity class weights
        }
    
    model = train_model(X_train, y_train, class_weights, severity_classes=len(preprocessor.severity_map), epochs=100, batch_size=64, learning_rate=0.0005)
    
    metadata = {
        'input_size': X_train.shape[1],
        'action_classes': len(class_weights['actionclass']),
        'bodypart_classes': len(class_weights['bodypart']),
        'offence_classes': len(class_weights['offence']),
        'touchball_classes': len(class_weights['touchball']),
        'trytoplay_classes': len(class_weights['trytoplay']),
        'severity_classes': len(class_weights['severity'])  # Add severity to metadata
    }
    
    save_model(model, "foul_detection_model.pth", metadata)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
