import torch
import logging
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils.FoulDataPreprocessor import FoulDataPreprocessor
from models.Decoder import Decoder

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

def train_model(X_train, y_train, X_val, y_val, class_weights, severity_classes, epochs=20, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert inputs and labels to tensors and move them to device
    X_train = X_train.to(device)
    y_train = {key: value.to(device) for key, value in y_train.items()}
    X_val = X_val.to(device)
    y_val = {key: value.to(device) for key, value in y_val.items()}
    
    # Initialize the model
    model = MultiTaskModel(
        input_size=X_train.shape[1], 
        action_classes=len(class_weights['actionclass']),
        bodypart_classes=len(class_weights['bodypart']),
        offence_classes=len(class_weights['offence']),
        touchball_classes=len(class_weights['touchball']),
        trytoplay_classes=len(class_weights['trytoplay']),
        severity_classes=severity_classes
    ).to(device)

    # Define loss functions and optimizer
    criteria = {
        'actionclass': nn.CrossEntropyLoss(weight=class_weights['actionclass'].to(device)),
        'bodypart': nn.CrossEntropyLoss(weight=class_weights['bodypart'].to(device)),
        'offence': nn.CrossEntropyLoss(weight=class_weights['offence'].to(device)),
        'touchball': nn.CrossEntropyLoss(weight=class_weights['touchball'].to(device)),
        'trytoplay': nn.CrossEntropyLoss(weight=class_weights['trytoplay'].to(device)),
        'severity': nn.CrossEntropyLoss(weight=class_weights['severity'].to(device))
    }

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    max_grad_norm = 1.0

    # Create DataLoaders for training and validation
    train_dataset = TensorDataset(X_train, y_train['actionclass'], y_train['bodypart'], 
                                y_train['offence'], y_train['touchball'], y_train['trytoplay'], 
                                y_train['severity'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val['actionclass'], y_val['bodypart'], 
                              y_val['offence'], y_val['touchball'], y_val['trytoplay'], 
                              y_val['severity'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize loss tracking
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_losses': {k: [] for k in criteria.keys()},
        'val_losses': {k: [] for k in criteria.keys()}
    }

    def evaluate(model, data_loader, criteria):
        model.eval()
        total_loss = 0
        task_losses = {k: 0 for k in criteria.keys()}
        
        with torch.no_grad():
            for inputs, *labels in data_loader:
                inputs = inputs.to(device)
                labels = [label.to(device) for label in labels]
                outputs = model(inputs)
                
                for i, (task, criterion) in enumerate(criteria.items()):
                    loss = criterion(outputs[i], labels[i])
                    task_losses[task] += loss.item()
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        avg_task_losses = {k: v / len(data_loader) for k, v in task_losses.items()}
        return avg_loss, avg_task_losses

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_task_losses = {k: 0 for k in criteria.keys()}
        
        # Training loop
        for inputs, *labels in train_loader:
            inputs = inputs.to(device)
            labels = [label.to(device) for label in labels]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            batch_loss = 0
            for i, (task, criterion) in enumerate(criteria.items()):
                loss = criterion(outputs[i], labels[i])
                train_task_losses[task] += loss.item()
                batch_loss += loss
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_total_loss += batch_loss.item()

        # Calculate average training losses
        avg_train_loss = train_total_loss / len(train_loader)
        avg_train_task_losses = {k: v / len(train_loader) for k, v in train_task_losses.items()}
        
        # Validation phase
        val_loss, val_task_losses = evaluate(model, val_loader, criteria)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        for task in criteria.keys():
            history['train_losses'][task].append(avg_train_task_losses[task])
            history['val_losses'][task].append(val_task_losses[task])
        
        # Log progress
        logging.info(f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        for task in criteria.keys():
            logging.info(f"{task.capitalize()} - Train: {avg_train_task_losses[task]:.4f}, "
                        f"Val: {val_task_losses[task]:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training and validation losses."""
    plt.figure(figsize=(15, 10))
    
    # Plot total loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Total Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot task-specific losses
    plt.subplot(2, 1, 2)
    for task in history['train_losses'].keys():
        plt.plot(history['train_losses'][task], label=f'{task} (Train)')
        plt.plot(history['val_losses'][task], label=f'{task} (Val)', linestyle='--')
    
    plt.title('Task-Specific Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    # TODO: Add validation data processing
    
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