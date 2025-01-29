import torch
import logging
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import datetime
from collections import defaultdict
import math
import json
from torch.utils.data import DataLoader, Dataset
from utils.FoulDataPreprocessor import FoulDataPreprocessor

class TaskSpecificHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, dropout_rate=0.5):
        super().__init__()
        if hidden_size is None:
            hidden_size = max(output_size * 2, input_size // 2)
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),  # Changed to GELU for better gradient flow
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class ImprovedMultiTaskModel(nn.Module):
    def __init__(self, input_size, action_classes, bodypart_classes, offence_classes, 
                 touchball_classes, trytoplay_classes, severity_classes, dropout_rate=0.5):
        super().__init__()
        
        # Increased capacity in shared layers
        self.shared_net = nn.Sequential(
            nn.Linear(input_size, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Primary task heads
        self.primary_heads = nn.ModuleDict({
            'actionclass': TaskSpecificHead(384, action_classes, hidden_size=512),
            'bodypart': TaskSpecificHead(384, bodypart_classes, hidden_size=512),
            'offence': TaskSpecificHead(384, offence_classes, hidden_size=384),
            'touchball': TaskSpecificHead(384, touchball_classes, hidden_size=384),
            'trytoplay': TaskSpecificHead(384, trytoplay_classes, hidden_size=256)
        })
        
        # Severity prediction using other task outputs
        total_task_outputs = (action_classes + bodypart_classes + offence_classes + 
                            touchball_classes + trytoplay_classes)
        
        self.severity_head = nn.Sequential(
            nn.Linear(384 + total_task_outputs, 512),  # Concatenated features + task outputs
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, severity_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shared_features = self.shared_net(x)
        
        # Get primary task outputs
        primary_outputs = {
            task: head(shared_features) for task, head in self.primary_heads.items()
        }
        
        # Concatenate all task logits with shared features for severity prediction
        task_logits = torch.cat([primary_outputs[task] for task in self.primary_heads.keys()], dim=1)
        severity_input = torch.cat([shared_features, task_logits], dim=1)
        
        # Add severity prediction
        primary_outputs['severity'] = self.severity_head(severity_input)
        
        return primary_outputs
    
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_dict):
        self.X = torch.FloatTensor(X)
        self.y_dict = {task: torch.LongTensor(labels) for task, labels in y_dict.items()}
        self.tasks = list(y_dict.keys())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = {task: self.y_dict[task][idx] for task in self.tasks}
        return x, y

def plot_training_history(history, save_path):
    """
    Plot and save training and validation losses.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot task-specific losses
    for idx, task in enumerate(history['train_losses'].keys(), 1):
        plt.subplot(3, 2, idx)
        plt.plot(history['train_losses'][task], label='Train')
        plt.plot(history['val_losses'][task], label='Validation')
        plt.title(f'{task.capitalize()} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(X_train, y_train, X_val, y_val, severity_classes, 
                epochs=100, batch_size=128, learning_rate=0.0003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets without weighted sampling
    train_dataset = MultiTaskDataset(X_train, y_train)
    val_dataset = MultiTaskDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model initialization without class weights
    model = ImprovedMultiTaskModel(
        input_size=X_train.shape[1],
        action_classes=max(y_train['actionclass']) + 1,
        bodypart_classes=max(y_train['bodypart']) + 1,
        offence_classes=max(y_train['offence']) + 1,
        touchball_classes=max(y_train['touchball']) + 1,
        trytoplay_classes=max(y_train['trytoplay']) + 1,
        severity_classes=severity_classes
    ).to(device)
    
    # Simple cross entropy loss without weights or focal loss
    criteria = {
        task: nn.CrossEntropyLoss() for task in y_train.keys()
    }
    
    # Basic optimizer setup
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training history tracking
    history = {
        'train_losses': {task: [] for task in y_train.keys()},
        'val_losses': {task: [] for task in y_train.keys()},
        'lrs': []
    }
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    no_improve_count = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = defaultdict(float)
        
        for inputs, labels_dict in train_loader:
            inputs = inputs.to(device)
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate losses without task weights
            batch_loss = 0
            for task, criterion in criteria.items():
                task_loss = criterion(outputs[task], labels_dict[task])
                batch_loss += task_loss
                train_losses[task] += task_loss.item()
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_losses = defaultdict(float)
        
        with torch.no_grad():
            for inputs, labels_dict in val_loader:
                inputs = inputs.to(device)
                labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
                outputs = model(inputs)
                
                for task, criterion in criteria.items():
                    val_losses[task] += criterion(outputs[task], labels_dict[task]).item()
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history['lrs'].append(current_lr)
        
        # Update history
        for task in criteria.keys():
            history['train_losses'][task].append(train_losses[task] / len(train_loader))
            history['val_losses'][task].append(val_losses[task] / len(val_loader))
        
        # Early stopping check
        val_loss = sum(val_losses.values()) / len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] LR: {current_lr:.6f}")
            for task in criteria.keys():
                train_loss = train_losses[task] / len(train_loader)
                val_loss = val_losses[task] / len(val_loader)
                print(f"{task} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def train_epoch(model, train_loader, criteria, optimizer, scheduler, task_weights, device):
    """Run one epoch of training."""
    model.train()
    train_task_losses = {task: 0.0 for task in criteria.keys()}
    num_batches = 0
    
    for inputs, labels_dict in train_loader:
        inputs = inputs.to(device)
        labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute weighted task losses
        batch_loss = 0
        for task, criterion in criteria.items():
            task_loss = criterion(outputs[task], labels_dict[task])
            weighted_loss = task_loss * task_weights[task]
            batch_loss += weighted_loss
            train_task_losses[task] += task_loss.item()
        
        batch_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        num_batches += 1
    
    # Average losses over batches
    return {task: loss / num_batches for task, loss in train_task_losses.items()}

def validate_epoch(model, val_loader, criteria, task_weights, device):
    """Run one epoch of validation."""
    model.eval()
    val_task_losses = {task: 0.0 for task in criteria.keys()}
    num_batches = 0
    
    with torch.no_grad():
        for inputs, labels_dict in val_loader:
            inputs = inputs.to(device)
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            outputs = model(inputs)
            
            for task, criterion in criteria.items():
                task_loss = criterion(outputs[task], labels_dict[task])
                val_task_losses[task] += task_loss.item()
            num_batches += 1
    
    # Average losses over batches
    return {task: loss / num_batches for task, loss in val_task_losses.items()}

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none', label_smoothing=label_smoothing)
        
    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def compute_sample_weights(y_dict):
    """Compute sample weights based on task difficulty and class distribution"""
    weights = torch.ones(len(next(iter(y_dict.values()))))
    
    for task, labels in y_dict.items():
        # Convert to tensor if necessary
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
            
        # Count class frequencies
        unique, counts = torch.unique(labels, return_counts=True)
        total = len(labels)
        class_weights = 1.0 / counts.float()
        class_weights = class_weights / class_weights.sum()
        
        # In train_model, after calculating class weights
        print(f"\n{task}:")
        print("Class frequencies:")
        for cls, count in zip(unique.tolist(), counts.tolist()):
            print(f"  Class {cls}: {count}/{total} ({100*count/total:.2f}%)")
        print("Class weights:")
        for cls, weight in zip(unique.tolist(), class_weights.tolist()):
            print(f"  Class {cls}: {weight:.4f}")
        
        # Apply weights to samples
        sample_weights = class_weights[labels]
        weights *= sample_weights
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    return weights

# Modified save function without class weights
def save_model(model, file_path, training_history=None):
    """
    Save the model along with metadata and training history.
    """
    base_path = file_path.rsplit('.', 1)[0]
    
    if training_history is not None:
        plot_path = f"{base_path}_training_history.png"
        plot_training_history(training_history, plot_path)
    
    metadata = {
        'model_config': {
            'input_size': int(model.shared_net[0].in_features),
            'action_classes': int(model.primary_heads['actionclass'].net[-1].out_features),
            'bodypart_classes': int(model.primary_heads['bodypart'].net[-1].out_features),
            'offence_classes': int(model.primary_heads['offence'].net[-1].out_features),
            'touchball_classes': int(model.primary_heads['touchball'].net[-1].out_features),
            'trytoplay_classes': int(model.primary_heads['trytoplay'].net[-1].out_features),
            'severity_classes': int(model.severity_head[-1].out_features)
        },
        'architecture_version': '2.0',
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'training_history': training_history
    }
    
    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")
    print("Saved metadata:", json.dumps(metadata, indent=2))

def load_model(file_path, device=None):
    """
    Load the model and associated metadata.
    
    Args:
        file_path: Path to the saved model
        device: Optional torch device to load the model to
        
    Returns:
        model: Loaded model
        metadata: Model metadata
        training_history: Training metrics history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(file_path, map_location=device)
    metadata = checkpoint['metadata']
    
    # Version compatibility check
    if metadata.get('architecture_version', '1.0') != '2.0':
        print("Warning: Loading model from different architecture version. Some features might not be available.")
    
    # Initialize model with saved configuration
    model = ImprovedMultiTaskModel(
        **metadata['model_config']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {file_path}")
    print("Loaded metadata:", json.dumps(metadata, indent=2))
    
    # Return only relevant components
    return (
        model,
        metadata,
        checkpoint.get('training_history')
    )


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