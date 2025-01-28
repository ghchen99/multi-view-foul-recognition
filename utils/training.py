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

def train_model(X_train, y_train, X_val, y_val, class_weights, severity_classes, 
                epochs=100, batch_size=128, learning_rate=0.0003):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Gradient Accumulation for larger effective batch size
    accum_iter = 4
    effective_batch_size = batch_size * accum_iter
    
    # 2. Create datasets with weighted sampling
    train_weights = compute_sample_weights(y_train)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_weights, len(train_weights), replacement=True
    )
    
    train_dataset = MultiTaskDataset(X_train, y_train)
    val_dataset = MultiTaskDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 3. Model with LayerNorm and residual connections
    model = ImprovedMultiTaskModel(
        input_size=X_train.shape[1],
        action_classes=len(class_weights['actionclass']),
        bodypart_classes=len(class_weights['bodypart']),
        offence_classes=len(class_weights['offence']),
        touchball_classes=len(class_weights['touchball']),
        trytoplay_classes=len(class_weights['trytoplay']),
        severity_classes=severity_classes
    ).to(device)
    
    # 4. Improved learning rate scheduling
    base_lr = learning_rate
    warmup_epochs = 5
    
    # 5. Automatic Mixed Precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. Dynamic task weighting based on loss magnitudes
    task_weights = {task: 1.0 for task in class_weights.keys()}
    
    # 7. Label smoothing in loss function
    criteria = {
        task: FocalLoss(
            weight=class_weights[task].to(device),
            gamma=2 if task in ['actionclass', 'severity'] else 1,
            label_smoothing=0.1
        ) for task in class_weights.keys()
    }
    
    # 8. Layer-wise learning rates
    param_groups = [
        {'params': model.shared_net.parameters(), 'lr': base_lr},
        {'params': model.primary_heads.parameters(), 'lr': base_lr * 1.5},
        {'params': model.severity_head.parameters(), 'lr': base_lr * 2.0}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # 9. Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 10. Improved early stopping with loss plateau detection
    best_val_loss = float('inf')
    patience = 15
    plateau_threshold = 0.001
    plateau_count = 0
    best_model_state = None
    
    history = {
        'train_losses': {task: [] for task in class_weights.keys()},
        'val_losses': {task: [] for task in class_weights.keys()},
        'lrs': []
    }
    
    for epoch in range(epochs):
        # Training phase with gradient accumulation
        model.train()
        train_losses = defaultdict(float)
        optimizer.zero_grad()
        
        for i, (inputs, labels_dict) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                batch_loss = 0
                for task, criterion in criteria.items():
                    task_loss = criterion(outputs[task], labels_dict[task])
                    weighted_loss = task_loss * task_weights[task]
                    batch_loss += weighted_loss
                    train_losses[task] += task_loss.item()
                
                # Scale loss by accumulation factor
                batch_loss = batch_loss / accum_iter
            
            # Accumulate gradients
            scaler.scale(batch_loss).backward()
            
            if (i + 1) % accum_iter == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Validation phase
        val_losses = validate_epoch(model, val_loader, criteria, task_weights, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history['lrs'].append(current_lr)
        
        # Dynamic task weight adjustment
        val_losses_tensor = torch.tensor([val_losses[task] for task in criteria.keys()])
        max_loss = val_losses_tensor.max().item()
        
        for task in task_weights:
            if task == 'severity':
                min_other_loss = min([val_losses[t] for t in criteria.keys() if t != 'severity'])
                task_weights[task] = (val_losses[task] / max_loss) * (1.5 if min_other_loss < 0.1 else 1.0)
            else:
                task_weights[task] = (val_losses[task] / max_loss)
        
        # Update history
        for task in criteria.keys():
            history['train_losses'][task].append(train_losses[task] / len(train_loader))
            history['val_losses'][task].append(val_losses[task])
        
        # Early stopping with plateau detection
        val_loss = sum(val_losses.values())
        if val_loss < best_val_loss - plateau_threshold:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            plateau_count = 0
        else:
            plateau_count += 1
            
        if plateau_count >= patience:
            print(f"Training stopped at epoch {epoch + 1} due to loss plateau")
            break
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] LR: {current_lr:.6f}")
            for task in criteria.keys():
                print(f"{task}: Train = {train_losses[task]/len(train_loader):.4f}, "
                      f"Val = {val_losses[task]:.4f}, Weight = {task_weights[task]:.3f}")
    
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
        class_weights = 1.0 / counts.float()
        class_weights = class_weights / class_weights.sum()
        
        # Apply weights to samples
        sample_weights = class_weights[labels]
        weights *= sample_weights
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    return weights

def save_model(model, file_path, class_weights=None, training_history=None, scaler=None):
    """
    Save the model along with metadata and optional training artifacts.
    """
    # Get base path without extension
    base_path = file_path.rsplit('.', 1)[0]
    
    # Save the training history plot if history exists
    if training_history is not None:
        plot_path = f"{base_path}_training_history.png"
        plot_training_history(training_history, plot_path)
        print(f"Training history plot saved to {plot_path}")
    
    # Get model configuration
    metadata = {
        'model_config': {
            'input_size': model.shared_net[0].in_features,
            'action_classes': model.primary_heads['actionclass'].net[-1].out_features,
            'bodypart_classes': model.primary_heads['bodypart'].net[-1].out_features,
            'offence_classes': model.primary_heads['offence'].net[-1].out_features,
            'touchball_classes': model.primary_heads['touchball'].net[-1].out_features,
            'trytoplay_classes': model.primary_heads['trytoplay'].net[-1].out_features,
            'severity_classes': model.severity_head[-1].out_features
        },
        'architecture_version': '2.0',
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'class_weights': class_weights,
        'training_history': training_history
    }
    
    if scaler is not None:
        checkpoint['scaler'] = scaler
    
    # Save to file
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
        class_weights: Class weights used during training
        training_history: Training metrics history
        scaler: Data scaler if it was saved, None otherwise
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
    
    # Return all saved components
    return (
        model,
        metadata,
        checkpoint.get('class_weights'),
        checkpoint.get('training_history'),
        checkpoint.get('scaler')
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