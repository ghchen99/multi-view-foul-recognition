import torch
import logging
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import datetime
from collections import defaultdict
from torch.nn import functional as F
import math
import json
from collections import Counter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
    
class BalancedMultiTaskDataset(Dataset):
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

    def get_sampler(self, task_weights=None):
        """
        Create a weighted sampler that balances multiple tasks simultaneously.
        task_weights: dict of float values to weight the importance of each task
        """
        if task_weights is None:
            task_weights = {task: 1.0 for task in self.tasks}

        # Calculate weights for each sample based on all its labels
        sample_weights = torch.zeros(len(self))
        
        for task in self.tasks:
            labels = self.y_dict[task].numpy()
            class_counts = Counter(labels)
            n_samples = len(labels)
            
            # Weight for each class is inverse to its frequency
            class_weights = {cls: n_samples / (count * len(class_counts)) 
                           for cls, count in class_counts.items()}
            
            # Apply weights to each sample
            task_sample_weights = torch.tensor([class_weights[label.item()] 
                                             for label in self.y_dict[task]])
            
            # Add to total weights, weighted by task importance
            sample_weights += task_sample_weights * task_weights[task]

        return WeightedRandomSampler(sample_weights, len(sample_weights))

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def create_weighted_loss(class_counts, use_focal=True, gamma=2.0):
    """Create weighted Focal Loss for better handling of minority classes."""
    num_classes = max(class_counts.keys()) + 1
    total_samples = sum(class_counts.values())
    max_count = max(class_counts.values())
    
    # Initialize weights for all possible classes
    weights = torch.ones(num_classes)
    
    # More aggressive weighting formula with better minority class handling
    for class_idx, count in class_counts.items():
        # Combination of inverse frequency and square root scaling
        ratio = (max_count / count) ** 0.5
        minority_factor = 1 + math.log(max_count / count)
        weights[class_idx] = max(ratio * minority_factor, 1.0)
    
    # Normalize weights to prevent loss scaling issues
    weights = weights / weights.mean()
    
    # Print detailed class weight information
    weight_info = {idx: {
        'weight': round(float(w), 3),
        'count': class_counts.get(idx, 0)
    } for idx, w in enumerate(weights)}
    print(f"\nClass weights and counts:")
    print(json.dumps(weight_info, indent=2))
    
    if use_focal:
        return FocalLoss(weight=weights.float(), gamma=gamma)
    return torch.nn.CrossEntropyLoss(weight=weights.float())

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
    
    # Create datasets with balanced sampling
    train_dataset = BalancedMultiTaskDataset(X_train, y_train)
    val_dataset = BalancedMultiTaskDataset(X_val, y_val)
    
    # Task importance weights with stronger emphasis on imbalanced tasks
    task_weights = {
        'severity': 3.0,  # Much higher weight for severity due to critical red/yellow card cases
        'actionclass': 2.5,  # Higher weight due to many minority classes
        'bodypart': 1.5,
        'offence': 2.0,  # Important for foul detection
        'touchball': 1.0,
        'trytoplay': 1.0
    }
    
    # Create weighted sampler for training
    train_sampler = train_dataset.get_sampler(task_weights)
    
    # Calculate effective batch size and samples per epoch
    min_samples_per_class = 50  # Minimum times we want to see each class per epoch
    max_class_counts = {
        task: len(set(y_train[task])) for task in y_train.keys()
    }
    
    # Adjust batch size down for better granularity
    adjusted_batch_size = min(32, batch_size)  # Smaller batches
    
    # Calculate minimum samples needed per epoch
    min_samples = max(
        min_samples_per_class * max(max_class_counts.values()),  # Enough to see each class
        len(X_train)  # At least one full pass through dataset
    )
    
    # Create data loaders with replacement sampling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=adjusted_batch_size,
        sampler=WeightedRandomSampler(
            weights=train_sampler.weights,
            num_samples=min_samples,
            replacement=True
        ),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = ImprovedMultiTaskModel(
        input_size=X_train.shape[1],
        action_classes=max(y_train['actionclass']) + 1,
        bodypart_classes=max(y_train['bodypart']) + 1,
        offence_classes=max(y_train['offence']) + 1,
        touchball_classes=max(y_train['touchball']) + 1,
        trytoplay_classes=max(y_train['trytoplay']) + 1,
        severity_classes=severity_classes
    ).to(device)
    
    # Create weighted loss functions for each task
    criteria = {}
    for task in y_train.keys():
        class_counts = Counter(y_train[task])
        criteria[task] = create_weighted_loss(class_counts).to(device)
        print(f"Created loss function with weights shape: {criteria[task].weight.shape}")
    
    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = epochs * len(train_loader)
    warmup_steps = total_steps // 10
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history tracking
    history = {
        'train_losses': {task: [] for task in y_train.keys()},
        'val_losses': {task: [] for task in y_train.keys()},
        'train_metrics': {task: [] for task in y_train.keys()},
        'val_metrics': {task: [] for task in y_train.keys()},
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
        train_losses = {task: 0.0 for task in y_train.keys()}
        train_correct = {task: 0 for task in y_train.keys()}
        train_total = {task: 0 for task in y_train.keys()}
        
        for batch_idx, (inputs, labels_dict) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            batch_loss = 0
            # Calculate weighted loss for each task
            for task in y_train.keys():
                task_loss = criteria[task](outputs[task], labels_dict[task])
                batch_loss += task_weights[task] * task_loss
                train_losses[task] += task_loss.item()
                
                # Calculate accuracy
                _, predicted = outputs[task].max(1)
                train_correct[task] += predicted.eq(labels_dict[task]).sum().item()
                train_total[task] += labels_dict[task].size(0)
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # Validation phase
        model.eval()
        val_losses = {task: 0.0 for task in y_train.keys()}
        val_correct = {task: 0 for task in y_train.keys()}
        val_total = {task: 0 for task in y_train.keys()}
        
        with torch.no_grad():
            for inputs, labels_dict in val_loader:
                inputs = inputs.to(device)
                labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
                outputs = model(inputs)
                
                for task in y_train.keys():
                    val_losses[task] += criteria[task](outputs[task], labels_dict[task]).item()
                    _, predicted = outputs[task].max(1)
                    val_correct[task] += predicted.eq(labels_dict[task]).sum().item()
                    val_total[task] += labels_dict[task].size(0)
        
        # Update history
        current_lr = scheduler.get_last_lr()[0]
        history['lrs'].append(current_lr)
        
        for task in y_train.keys():
            # Update losses
            history['train_losses'][task].append(train_losses[task] / len(train_loader))
            history['val_losses'][task].append(val_losses[task] / len(val_loader))
            
            # Update metrics (accuracy)
            train_acc = train_correct[task] / train_total[task]
            val_acc = val_correct[task] / val_total[task]
            history['train_metrics'][task].append(train_acc)
            history['val_metrics'][task].append(val_acc)
        
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
            for task in y_train.keys():
                train_loss = train_losses[task] / len(train_loader)
                val_loss = val_losses[task] / len(val_loader)
                train_acc = train_correct[task] / train_total[task]
                val_acc = val_correct[task] / val_total[task]
                print(f"{task} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"{task} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

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