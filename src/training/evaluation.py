"""Model evaluation utilities"""

from torch import no_grad
from sklearn.metrics import precision_recall_fscore_support


def evaluate_model(model, test_loader, loss_func, device, X_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        loss_func: Loss function
        device: Torch device (CPU/GPU)
        X_test: Test dataset (for class names)
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    ##################################################################################################
    # Evaluate model on the test set
    model.eval()   # set model to evaluation mode
    
    test_loss_list = []
    all_predictions = []      # store all predictions
    all_targets = []    # store all true labels
    
    correct = 0
    total_samples = len(X_test)
    class_names = X_test.classes
    
    with no_grad():   # deactivate gradients for model evaluation
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)   # rank-2 tensor. Dimensions: [batch_size, n_classes]
            if output.ndim == 1:   # if output has rank = 1 <== single-image batches
                output = output.reshape(1, *output.shape)   # safety measure for single-image batches
            
            predictions = output.argmax(dim=1)   # select the highest number among dim number 1 of output. Return a batch_size rank-1 tensor
            correct += predictions.eq(target).sum().item()   # compare pred with the batch_size rank-1 tensor "target"
            
            all_predictions.extend(predictions.cpu().numpy())    # sklearn metrics require numpy arrays, not Torch tensors, and numpy works on the cpu
            all_targets.extend(target.cpu().numpy())
            
            loss = loss_func(output, target)
            test_loss_list.append(loss.item())
        
        ##################################################################################################
        # Compute and print metrics
        # Compute average loss
        avg_test_loss = sum(test_loss_list) / len(test_loss_list)
        
        # Compute accuracy
        accuracy = correct / total_samples
        
        # Compute weighted precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average='weighted',
            zero_division=0
        )
        
        # Print metrics
        print(f"Performance on test data:")
        print(f"\tLoss: {avg_test_loss:.4f}")
        print(f"\tAccuracy: {accuracy*100:.1f}% ({correct}/{total_samples})")
        print(f"\tPrecision (weighted): {precision:.4f}")
        print(f"\tRecall (weighted): {recall:.4f}")
        print(f"\tF1-score (weighted): {f1:.4f}")
        
        ##################################################################################################
        # Prepare metrics dictionary
        metrics_dict = {
            'accuracy': accuracy,
            'loss': avg_test_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics_dict
