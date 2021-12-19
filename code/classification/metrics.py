import torch
import torch.nn.functional as F

def compute_metrics(pred_list, gt_list, num_classes=4):
    epsilon = 1e-9

    pred_tensor = torch.tensor(pred_list).squeeze().long()
    gt_tensor = torch.tensor(gt_list).squeeze().long()
    
    pred_tensor = F.one_hot(pred_tensor, num_classes).to(torch.float32)
    gt_tensor = F.one_hot(gt_tensor, num_classes).to(torch.float32)
    
    tp = (gt_tensor * pred_tensor).sum(dim=0)
    fp = ((1 - gt_tensor) * pred_tensor).sum(dim=0)
    fn = (gt_tensor * (1 - pred_tensor)).sum(dim=0)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)

    return f1.cpu().numpy(), precision.cpu().numpy(), recall.cpu().numpy()