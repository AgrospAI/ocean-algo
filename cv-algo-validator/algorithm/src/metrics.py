import numpy as np

def IoU(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask).sum()
    union = np.logical_or(predicted_mask, true_mask).sum()

    if union == 0:
        return 1.0
    
    return np.round(intersection / union, 2)

def dice_coeff(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask).sum()
    total_pixels = predicted_mask.sum() + true_mask.sum()
    
    if total_pixels == 0:
        return 1.0
    
    return np.round((2 * intersection) / total_pixels, 2)

def precision_recall_f1(predicted_mask, true_mask):
    tp = np.logical_and(predicted_mask == 1, true_mask == 1).sum()
    fp = np.logical_and(predicted_mask == 1, true_mask == 0).sum()
    fn = np.logical_and(predicted_mask == 0, true_mask == 1).sum()
    tn = np.logical_and(predicted_mask == 0, true_mask == 0).sum()
    
    if tp + fp == 0:
        precision = 1.0 if fp == 0 else 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 1.0 if fn == 0 else 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return np.round(precision, 2), np.round(recall, 2), np.round(f1, 2)

def accuracy(predicted_mask, true_mask):
    correct = (predicted_mask == true_mask).sum()
    total = true_mask.size
    return np.round(correct / total, 2)


