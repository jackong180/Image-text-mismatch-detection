from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def classification_metrics(labels, probabilities, threshold=0.5):
    predictions = [int(p >= threshold) for p in probabilities]
    p,r,f,_ = precision_recall_fscore_support(labels, predictions, average='binary', pos_label=1, zero_division=0)
    return {'n': len(labels), 'threshold': threshold, 'accuracy': float(accuracy_score(labels,predictions)),
            'precision': float(p), 'recall': float(r), 'f1': float(f),
            'roc_auc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
            'confusion_matrix': confusion_matrix(labels,predictions,labels=[0,1]).tolist(),
            'confusion_matrix_order': 'rows=true [0,1]; columns=predicted [0,1]; [[TN,FP],[FN,TP]]'}
