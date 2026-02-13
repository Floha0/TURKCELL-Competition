# PROJE METRİK VE PERFORMANS DEĞERLENDİRME MODÜLÜ
#
# Bu modül, endüstriyel güvenlik odaklı bir yapay zeka modelinin performansını değerlendirmek için tasarlanmıştır.
# Amaç: Modelin endüstriyel güvenilirliğini (Recall, ROC-AUC) kanıtlamaktır.
#
# METRİK STRATEJİSİ:
# 1. Recall (Duyarlılık): Ana metrik. False Negative ayıklama.
# 2. Confusion Matrix: Modelin nerede hata yaptığını gösteren şema.
# 3. Accuracy (Doğruluk): Genel başarı göstergesi (Ancak dengesiz veride yanıltıcı olabilir).
# 4. F1 Score: Precision ve Recall arasındaki denge.
# 5. ROC-AUC: Sınıflandırma kalitesi.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    recall_score, 
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    classification_report
)



class PerformanceEvaluator:
    def __init__(self):
        self.y_true = []      # Ground truth (0: Normal, 1: Failure)
        self.y_pred = []      # Model prediction
        self.y_prob = []      # Confidence score / Probability

    def add_record(self, ground_truth, prediction, probability=None):
        """
        Appends a single simulation step result to the history.
        """
        self.y_true.append(ground_truth)
        self.y_pred.append(prediction)
        
        # Use prediction as probability if no specific score is provided
        if probability is not None: # TODO: if probability?
            self.y_prob.append(probability)
        else:
            self.y_prob.append(prediction)

    def generate_report(self):
        """
        Calculates and prints standard industrial safety metrics.
        Returns: recall, accuracy, f1, auc
        """
        print("\n" + "="*40)
        print("📊 MODEL PERFORMANCE REPORT (MVP)")
        print("="*40)

        # Calculate Core Metrics
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        acc = accuracy_score(self.y_true, self.y_pred)
        
        # Calculate ROC-AUC (Requires both classes to be present in data)
        try:
            auc = roc_auc_score(self.y_true, self.y_prob)
        except ValueError:
            auc = 0.0
            print("(!) Warning: Only one class present in data. AUC cannot be calculated.")

        # Console Output
        print(f"✅ Recall (Safety Critical):      {recall*100:.2f}%")
        print(f"🎯 Accuracy (Baseline):           {acc*100:.2f}%") 
        print(f"⚖️  F1 Score (Balance):           {f1*100:.2f}%")
        print(f"📈 ROC-AUC Score:                 {auc:.4f}")
        
        print("\n" + "-"*40)
        print("🔍 Detailed Classification Report:")
        print(classification_report(self.y_true, self.y_pred, target_names=['Normal', 'Failure'], zero_division=0))
        
        return recall, acc, f1, auc

    def plot_confusion_matrix(self):
        """
        Visualizes the Confusion Matrix to highlight False Negatives.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred: Normal', 'Pred: Failure'],
                    yticklabels=['True: Normal', 'True: Failure'])
        
        plt.title('Confusion Matrix (Zero-Miss Target)')
        plt.ylabel('Ground Truth')
        plt.xlabel('Model Prediction')
        
        plt.tight_layout()
        plt.show()