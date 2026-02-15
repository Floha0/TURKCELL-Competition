# PROJE METRİK VE PERFORMANS DEĞERLENDİRME MODÜLÜ
#
# Bu modül, endüstriyel güvenlik odaklı bir yapay zeka modelinin performansını değerlendirmek için tasarlanmıştır.
# Amaç: Modelin endüstriyel güvenilirliğini (Recall, ROC-AUC) kanıtlamaktır.
#
# WATCHDOG / PdM EK METRİKLERİ:
# - Lead Time: Arızadan kaç cycle önce ilk uyarıyı verdik?
# - First Warning Cycle: İlk alarm cycle'ı
# - False Alarm Rate (/100 cycle): failure window dışında verilen alarm oranı

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
    def __init__(self, fail_window: int = 10):
        # Klasik metrikler
        self.y_true = []      # Ground truth (0: Normal, 1: Failure)
        self.y_pred = []      # Model prediction
        self.y_prob = []      # Confidence score / Probability

        # Watchdog/PdM metrikleri
        self.cycles = []                      # Her adımın cycle bilgisi (opsiyonel)
        self.fail_cycle = None                # Gerçek arıza cycle'ı (set edilirse)
        self.fail_window = int(fail_window)   # fail_cycle öncesi "failure window" uzunluğu

        self.first_warning_cycle = None
        self.total_triggers = 0
        self.false_alarm_count = 0

    def set_fail_cycle(self, fail_cycle: int):
        """Gerçek arıza cycle'ını (dossier'daki FAIL CYCLE) evaluator'a bildir."""
        self.fail_cycle = int(fail_cycle)

    def add_record(self, ground_truth, prediction, probability=None, cycle: int = None):
        """
        Appends a single simulation step result to the history.

        cycle verilirse watchdog metrikleri de hesaplanır.
        """
        gt = int(ground_truth)
        pred = int(prediction)

        self.y_true.append(gt)
        self.y_pred.append(pred)

        if probability is not None:
            self.y_prob.append(float(probability))
        else:
            self.y_prob.append(float(pred))

        # Watchdog metrikleri (cycle opsiyonel)
        if cycle is not None:
            c = int(cycle)
            self.cycles.append(c)

            if pred == 1:
                self.total_triggers += 1

                if self.first_warning_cycle is None:
                    self.first_warning_cycle = c

                # fail_cycle set ise: failure window DIŞINDA alarm = false alarm
                if self.fail_cycle is not None:
                    in_failure_window = c >= (self.fail_cycle - self.fail_window)
                    if not in_failure_window:
                        self.false_alarm_count += 1

    # --- Watchdog metrik yardımcıları ---
    def calculate_lead_time(self):
        """fail_cycle - first_warning_cycle"""
        if self.fail_cycle is None or self.first_warning_cycle is None:
            return None
        return int(self.fail_cycle - self.first_warning_cycle)

    def calculate_false_alarm_rate_per_100(self):
        """Failure window dışında verilen alarm oranı (100 cycle başına)."""
        denom = len(self.cycles) if self.cycles else len(self.y_pred)
        if denom == 0:
            return 0.0
        return (self.false_alarm_count / denom) * 100.0

    def generate_report(self):
        """
        Calculates and prints standard industrial safety metrics + Watchdog metrics.
        Returns: recall, accuracy, f1, auc, lead_time, false_alarm_rate
        """
        print("\n" + "=" * 46)
        print("📊 MODEL PERFORMANCE REPORT (MVP + WATCHDOG)")
        print("=" * 46)

        # --- Core Metrics ---
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        acc = accuracy_score(self.y_true, self.y_pred)

        try:
            auc = roc_auc_score(self.y_true, self.y_prob)
        except ValueError:
            auc = 0.0
            print("(!) Warning: Only one class present in data. AUC cannot be calculated.")

        print(f"✅ Recall (Safety Critical):      {recall * 100:.2f}%")
        print(f"🎯 Accuracy (Baseline):           {acc * 100:.2f}%")
        print(f"⚖️  F1 Score (Balance):           {f1 * 100:.2f}%")
        print(f"📈 ROC-AUC Score:                 {auc:.4f}")

        # --- Watchdog Metrics ---
        lead_time = self.calculate_lead_time()
        far_100 = self.calculate_false_alarm_rate_per_100()

        print("\n" + "-" * 46)
        print("🛡️ WATCHDOG / PdM METRICS")
        if self.fail_cycle is None:
            print("ℹ️ fail_cycle set edilmedi (Lead Time hesaplanamaz).")
        else:
            print(f"🧨 Fail Cycle:                    {self.fail_cycle}")

        if self.first_warning_cycle is None:
            print("🚨 First Warning Cycle:           N/A (hiç alarm yok)")
            print("⏱️ Lead Time (cycles):            N/A")
        else:
            print(f"🚨 First Warning Cycle:           {self.first_warning_cycle}")
            print(f"⏱️ Lead Time (cycles):            {lead_time}")

        print(f"📉 False Alarm Rate (/100 cycle): {far_100:.2f}")
        print(f"🔔 Total Triggers:                {self.total_triggers}")

        print("\n" + "-" * 46)
        print("🔍 Detailed Classification Report:")
        print(classification_report(
            self.y_true,
            self.y_pred,
            target_names=['Normal', 'Failure'],
            zero_division=0
        ))

        return recall, acc, f1, auc, lead_time, far_100

    def plot_confusion_matrix(self):
        """
        Visualizes the Confusion Matrix to highlight False Negatives.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred: Normal', 'Pred: Failure'],
            yticklabels=['True: Normal', 'True: Failure']
        )

        plt.title('Confusion Matrix (Zero-Miss Target)')
        plt.ylabel('Ground Truth')
        plt.xlabel('Model Prediction')

        plt.tight_layout()
        plt.show()
