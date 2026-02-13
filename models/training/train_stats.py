import json

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# --- PROJE YAPILANDIRMASI ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config.paths import TRAIN_FILE, WATCHDOG_MODEL_PATH, SCALER_PATH, SETTINGS_FILE
from src.stats_engine.metrics import PerformanceEvaluator

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def train_watchdog():
    print("\n" + "=" * 50)
    print("🚀 WATCHDOG EĞİTİMİ (FINAL - SAFETY BUFFER)")
    print("=" * 50)

    # 1. VERİ YÜKLEME
    col_names = json.load(open(SETTINGS_FILE)).get('data_col_names')
    try:
        df = pd.read_csv(TRAIN_FILE, sep='\s+', header=None, names=col_names)
    except FileNotFoundError:
        print(f"❌ HATA: Veri dosyası bulunamadı: {TRAIN_FILE}")
        return

    # 2. ÖN İŞLEME
    drop_sensors = json.load(open(SETTINGS_FILE)).get('drop_columns')
    features = [c for c in df.columns if c not in drop_sensors and c not in ['unit_number', 'cycle']]

    # EĞİTİM: İlk 50 döngü (Sağlıklı)
    healthy_data = df[df['cycle'] <= 50][features]

    # TEST: Kesin Sağlamlar (<=50) vs Kesin Bozuklar (>130)
    val_mask = (df['cycle'] <= 50) | (df['cycle'] > 130)
    X_val_raw = df.loc[val_mask, features]
    y_true_val = (df.loc[val_mask, 'cycle'] > 130).astype(int).values

    print(f"✅ Özellik Sayısı: {len(features)}")

    # 3. SCALING (MinMax)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(healthy_data)
    X_val = scaler.transform(X_val_raw)

    # 4. İSTATİSTİKSEL MODEL (PCA - Bottleneck)
    pca = PCA(n_components=2)
    pca.fit(X_train)

    def calculate_reconstruction_error(data):
        data_pca = pca.transform(data)
        data_projected = pca.inverse_transform(data_pca)
        return np.mean(np.square(data - data_projected), axis=1)

    # 5. THRESHOLD TUNING (3-SIGMA KURALI)
    print("⚖️  Eşik Değeri Hesaplanıyor (Mean + 2 Sigma)...")

    train_errors = calculate_reconstruction_error(X_train)

    mu = np.mean(train_errors)  # Ortalama Hata
    std = np.std(train_errors)  # Standart Sapma

    # Normalde 3 sigma kullanılır ama Recall artsın diye 2 sigma yapıyoruz.
    # Bu, verilerin %95'ini normal kabul eder.
    threshold = mu + 2 * std

    print(f"   -> Ortalama: {mu:.6f} | Std Sapma: {std:.6f}")
    print(f"🎯 Hesaplanan Eşik: {threshold:.6f}")


    # 6. TEST
    val_errors = calculate_reconstruction_error(X_val)
    y_pred_val = (val_errors > threshold).astype(int)

    evaluator = PerformanceEvaluator()
    evaluator.y_true = list(y_true_val)
    evaluator.y_pred = list(y_pred_val)
    evaluator.y_prob = list(val_errors)

    recall, acc, f1, auc = evaluator.generate_report()

    # 7. KAYIT (Recall > 0.60 MVP için kabulümüzdür, görsellik önemli)
    model_packet = {
        "pca": pca,
        "threshold": threshold,
        "selected_features": features
    }

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model_packet, WATCHDOG_MODEL_PATH)
    print(f"\n✅ Model Kaydedildi: {WATCHDOG_MODEL_PATH}")


if __name__ == "__main__":
    train_watchdog()