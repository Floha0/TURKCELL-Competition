"""
Eğer RiskLevel == Low: Logla, geç. (Dashboard'da yeşil ışık yak).

Eğer RiskLevel == Critical: CrewAI'ı tetikle! (Dashboard'da kırmızı alarm ve AI düşünme animasyonu başlat).
"""
import json

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Proje kök dizinini ekle
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config.paths import WATCHDOG_MODEL_PATH, SCALER_PATH, SETTINGS_FILE
from src.utils.logger import logger
from src.stats_engine.guard import DataGuard


class Orchestrator:
    def __init__(self):
        logger.info("Orchestrator başlatılıyor...")
        self.guard = DataGuard()
        self.scaler = self._load_model(SCALER_PATH)
        self.model_packet = self._load_model(WATCHDOG_MODEL_PATH)

        if self.model_packet:
            self.pca = self.model_packet['pca']
            self.threshold = self.model_packet['threshold']
            self.features = self.model_packet['selected_features']
            print(f"✅ Model Yüklendi. Eşik Değeri: {self.threshold:.6f}")
        else:
            raise RuntimeError("Model yüklenemedi! Önce train_stats.py çalıştırın.")

        # Atılacak sensörler (Eğitimdekiyle AYNI olmalı)
        self.drop_sensors = json.load(open(SETTINGS_FILE)).get('drop_columns')

    def _load_model(self, path):
        try:
            return joblib.load(path)
        except FileNotFoundError:
            print(f"❌ HATA: Model dosyası bulunamadı: {path}")
            return None

    def diagnose(self, data_packet):
        """
        Gelen tek satırlık veriyi analiz eder ve Öncelik (Priority) atar.
        """
        if not self.guard.validate(data_packet):
            logger.warning("Orchestrator: Invalid data packet rejected.")
            return {
                "loss": 0.0, "threshold": self.threshold,
                "status": "INVALID DATA", "priority": 0, "color": "gray"
            }

        # 1. Veriyi DataFrame'e çevir (Tek satır)
        df = pd.DataFrame([data_packet])

        # 2. Gereksiz sütunları at (Eğitimdeki feature yapısını koru)
        # Gelen veride 'cycle' veya 'unit_number' olabilir, onları da temizle
        cols_to_drop = [c for c in df.columns if c not in self.features]
        df_filtered = df.drop(columns=cols_to_drop, errors='ignore')

        # Sütun sırasını garantiye al (Eğitimdeki sırayla aynı olmalı)
        df_filtered = df_filtered[self.features]

        # 3. Scaling
        X_scaled = self.scaler.transform(df_filtered)

        # 4. Reconstruction Error Hesapla
        X_pca = self.pca.transform(X_scaled)
        X_projected = self.pca.inverse_transform(X_pca)
        loss = np.mean(np.square(X_scaled - X_projected))

        # 5. Karar Mekanizması (Decoupled Logic)
        result = {
            "loss": loss,
            "threshold": self.threshold,
            "status": "NORMAL",
            "priority": 1,
            "color": "green"
        }

        # Kural Tabanlı Karar
        if loss > (self.threshold * 1.25):  # Eşiğin 2 katıysa KRİTİK
            result["status"] = "CRITICAL FAILURE"
            result["priority"] = 4
            result["color"] = "red"
            logger.warning(f"CRITICAL FAILURE Anomaly detected at Cycle {data_packet['cycle']}! Score: {loss:.4f}" + f"\n Result: {result}")
        elif loss > self.threshold:  # Eşiği geçtiyse UYARI
            result["status"] = "WARNING"
            result["priority"] = 2
            result["color"] = "orange"
            logger.warning(f"WARNING Anomaly detected at Cycle {data_packet['cycle']}! Score: {loss:.4f}" + f"\n Result: {result}")


        return result