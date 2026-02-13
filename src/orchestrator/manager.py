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

        # --- YENİ: RUL Modelini Yükle ---
        # RUL model yolunu manuel oluşturuyoruz (config'de yoksa diye garantiye alalım)
        self.rul_model_path = ROOT_DIR / 'models' / 'saved' / 'rul_model.pkl'
        self.rul_model = self._load_model(self.rul_model_path)

        if self.model_packet:
            self.pca = self.model_packet['pca']
            self.threshold = self.model_packet['threshold']
            self.features = self.model_packet['selected_features']
            print(f"✅ Model Yüklendi. Eşik Değeri: {self.threshold:.6f}")
        else:
            raise RuntimeError("Model yüklenemedi! Önce train_stats.py çalıştırın.")

        if self.rul_model:
            print("✅ RUL Tahmin Modeli Yüklendi.")
        else:
            print("⚠️ UYARI: RUL Modeli bulunamadı. Tahminler -1 dönecek.")

        # Atılacak sensörler (Eğitimdekiyle AYNI olmalı)
        # SENİN ÖZELLEŞTİRMEN:
        try:
            self.drop_sensors = json.load(open(SETTINGS_FILE)).get('drop_columns')
        except:
            self.drop_sensors = []  # Dosya yoksa boş geç

    def _load_model(self, path):
        try:
            return joblib.load(path)
        except FileNotFoundError:
            # RUL modeli opsiyonel olabilir, o yüzden sadece print basıyoruz, hata fırlatmıyoruz
            print(f"⚠️ Bilgi: Model dosyası bulunamadı: {path}")
            return None

    def diagnose(self, data_packet):
        """
        Gelen tek satırlık veriyi analiz eder ve Öncelik (Priority) atar.
        """
        if not self.guard.validate(data_packet):
            logger.warning("Orchestrator: Invalid data packet rejected.")
            return {
                "loss": 0.0, "threshold": self.threshold,
                "status": "INVALID DATA", "priority": 0, "color": "gray",
                "predicted_rul": 0  # Hata durumunda 0
            }

        # 1. Veriyi DataFrame'e çevir (Tek satır)
        df = pd.DataFrame([data_packet])

        # --- YENİ: RUL Tahmini (Filtrelemeden ÖNCE yapılmalı) ---
        # RUL Modeli genelde tüm sensörlerle eğitilir, o yüzden ham df'yi veriyoruz.
        predicted_rul = -1.0
        if self.rul_model:
            try:
                # Cycle ve Unit Number tahmine girmemeli
                rul_input = df.drop(columns=['unit_number', 'cycle'], errors='ignore')
                predicted_rul = float(self.rul_model.predict(rul_input)[0])
            except Exception as e:
                logger.error(f"RUL Tahmin Hatası: {e}")

        # 2. Gereksiz sütunları at (Eğitimdeki feature yapısını koru - SENİN KODUN)
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
            "color": "green",
            "predicted_rul": predicted_rul  # YENİ ÇIKTI
        }

        # Kural Tabanlı Karar
        if loss > (self.threshold * 1.25):  # Eşiğin 1.25 katıysa KRİTİK
            result["status"] = "CRITICAL FAILURE"
            result["priority"] = 4
            result["color"] = "red"
            logger.warning(f"CRITICAL FAILURE Anomaly detected at Cycle {data_packet['cycle']}! Score: {loss:.4f}")
        elif loss > self.threshold:  # Eşiği geçtiyse UYARI
            result["status"] = "WARNING"
            result["priority"] = 2
            result["color"] = "orange"
            logger.warning(f"WARNING Anomaly detected at Cycle {data_packet['cycle']}! Score: {loss:.4f}")

        # --- YENİ: RUL Tabanlı Ekstra Güvenlik Kontrolü ---
        # Eğer Anomali Skoru henüz kırmızı yakmadıysa AMA RUL çok düştüyse Sarı yak.
        if 0 < predicted_rul < 20 and result["priority"] < 2:
            result["status"] = "LOW RUL WARNING"
            result["priority"] = 2
            result["color"] = "orange"
            logger.info(f"Low RUL detected ({predicted_rul:.1f}). Elevating status to WARNING.")

        return result