import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils.logger import logger
from src.stats_engine.guard import DataGuard, StatsGuard


class Orchestrator:
    def __init__(self):
        logger.info("Orchestrator başlatılıyor...")

        self.data_guard = DataGuard()
        self.stats_guard = StatsGuard(use_ewma=True, ewma_alpha=0.2)

        # AI Ekibinin RUL (Kestirimci Bakım) Modeli
        self.rul_model_path = ROOT_DIR / 'models' / 'saved' / 'rul_model.pkl'
        self.rul_model = None
        try:
            import joblib
            self.rul_model = joblib.load(self.rul_model_path)
            logger.info("✅ RUL Tahmin Modeli Yüklendi.")
        except Exception:
            logger.warning(
                "⚠️ RUL Modeli yok/okunamadı. predicted_rul=-1 dönecek.")

    def diagnose(self, data_packet: dict) -> dict:
        # 1. Veri Bütünlüğü Kontrolü (DataGuard)
        if not self.data_guard.validate(data_packet):
            logger.warning("Orchestrator: Geçersiz veri paketi reddedildi.")
            return {
                "status": "INVALID DATA", "priority": 0, "color": "gray",
                "loss": 0.0, "threshold": self.stats_guard.threshold,
                "risk_score": 0.0, "predicted_rul": -1.0
            }

        # 2. RUL Tahmini (AI Modeli)
        predicted_rul = -1.0
        if self.rul_model is not None:
            try:
                df = pd.DataFrame([data_packet])
                rul_input = df.drop(columns=['unit_number', 'cycle'],
                                    errors='ignore')
                predicted_rul = float(self.rul_model.predict(rul_input)[0])
            except Exception as e:
                logger.error(f"RUL Tahmin Hatası: {e}")

        # 3. İstatistiksel Motor (Yeni guard.py'den dönen veriler)
        stats = self.stats_guard.score(data_packet)
        if not stats.get("ok", False):
            return {
                "status": "MISSING FEATURES", "priority": 0, "color": "gray",
                "loss": 0.0, "threshold": self.stats_guard.threshold,
                "risk_score": 0.0, "predicted_rul": predicted_rul
            }

        # 4. İstatistikten Gelen Kararı UI İçin Çevirme (Policy Mapping)
        risk_level = stats["risk_level"]

        level_map = {
            "LOW": ("NORMAL", 1, "green"),
            "MEDIUM": ("MONITORING", 2, "orange"),
            "HIGH": ("WARNING", 3, "orange"),
            "CRITICAL": ("CRITICAL FAILURE", 4, "red"),
        }

        status, prio, color = level_map.get(risk_level, ("UNKNOWN", 0, "gray"))

        # Ek Güvenlik: İstatistik "Normal" dese bile AI "RUL çok düşük" derse önceliği artır
        # Ek Güvenlik 1: RUL 20'nin altındaysa Sarı Alarm (Warning) ver
        if 5 < predicted_rul <= 20 and prio < 4:
            status = "LOW RUL WARNING"
            prio = max(prio, 3)
            color = "orange"

        # Ek Güvenlik 2: RUL 5'in altına düştüyse (Motor ölmek üzere) Kırmızı Alarm ver!
        elif 0 <= predicted_rul <= 5:
            status = "CRITICAL: END OF LIFE"
            prio = 4
            color = "red"
            logger.critical(
                f"RUL Tükendi ({predicted_rul:.1f}). AI Crew göreve çağrılıyor!")

        # 5. Loglama
        current_cycle = data_packet.get('cycle')
        log_msg = f"Cycle {current_cycle} | Risk: {stats['risk_score']:.3f} | Ratio: {stats['ratio']:.2f} | RUL: {predicted_rul:.1f}"

        if prio == 4:
            logger.critical(f"CRITICAL FAILURE | {log_msg}")
        elif prio >= 2:
            logger.warning(f"{status} | {log_msg}")

        # 6. app.py'ın beklediği final sözlük (dictionary)
        return {
            "spe": stats["spe"],
            "threshold": stats["threshold"],  # Orijinal Eşik (Grafik çizecek)
            "ratio": stats["ratio"],  # Ham oran
            "risk_score": stats["risk_score"],
            # Olasılık 0-1 (Evaluator bunu kullanacak)
            "status": status,
            "priority": prio,
            "color": color,
            "predicted_rul": predicted_rul
        }