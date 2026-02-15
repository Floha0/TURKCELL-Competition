import pandas as pd
import numpy as np
import joblib

from src.utils.logger import logger
from config.paths import WATCHDOG_MODEL_PATH, SCALER_PATH

class DataGuard:
    """
    Safety layer to validate sensor data before it enters the AI model.
    Implements 'Fail-Safe' logic ensuring absolute data integrity.
    """

    def __init__(self):
        self.required_sensors = [f'sensor_measurement{i}' for i in range(1, 22)]

    def validate(self, data_packet: dict) -> bool:
        if not data_packet:
            logger.warning("DataGuard: Empty data packet received.")
            return False

        # 1. CYCLE KONTROLÜ
        if 'cycle' not in data_packet:
            logger.error("DataGuard: Critical missing key 'cycle'.")
            return False

        # 2. SENSÖR VARLIK KONTROLÜ
        missing = [s for s in self.required_sensors if s not in data_packet]
        if missing:
            logger.error(f"DataGuard: Missing sensors in cycle {data_packet.get('cycle')}: {missing}")
            return False

        # 3. NULL DEĞER KONTROLÜ
        sensor_vals = [data_packet.get(s) for s in self.required_sensors]
        if any(v is None for v in sensor_vals):
            logger.warning(f"DataGuard: Null sensor value detected in cycle {data_packet.get('cycle')}")
            return False

        # 4. FİZİKSEL SINIR VE TİP KONTROLÜ (Example: T24 Temperature)
        t24_raw = data_packet.get('sensor_measurement2')
        try:
            t24 = float(t24_raw) if t24_raw is not None else None
            if t24 is not None and (t24 < 0 or t24 > 1000):
                logger.critical(f"DataGuard: Sensor 2 (T24) Out of Physical Bounds: {t24}")
                return False
        except (TypeError, ValueError):
            logger.error(f"DataGuard: Sensor 2 (T24) non-numeric value: {t24_raw}")
            return False

        return True

class StatsGuard:
    """
    Statistical Watchdog (Industrial Grade):
    - Multivariate Anomaly Detection via SPE (Squared Prediction Error)
    - Temporal Smoothing via EWMA (Low-pass Filter)
    - Decision Logic via Probabilistic Risk Mapping
    """

    def __init__(self, use_ewma: bool = True, ewma_alpha: float = 0.2):
        self.ready = False
        try:
            self.scaler = joblib.load(SCALER_PATH)
            packet = joblib.load(WATCHDOG_MODEL_PATH)

            self.pca = packet["pca"]
            self.threshold = float(packet["threshold"])
            self.features = packet["selected_features"]
            self.ready = True
            logger.info("StatsGuard: Model and Scaler loaded successfully.")
        except Exception as e:
            logger.error(f"StatsGuard: FAILED to load model components: {e}")

        self.use_ewma = use_ewma
        self.ewma_alpha = float(ewma_alpha)
        self._ewma = None

    def reset(self):
        """Resets the EWMA memory for a clean simulation start."""
        self._ewma = None
        logger.info("StatsGuard: EWMA memory reset.")

    def score(self, data_packet: dict) -> dict:
        # 0. MODEL HAZIRLIK KONTROLÜ
        if not self.ready:
            return {"ok": False, "reason": "MODEL_NOT_READY"}

        # 1. ÖN KONTROL: Eksik özellik kontrolü
        missing = [c for c in self.features if c not in data_packet]
        if missing:
            logger.error(f"StatsGuard: Missing features for PCA: {missing}")
            return {"ok": False, "reason": "MISSING_FEATURES", "missing": missing}

        # 2. VERİ TİPİ GÜVENLİĞİ (Numeric Cast)
        df = pd.DataFrame([data_packet])[self.features]
        df = df.apply(pd.to_numeric, errors="coerce")
        if df.isna().any().any():
            logger.error("StatsGuard: Non-numeric or NaN detected after casting.")
            return {"ok": False, "reason": "NON_NUMERIC_OR_NAN_DETECTED"}

        # 3. SPE (Squared Prediction Error) HESAPLAMA
        X = self.scaler.transform(df)
        X_pca = self.pca.transform(X)
        X_proj = self.pca.inverse_transform(X_pca)

        # SPE (Q-Residual) hesabı
        spe = float(np.mean(np.square(X - X_proj), axis=1)[0])
        ratio = float(spe / self.threshold) if self.threshold > 0 else 0.0

        # 4. TEMPORAL SMOOTHING (EWMA)
        if self.use_ewma:
            if self._ewma is None:
                self._ewma = ratio
            else:
                self._ewma = self.ewma_alpha * ratio + (1.0 - self.ewma_alpha) * self._ewma
            smoothed_ratio = float(self._ewma)
        else:
            smoothed_ratio = ratio

        # 5. RISK MAPPING (Sigmoid)
        risk_score = float(1.0 / (1.0 + np.exp(-10.0 * (smoothed_ratio - 1.0))))

        # 6. HİBRİT KARAR MEKANİZMASI
        if ratio >= 1.5:  # Hard Override (Spike detection)
            level = "CRITICAL"
        else:
            if risk_score >= 0.85: level = "CRITICAL"
            elif risk_score >= 0.60: level = "HIGH"
            elif risk_score >= 0.40: level = "MEDIUM"
            else: level = "LOW"

        return {
            "ok": True,
            "spe": spe,
            "threshold": self.threshold,
            "ratio": ratio,
            "smoothed_ratio": smoothed_ratio,
            "risk_score": risk_score,
            "risk_level": level
        }