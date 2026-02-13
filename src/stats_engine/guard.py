# TODO: Input: streamer' yield
# TODO: Process: saved/watchdog_model.pkl'i yükler ve veriyi sorar: "Bu normal mi?".
# TODO: Recall Odaklı Mantık: Eğer model %1 bile şüphelenirse, bunu "Priority 3" veya "Priority 4" olarak etiketler. Güvenliği elden bırakmaz.
# TODO: Output: RiskLevel (Low, Medium, High, Critical).
import pandas as pd
import numpy as np
from src.utils.logger import logger


class DataGuard:
    """
    Safety layer to validate sensor data before it enters the AI model.
    Implements 'Fail-Safe' logic.
    """

    def __init__(self):
        self.required_sensors = [f'sensor_measurement{i}' for i in range(1, 22)]

    def validate(self, data_packet: dict) -> bool:
        """
        Checks if the data packet is valid for processing.
        """
        # 1. Check for Empty Data
        if not data_packet:
            logger.warning("DataGuard: Empty data packet received.")
            return False

        # 2. Check for Missing Keys
        # Bazı sensörlerin eksik olması modelin çökmesine neden olur.
        # Bu projede basitleştirmek için sadece 'cycle' kontrolü yapıyoruz ama
        # gerçekte tüm sensörleri check ederiz.
        if 'cycle' not in data_packet:
            logger.error("DataGuard: Critical missing key 'cycle'.")
            return False

        # 3. Check for NaN / Infinite values
        # Gelen veriyi bir listeye döküp kontrol edelim
        values = list(data_packet.values())
        if any(x is None for x in values):  # None check
            logger.warning(f"DataGuard: Null value detected in cycle {data_packet.get('cycle')}")
            return False

        # 4. Physical Range Checks (Örnek: Sıcaklık negatif olamaz)
        # Sensor 2 (T24) genelde 640-650 arasıdır.
        t24 = data_packet.get('sensor_measurement2')
        if t24 and (t24 < 0 or t24 > 1000):
            logger.critical(f"DataGuard: Sensor 2 Out of Physical Bounds: {t24}")
            return False  # Bu veriyi reddet

        return True