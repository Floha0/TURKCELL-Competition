"""
TODO: Input: train_FD.txt
TODO: Process: Remove noise,
TODO: "Basit bir "Anomaly Detection" modeli eğitir (Örn: One-Class SVM veya basit bir Thresholding/Mahalanobis Distance).",
TODO: recall_score takıntılı optimizasyon burada yapılır. False Negative (Hatayı kaçırma) cezasını çok yüksek tutarak threshold (eşik) belirlenir.
Örn: "Sensör 2, 30 birim saparsa hata ver" kuralını matematiksel olarak çıkarır.
TODO: Çıktı: watchdog_model.pkl ve scaler.pkl dosyalarını models/saved/ altına kaydeder.
"""

"""
Hedef: Hangi verinin normal, hangi verinin "Priority 3/4" olduğunu bilen basit bir model kurmak.
Dosya: models/training/train_stats.py

AI henüz yok. Sadece matematik konuşacak.

Yapılacaklar:

    Veriyi normalleştir (MinMaxScaler).

    Basit bir "Normal Davranış Profili" çıkar. (Örn: Motorun sağlıklı olduğu ilk 50 döngünün ortalamasını ve standart sapmasını al).

    Bir threshold (eşik) belirle. Eğer yeni veri (Ortalama + 3 * Standart Sapma) değerini geçerse -> ANOMALİ.

    Bu modeli (scaler ve threshold değerlerini) .pkl veya .json olarak kaydet.
"""