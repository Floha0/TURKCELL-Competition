import streamlit
# TODO: streamlit entegrasyonu / görselleştirme
"""
Görselleştirme:

    Sol Panel: Motorun anlık sensör değerleri (Hız göstergeleri gibi gauge chartlar).

    Orta Panel: Risk Grafiği (Zamanla değişen risk skoru).

    Sağ Panel (AI Logs): "Sistem Stabil..." diye başlar, hata anında "AI Analiz Ediyor..." yazar ve CrewAI'dan gelen metni daktilo efektiyle basar.

    Alt Panel: Confusion Matrix görseli (Modelinizin ne kadar güvenilir olduğunu statik olarak gösterir).
"""


"""
 Gemini said

Harika bir seçim. NASA'nın CMAPSS (Commercial Modular Aero-Propulsion System Simulation) veri seti bu iş için "altın standart"tır. Özellikle RUL (Remaining Useful Life) tahmini ve anomali tespiti için mükemmeldir. MVP (Minimum Viable Product) olduğu için mimariyi biraz daha "gösterişe uygun" (demo-friendly) ve yönetilebilir hale getireceğiz.

Gerçek zamanlı sensör yerine "Simüle Edilmiş Akış" (Stream Simulation) kullanacağız. Yani veriyi satır satır okuyup sanki o an geliyormuş gibi sisteme sokacağız.

İşte NASA CMAPSS verisiyle çalışacak, Python tabanlı MVP Proje Şeması:
I. Proje Klasör Yapısı (Directory Tree)

Bu yapı, hem model eğitimini (offline) hem de canlı demoyu (online) kapsar.
Plaintext

jet_engine_guard/
├── config/
│   ├── settings.yaml        # Simülasyon hızı, sensör kolon isimleri
│   └── agents.yaml          # CrewAI ajanlarının promptları ve rolleri
├── data/
│   ├── raw/                 # NASA CMAPSS dosyaları (train_FD001.txt vb.)
│   ├── processed/           # Eğitim için temizlenmiş/hazırlanmış veri
│   └── logs/                # MVP sırasında oluşan canlı loglar (json_db)
├── models/
│   ├── saved/               # Eğitilen .pkl modelleri (Scaler, PCA, Classifier)
│   └── training/            # Modelleri eğiten scriptler (OFFLINE AŞAMA)
│       ├── train_stats.py   # İstatistiksel model (Recall odaklı)
│       └── train_rul.py     # Opsiyonel: Kalan ömür tahmini modeli
├── src/
│   ├── simulation/          # Veri akışını simüle eden modül
│   │   └── streamer.py      # Dosyadan satır satır veri okuyan "Generator"
│   ├── stats_engine/        # The Watchdog (Bekçi)
│   │   ├── metrics.py       # Recall, ROC/AUC hesaplamaları
│   │   └── guard.py         # Gelen veriyi eşik değeriyle kontrol eden yer
│   ├── ai_core/             # The Brain (AI)
│   │   ├── crew.py          # CrewAI orkestrasyonu
│   │   └── tools.py         # AI'ın kullanacağı özel araçlar
│   ├── orchestrator/        # Karar mekanizması (Priority 1-4 ataması)
│   │   └── manager.py
│   └── utils/
│       ├── logger.py        # JSON loglama sistemi
│       └── visualizer.py    # Grafik çizim yardımcıları
├── dashboard/               # Arayüz Katmanı
│   └── app.py               # Streamlit ana uygulaması (Çalıştırılacak dosya)
├── requirements.txt
└── README.md

II. Modül Detayları ve Sorumluluklar

MVP'de iki ana aşama olacak: 1. Hazırlık (Eğitim) ve 2. Canlı Demo (Run-time). Kodları buna göre ayırıyoruz.
A. Hazırlık Aşaması (models/training/)

Yarışma öncesi çalıştırıp modelleri kaydedeceğiniz yer.

    train_stats.py:

        Girdi: NASA train_FD001.txt verisi.

        İşlem:

            Sensör verilerini temizler.

            Basit bir "Anomaly Detection" modeli eğitir (Örn: One-Class SVM veya basit bir Thresholding/Mahalanobis Distance).

            Kritik Nokta: recall_score takıntılı optimizasyon burada yapılır. False Negative (Hatayı kaçırma) cezasını çok yüksek tutarak threshold (eşik) belirlenir.

            Örn: "Sensör 2, 30 birim saparsa hata ver" kuralını matematiksel olarak çıkarır.

        Çıktı: watchdog_model.pkl ve scaler.pkl dosyalarını models/saved/ altına kaydeder.

B. Canlı Demo Aşaması (Runtime)
1. Simülasyon (src/simulation/streamer.py)

Gerçek sensörü taklit eder.

    Fonksiyon: stream_engine_data(engine_id=1)

    Görevi: NASA test setinden seçilen bir motorun verisini alır. Her çağrıldığında bir sonraki zaman döngüsünü (cycle) yield eder (döndürür).

    Amaç: Streamlit arayüzü her "refresh" yaptığında yeni bir saniye geçmiş gibi veri sağlar.

2. The Watchdog (src/stats_engine/guard.py)

İlk savunma hattı.

    Girdi: Streamer'dan gelen tek satırlık sensör verisi.

    İşlem: saved/watchdog_model.pkl'i yükler ve veriyi sorar: "Bu normal mi?".

    Recall Odaklı Mantık: Eğer model %1 bile şüphelenirse, bunu "Priority 3" veya "Priority 4" olarak etiketler. Güvenliği elden bırakmaz.

    Çıktı: RiskLevel (Low, Medium, High, Critical).

3. Orchestrator (src/orchestrator/manager.py)

Trafik polisi.

    Mantık:

        Eğer RiskLevel == Low: Logla, geç. (Dashboard'da yeşil ışık yak).

        Eğer RiskLevel == Critical: CrewAI'ı tetikle! (Dashboard'da kırmızı alarm ve AI düşünme animasyonu başlat).

4. The Brain (src/ai_core/crew.py)

Sadece sorun olduğunda devreye giren akıllı ekip.

    Agentlar:

        Sensor Analyst Agent: "Sensör 11 ve 12 artarken Sensör 7 düşmüş, bu kompresör arızasına işaret ediyor olabilir." yorumunu yapar.

        Maintenance Planner Agent: "Bu motorun acil bakıma girmesi lazım, şu anki uçuş döngüsü tamamlanınca hangara çekin." aksiyonunu önerir.

    Kullanım: LangChain üzerinden bu ajanlar birbirine data paslar ve final raporu oluşturur.

5. Dashboard (dashboard/app.py)

Jürinin göreceği ekran. Streamlit kütüphanesi kullanılacak.

    Görselleştirme:

        Sol Panel: Motorun anlık sensör değerleri (Hız göstergeleri gibi gauge chartlar).

        Orta Panel: Risk Grafiği (Zamanla değişen risk skoru).

        Sağ Panel (AI Logs): "Sistem Stabil..." diye başlar, hata anında "AI Analiz Ediyor..." yazar ve CrewAI'dan gelen metni daktilo efektiyle basar.

        Alt Panel: Confusion Matrix görseli (Modelinizin ne kadar güvenilir olduğunu statik olarak gösterir).

III. Proje Akış Senaryosu (Demo Sırası)

Bunu yarışmada sunarken şu sırayla çalıştıracaksınız:

    Terminal: python models/training/train_stats.py (Modelleri eğittiniz, bitti).

    Terminal: streamlit run dashboard/app.py (Arayüz açıldı).

    Ekranda:

        "Start Simulation" butonuna basılır.

        Grafikler NASA verisiyle oynamaya başlar (Motor çalışıyor).

        İlk 50 döngü (cycle) her şey yeşil. (Priority 4-3-2 logic çalışmıyor, sadece izliyor).

        Döngü 60'ta sensörlerde hafif sapma başlar.

        Stats Engine: "Şüpheli durum!" der. (Priority 3).

        Dashboard: Sarı ışık yanar. "AI Beklemede (Idle Analysis)" yazar.

        Döngü 85'te değerler kopar.

        Stats Engine: "KRİTİK HATA! RUL (Kalan Ömür) < 10 Cycle!" der. (Priority 4).

        Orchestrator: AI Ajanlarını göreve çağırır.

        Ekranda: CrewAI çalışır, analiz metnini ekrana basar: "Yakıt pompası basıncı kritik seviyede. Patlama riski var. Sistemi derhal kapatıyorum."
"""