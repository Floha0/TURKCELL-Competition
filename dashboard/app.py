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
│   ├── settings.json        # Simülasyon hızı, sensör kolon isimleri
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

import streamlit as st
import pandas as pd
import time
import altair as alt
import sys
from pathlib import Path
import os
# --- PATH AYARLARI ---
# Dashboard klasöründen bir üst dizine (kök) çıkıyoruz
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ai_core.crew import JetEngineCrew
from src.simulation.streamer import SensorStreamer
from src.orchestrator.manager import Orchestrator
from src.utils.visualizer import DashboardVisualizer
from src.stats_engine.metrics import PerformanceEvaluator
from src.utils.logger import logger
from src.stats_engine.guard import DataGuard


os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA" # Sahte key, CrewAI kontrolünü geçmek için

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="JetEngine Guard AI", page_icon="✈️", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .status-normal { color: #00FF00; font-weight: bold; }
    .status-warning { color: #FFA500; font-weight: bold; }
    .status-critical { color: #FF0000; font-weight: bold; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# --- BAŞLIK ---
st.title("✈️ JetEngine Guard: Autonomous AI Defense System")
st.markdown("**Real-Time Anomaly Detection & Generative AI Diagnostics**")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Kontrol Paneli")
    engine_id = st.number_input("Motor ID", 1, 100, 1)
    speed = st.slider("Simülasyon Hızı", 0.01, 1.0, 0.05)
    start_btn = st.button("🚀 SİSTEMİ BAŞLAT", type="primary")

    st.divider()
    # Metrikleri göstermek için yer tutucular
    st.subheader("📈 Canlı Performans")
    metric_cycle = st.empty()
    metric_accuracy = st.empty()
    metric_recall = st.empty()

# --- ARAYÜZ YERLEŞİMİ ---
col1, col2, col3, col4 = st.columns(4)
with col1: cycle_disp = st.empty()
with col2: status_disp = st.empty()
with col3: loss_disp = st.empty()
with col4: ai_disp = st.empty()

st.divider()
col_left, col_right = st.columns([2, 1])
with col_left:
    st.subheader("📊 Sensör Anomali Grafiği")
    chart_placeholder = st.empty()
with col_right:
    st.subheader("🧠 AI Analiz Konsolu")
    ai_log = st.container(height=400)

# --- ANA DÖNGÜ ---
if start_btn:
    logger.info("Simülasyon başlatıldı.")

    # 1. Modülleri Başlat
    streamer = SensorStreamer(engine_id=engine_id)
    orchestrator = Orchestrator()
    guard = DataGuard()  # YENİ: Guard
    evaluator = PerformanceEvaluator()  # YENİ: Senin Metrik Sınıfın

    history_data = {
        'Cycle': [], 'Anomaly Score': [], 'Threshold': []
    }

    for data_packet in streamer.stream():

        # 2. Guard Kontrolü (Fail-Safe)
        if not guard.validate(data_packet):
            continue  # Hatalı veriyi atla

        # 3. Orchestrator Analizi
        decision = orchestrator.diagnose(data_packet)
        current_cycle = data_packet['cycle']
        loss = decision['loss']
        threshold = decision['threshold']
        priority = decision['priority']

        # 4. Metrik Takibi (Ground Truth Simülasyonu)
        # NASA setinde genelde 130. döngüden sonra bozulma başlar.
        # Bu yüzden 130 sonrasını "Gerçek Hata" (1), öncesini "Normal" (0) kabul ediyoruz.
        simulated_ground_truth = 1 if current_cycle > 90 else 0
        predicted_class = 1 if priority >= 2 else 0  # Warning veya Critical ise Hata(1)

        evaluator.add_record(simulated_ground_truth, predicted_class, probability=loss)

        # Sidebar İstatistiklerini Güncelle
        # Her döngüde generate_report çağırmak yerine basit hesap yapıyoruz
        metric_cycle.text(f"Cycle: {int(current_cycle)}")
        metric_accuracy.text(f"Anomalies Found: {sum(evaluator.y_pred)}")

        # 5. Grafik Verisi Güncelleme
        history_data['Cycle'].append(current_cycle)
        history_data['Anomaly Score'].append(loss)
        history_data['Threshold'].append(threshold)

        # Son 60 veriyi tut (Kayar Pencere)
        df_chart = pd.DataFrame(history_data).tail(60)

        # 6. Görselleştirme (Visualizer Kullanımı)
        chart = DashboardVisualizer.create_anomaly_chart(df_chart)
        if chart:
            chart_placeholder.altair_chart(chart, use_container_width=True)

        # 7. Üst Panel Güncelleme
        cycle_disp.metric("Döngü", int(current_cycle))
        loss_disp.metric("Hata Skoru", f"{loss:.4f}")

        status_text = decision['status']
        if priority == 1:
            status_html = f"<h3 class='status-normal'>🟢 {status_text}</h3>"
            ai_disp.info("AI: Hazır")
        elif priority == 2:
            status_html = f"<h3 class='status-warning'>⚠️ {status_text}</h3>"
            ai_disp.warning("AI: İzliyor")
        else:  # Priority 4
            status_html = f"<h3 class='status-critical'>🚨 {status_text}</h3>"
            ai_disp.error("AI: MÜDAHALE!")

        status_disp.markdown(status_html, unsafe_allow_html=True)

        # 8. KRİTİK HATA VE AI TETİKLEME
        if priority == 4:
            logger.critical(f"Kritik Hata! Cycle: {current_cycle}, Loss: {loss:.4f}")

            with ai_log:
                st.error(f"🔴 KRİTİK EŞİK AŞILDI! [Cycle {current_cycle}]")
                st.write(f"Limit: {threshold * 1.25:.4f} | Mevcut: {loss:.4f}")
                st.markdown("---")
                st.warning("⚠️ CrewAI Ajanları Göreve Çağrılıyor...")

                with st.spinner('Analiz yapılıyor (Groq Llama-3)...'):
                    try:
                        ai_crew = JetEngineCrew()
                        # Veriyi stringe çevirip gönder
                        report = ai_crew.run_mission(str(data_packet), f"{loss:.4f}")

                        st.success("✅ Analiz Tamamlandı!")
                        st.markdown(report)

                        # Raporu Loga da yaz
                        logger.info("AI Raporu oluşturuldu.")

                        # Son Metrik Raporunu Bas (Terminalde görebilirsin)
                        recall, acc, f1, auc = evaluator.generate_report()
                        st.info(f"📊 Session Metrics -> Recall: {recall:.2f} | F1: {f1:.2f}")

                    except Exception as e:
                        st.error(f"AI Hatası: {e}")
                        logger.error(f"AI Hatası: {e}")

            time.sleep(10)  # Rapor okunsun diye bekle
            st.stop()

        time.sleep(speed)