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

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA" # Sahte key, CrewAI kontrolünü geçmek için
# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="JetEngine Guard AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STİL (Görsellik İçin) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .status-normal { color: #00FF00; font-weight: bold; }
    .status-warning { color: #FFA500; font-weight: bold; }
    .status-critical { color: #FF0000; font-weight: bold; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# --- BAŞLIK ---
st.title("✈️ JetEngine Guard: AI Tabanlı Anomali Tespiti")
st.markdown("**Decoupled Decision Making & Safety-Critical Monitoring**")

# --- SIDEBAR (Kontrol Paneli) ---
with st.sidebar:
    st.header("🎮 Simülasyon Kontrol")
    engine_id = st.number_input("Motor ID", min_value=1, max_value=100, value=1)
    # Hızı artırıp azaltabilirsin. 0.05 ideal bir demo hızıdır.
    speed = st.slider("Simülasyon Gecikmesi (sn)", 0.01, 1.0, 0.05)
    start_btn = st.button("🚀 Simülasyonu Başlat", type="primary")
    stop_btn = st.button("🛑 Durdur")

# --- ANA ARAYÜZ YERLEŞİMİ ---
# Üst Kısım: Anlık Durum Paneli
col1, col2, col3, col4 = st.columns(4)
with col1:
    cycle_metric = st.empty()
with col2:
    status_metric = st.empty()
with col3:
    loss_metric = st.empty()
with col4:
    ai_status = st.empty()

st.divider()

# Orta Kısım: Grafikler ve Loglar
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Canlı Sensör Anomali Grafiği")
    chart_placeholder = st.empty()

with col_right:
    st.subheader("🧠 AI Analiz Konsolu")
    ai_log_container = st.container(height=400)

# --- SİMÜLASYON MANTIĞI ---
if start_btn:
    # 1. Sistemleri Başlat
    streamer = SensorStreamer(engine_id=engine_id)
    orchestrator = Orchestrator()

    # Grafik İçin Veri Tamponu
    history_loss = []
    history_threshold = []
    history_cycles = []

    # AI Logu için placeholder text
    with ai_log_container:
        st.info("Sistem başlatıldı. Sensör verileri bekleniyor...")

    # 2. Döngüyü Başlat
    for data_packet in streamer.stream():
        # Stop butonuna basılırsa (Streamlit rerun yapar, burası kırılır)

        # --- ORCHESTRATOR ANALİZİ ---
        decision = orchestrator.diagnose(data_packet)
        current_cycle = data_packet['cycle']
        loss = decision['loss']
        threshold = decision['threshold']
        priority = decision['priority']

        # --- VERİ GÜNCELLEME ---
        history_cycles.append(current_cycle)
        history_loss.append(loss)
        history_threshold.append(threshold)

        # Veri seti çok şişmesin, son 60 veriyi tut (Kayar Pencere)
        if len(history_cycles) > 60:
            history_cycles.pop(0)
            history_loss.pop(0)
            history_threshold.pop(0)

        # --- METRİKLERİ GÜNCELLE ---
        cycle_metric.metric("Döngü (Cycle)", f"{int(current_cycle)}")
        loss_metric.metric("Hata Skoru (MSE)", f"{loss:.4f}")

        # Renk ve Durum Ayarı
        status_text = decision['status']
        if priority == 1:
            status_html = f"<h3 class='status-normal'>🟢 {status_text}</h3>"
            ai_status.info("Durum: Stabil")
        elif priority == 2:
            status_html = f"<h3 class='status-warning'>⚠️ {status_text}</h3>"
            ai_status.warning("Durum: İzleniyor")
        else:  # Priority 4
            status_html = f"<h3 class='status-critical'>🚨 {status_text}</h3>"
            ai_status.error("Durum: MÜDAHALE!")

        status_metric.markdown(status_html, unsafe_allow_html=True)

        # --- GRAFİK ÇİZİMİ (Altair) ---
        chart_data = pd.DataFrame({
            'Cycle': history_cycles,
            'Anomaly Score': history_loss,
            'Threshold': history_threshold
        })

        # Grafik katmanları
        base = alt.Chart(chart_data).encode(x=alt.X('Cycle', axis=alt.Axis(title='Zaman (Döngü)')))

        # Mavi çizgi: Anlık Hata
        line_loss = base.mark_line(color='#00FFFF', strokeWidth=3).encode(
            y=alt.Y('Anomaly Score', axis=alt.Axis(title='Hata Skoru')),
            tooltip=['Cycle', 'Anomaly Score']
        )

        # Kırmızı kesikli çizgi: Eşik Değeri
        line_thresh = base.mark_line(color='#FF4B4B', strokeDash=[5, 5]).encode(
            y='Threshold'
        )

        # Grafiği birleştir ve bas
        chart_placeholder.altair_chart(
            (line_loss + line_thresh).properties(height=350),
            use_container_width=True
        )

        # --- AI Tetikleme ve CrewAI Entegrasyonu ---
        if priority == 4:
            # 1. Önce görsel uyarıyı ver
            with ai_log_container:
                st.error(f"🔴 [Cycle {current_cycle}] KRİTİK EŞİK AŞILDI!")
                st.write(f"Hata Skoru: **{loss:.4f}** > Limit: **{threshold * 1.25:.4f}**")
                st.markdown("---")
                st.warning("⚠️ CrewAI Ajanları Göreve Çağrılıyor... Lütfen Bekleyin.")

                # İlerlemeyi göstermek için bir spinner
                with st.spinner('Analiz yapılıyor... (Diagnostician & Commander)'):
                    try:
                        # 2. CrewAI'ı Başlat
                        ai_crew = JetEngineCrew()

                        # Veriyi stringe çevirip gönderiyoruz
                        crew_result = ai_crew.run_mission(
                            sensor_data=str(data_packet),
                            loss_score=f"{loss:.4f}"
                        )

                        # 3. Sonucu Ekrana Bas
                        st.success("✅ Analiz Tamamlandı!")
                        st.markdown("### 📋 AI Müdahale Raporu")
                        st.markdown(crew_result)  # Markdown formatında rapor

                    except Exception as e:
                        st.error(f"AI Hatası: API Anahtarı eksik olabilir. Detay: {e}")

            # Demoda rapor okunsun diye biraz bekle ve durdur
            st.error("🛑 SİMÜLASYON SONLANDIRILDI.")
            break
        time.sleep(speed)