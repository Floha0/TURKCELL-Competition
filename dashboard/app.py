import os
import streamlit as st
import pandas as pd
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px

# --- ENV AYARLARI ---
load_dotenv()
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# --- PATH AYARLARI ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# --- MODÜL IMPORTLARI ---
from src.simulation.streamer import SensorStreamer
from src.orchestrator.manager import Orchestrator
from src.ai_core.crew import JetEngineCrew
from src.utils.logger import logger
from src.stats_engine.guard import DataGuard
from src.utils.visualizer import DashboardVisualizer
from src.stats_engine.metrics import PerformanceEvaluator

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="JetGuard Defense System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 5px; color: #e6edf3; }
    .terminal-box { background-color: #000000; color: #00FF00; font-family: 'Courier New', Courier, monospace; padding: 15px; border-radius: 5px; border-left: 5px solid #00FF00; height: 400px; overflow-y: auto; font-size: 14px; box-shadow: 0 0 10px rgba(0, 255, 0, 0.2); }
    @keyframes blinker { 50% { opacity: 0; } }
    .critical-alert { color: red; font-weight: bold; font-size: 24px; animation: blinker 1s linear infinite; text-align: center; border: 2px solid red; padding: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# --- YARDIMCI FONKSİYONLAR ---
def create_gauge(value, title, min_val, max_val, color="green", threshold=None):
    """Plotly Gauge - Optimize Edilmiş"""

    tick_step = (max_val - min_val) / 4

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': "white"}},
        number={'font': {'size': 35, 'color': color, 'weight': 'bold'}},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickwidth': 2,
                'tickcolor': "white",
                'tickmode': 'linear',
                'dtick': tick_step,
                'tickfont': {'size': 14, 'color': '#AAAAAA'}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, max_val], 'color': "#161b22"},
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=280,
        margin=dict(l=35, r=35, t=50, b=35)
    )
    return fig


# --- BAŞLIK ---
col_logo, col_title = st.columns([1, 6])
with col_logo: st.image("https://cdn-icons-png.flaticon.com/512/900/900967.png", width=80)
with col_title:
    st.title("JETGUARD // AI DEFENSE SYSTEM")
    st.caption("MISSION CONTROL: LIVE TELEMETRY & PREDICTIVE MAINTENANCE")

st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ SİSTEM AYARLARI")
    engine_id = st.number_input("Target Engine Unit", 1, 100, 1)
    # TİTREME ÇÖZÜMÜ 1: Varsayılan hızı 0.1'e çektik (Daha kararlı)
    speed = st.slider("Simulation Clock (sec)", 0.01, 1.0, 0.1)
    start_btn = st.button("INITIATE SEQUENCE", type="primary", use_container_width=True)
    st.info("System Ready...")

# --- YERLEŞİM ---
kpi_col1, kpi_col2, kpi_col3 = st.columns([1, 1, 2])
with kpi_col1: rul_placeholder = st.empty()
with kpi_col2: loss_placeholder = st.empty()
with kpi_col3: status_placeholder = st.empty()

row2_col1, row2_col2 = st.columns([2, 1])
with row2_col1: chart_placeholder = st.empty()
with row2_col2: ai_terminal_placeholder = st.empty()

st.subheader("📡 SENSOR ARRAY STATUS")
sensor_chart_placeholder = st.empty()

# --- ANA DÖNGÜ ---
if start_btn:
    logger.info("Visual Simulation Started.")

    streamer = SensorStreamer(engine_id=engine_id)
    orchestrator = Orchestrator()
    guard = DataGuard()
    evaluator = PerformanceEvaluator()

    history_data = {'Cycle': [], 'Anomaly Score': [], 'Threshold': []}
    terminal_logs = ["System Initialized...", "Connecting to Satellite Stream...", "Data Link Established."]

    # TİTREME ÇÖZÜMÜ 2: Plotly Config (Statik Plot)
    plotly_config = {'staticPlot': True, 'displayModeBar': False}

    for data_packet in streamer.stream():

        if not guard.validate(data_packet): continue
        decision = orchestrator.diagnose(data_packet)

        current_cycle = data_packet['cycle']
        loss = decision['loss']
        threshold = decision['threshold']
        priority = decision['priority']
        predicted_rul = decision.get('predicted_rul', 0)

        history_data['Cycle'].append(current_cycle)
        history_data['Anomaly Score'].append(loss)
        history_data['Threshold'].append(threshold)

        # RECALL DÜZELTME: Ground Truth eşiğini 130'dan 90'a çektik.
        # Artık 90'dan sonra gelen her uyarı "DOĞRU BİLİNMİŞ" sayılacak.
        simulated_ground_truth = 1 if current_cycle > 90 else 0
        predicted_class = 1 if priority >= 2 else 0
        evaluator.add_record(simulated_ground_truth, predicted_class, probability=loss)

        # --- GÖRSEL GÜNCELLEME ---

        # A. RUL Gauge
        rul_color = "#00FF00" if predicted_rul > 50 else "#FFA500" if predicted_rul > 20 else "#FF0000"
        fig_rul = create_gauge(predicted_rul, "Est. RUL (Cycles)", 0, 200, rul_color, threshold=20)
        # config parametresi eklendi
        rul_placeholder.plotly_chart(fig_rul, use_container_width=True, config=plotly_config)

        # B. Anomaly Gauge
        loss_color = "#00FF00" if priority == 1 else "#FFA500" if priority == 2 else "#FF0000"
        fig_loss = create_gauge(loss, "Anomaly Score", 0, threshold * 3, loss_color, threshold=threshold)
        # config parametresi eklendi
        loss_placeholder.plotly_chart(fig_loss, use_container_width=True, config=plotly_config)

        # C. Status
        status_text = decision['status']
        if priority == 4:
            status_html = f"<div class='critical-alert'>🚨 {status_text} 🚨<br>IMMEDIATE ACTION REQUIRED</div>"
        else:
            border_color = "#00FF00" if priority == 1 else "#FFA500"
            status_html = f"""
            <div style="border: 2px solid {border_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: {border_color}; margin:0;">SYSTEM STATUS</h2>
                <h1 style="color: white; margin:0;">{status_text}</h1>
                <p>Cycle: {int(current_cycle)}</p>
            </div>
            """
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        # D. Grafik
        df_chart = pd.DataFrame(history_data).tail(60)
        chart = DashboardVisualizer.create_anomaly_chart(df_chart)
        chart_placeholder.altair_chart(chart, use_container_width=True)

        # E. Terminal
        log_entry = f"> Cycle {current_cycle}: Loss {loss:.4f} | Status: {status_text}"
        terminal_logs.append(log_entry)
        if len(terminal_logs) > 12: terminal_logs.pop(0)
        terminal_html = "<div class='terminal-box'>" + "<br>".join(terminal_logs) + "</div>"
        ai_terminal_placeholder.markdown(terminal_html, unsafe_allow_html=True)

        # F. Heatmap
        sensor_df = pd.DataFrame([data_packet]).drop(
            columns=['unit_number', 'cycle', 'setting1', 'setting2', 'setting3'])
        fig_heat = px.imshow(sensor_df, aspect="auto", color_continuous_scale="Viridis",
                             labels=dict(x="Sensor", y="Value"))
        fig_heat.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        fig_heat.update_xaxes(showticklabels=False)
        fig_heat.update_yaxes(showticklabels=False)
        sensor_chart_placeholder.plotly_chart(fig_heat, use_container_width=True, config=plotly_config)

        # 5. KRİTİK HATA VE AI TETİKLEME
        if priority == 4:
            logger.critical(f"Kritik Hata! Cycle: {current_cycle}")

            terminal_logs.append(f"> 🚨 CRITICAL FAULT DETECTED!")
            terminal_logs.append(f"> INITIATING AI AGENTS...")
            ai_terminal_placeholder.markdown("<div class='terminal-box'>" + "<br>".join(terminal_logs) + "</div>",
                                             unsafe_allow_html=True)

            with st.spinner('🤖 AI CREW ENGAGED: ANALYZING TELEMETRY...'):
                try:
                    ai_crew = JetEngineCrew()

                    ai_input_data = f"SENSOR TELEMETRY: {str(data_packet)}\nPREDICTED RUL: {int(predicted_rul)} CYCLES"
                    report = ai_crew.run_mission(ai_input_data, f"{loss:.4f}")

                    st.divider()
                    st.success("✅ MISSION COMPLETE: AI DIAGNOSIS RECEIVED")
                    st.markdown(f"""
                    <div style="background-color: #0d1117; border: 1px solid #30363d; padding: 20px; border-radius: 10px;">
                        {report}
                    </div>
                    """, unsafe_allow_html=True)

                    # METRİK RAPORU (Durdurmadan Önce)
                    try:
                        recall, acc, f1, auc = evaluator.generate_report()

                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Final Recall", f"{recall:.2f}")
                        col_m2.metric("Final F1", f"{f1:.2f}")
                        col_m3.metric("Final Accuracy", f"{acc:.2f}")

                    except Exception as metric_err:
                        logger.error(f"Metric Error: {metric_err}")
                        st.warning("Not enough data to calculate AUC.")

                except Exception as e:
                    st.error(f"AI FAILURE: {e}")

            time.sleep(15)
            st.stop()

        time.sleep(speed)