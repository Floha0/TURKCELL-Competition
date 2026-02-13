import altair as alt
import pandas as pd


class DashboardVisualizer:
    @staticmethod
    def create_anomaly_chart(data: pd.DataFrame):
        """
        Generates the Altair Line Chart for the Dashboard.
        Expects DataFrame with columns: ['Cycle', 'Anomaly Score', 'Threshold']
        """
        if data.empty:
            return None

        # Base Chart
        base = alt.Chart(data).encode(
            x=alt.X('Cycle', axis=alt.Axis(title='Engine Cycles (Time)')),
            tooltip=['Cycle', 'Anomaly Score', 'Threshold']
        )

        # 1. The Anomaly Score Line (Blue)
        line_loss = base.mark_line(
            color='#00FFFF',
            strokeWidth=3,
            interpolate='monotone'  # Eğrileri yumuşatır
        ).encode(
            y=alt.Y('Anomaly Score', axis=alt.Axis(title='Reconstruction Error (MSE)'))
        )

        # 2. The Threshold Line (Red Dashed)
        line_thresh = base.mark_line(
            color='#FF4B4B',
            strokeDash=[5, 5],
            strokeWidth=2
        ).encode(
            y='Threshold'
        )

        # 3. Critical Area Highlight (Area under curve if needed, or simple layering)
        # Kırmızı bölgeyi boyamak için gradient eklenebilir ama şimdilik simple tutalım.

        return (line_loss + line_thresh).properties(
            height=350,
            title="Real-Time Sensor Deviation Monitor"
        ).interactive()