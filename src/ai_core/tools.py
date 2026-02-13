from crewai.tools import tool
from config.paths import MANUAL_DIR

class AnalysisTools:

    @tool("Calculate Rate of Change")
    def calculate_roc(current_val: float, previous_val: float, time_delta: float):
        """
        Calculates the Rate of Change (RoC) or slope between two sensor readings.
        Formula: (Current - Previous) / TimeDelta.
        Useful for detecting rapid temperature spikes or pressure drops.
        """
        try:
            current_val = float(current_val)
            previous_val = float(previous_val)
            time_delta = float(time_delta)

            if time_delta == 0:
                return "Error: Time delta cannot be zero."

            roc = (current_val - previous_val) / time_delta
            return f"Calculated RoC: {roc:.4f} units/cycle"
        except Exception as e:
            return f"Calculation Error: {str(e)}"

    @tool("Fetch Sensor Limits")
    def fetch_sensor_limits(sensor_name: str):
        """
        Retrieves the nominal operating range (Min-Max) for a specific jet engine sensor.
        Useful for verifying if a sensor is out of bounds.
        """
        # Bu veriler normalde bir veritabanından gelir. Simüle ediyoruz.
        limits_db = {
            "sensor_measurement2": {"name": "T24 (LPC Outlet Temp)", "min": 640, "max": 645},
            "sensor_measurement3": {"name": "T30 (HPC Outlet Temp)", "min": 1570, "max": 1600},
            "sensor_measurement4": {"name": "T50 (LPT Outlet Temp)", "min": 1390, "max": 1420},
            "sensor_measurement11": {"name": "P30 (Static Pressure)", "min": 46, "max": 48},
        }

        # Gelen string'i temizle (örn: 'Sensor 11' -> 'sensor_measurement11')
        # Ajan bazen "Sensor 11" der, bazen "sensor_measurement11". Basit bir mapping:
        key = sensor_name.lower().strip().replace(" ", "_")
        if "sensor_" not in key and key.isdigit():  # Sadece "11" geldiyse
            key = f"sensor_measurement{key}"
        elif "sensor" in key and "measurement" not in key:  # "sensor11" geldiyse
            key = key.replace("sensor", "sensor_measurement")

        info = limits_db.get(key)
        if info:
            return f"LIMITS for {key} ({info['name']}): Min={info['min']}, Max={info['max']}"
        else:
            return f"No limit data found for {sensor_name}. Assume standard deviation check."

    @tool("Consult Technical Manual")
    def consult_manual(search_query: str):
        """
        Searches the 'engine_manual.txt' for specific failure symptoms or components
        (e.g., 'Compressor', 'Vibration', 'Sensor 11').
        Returns the relevant section from the maintenance manual.
        """
        manual_path = MANUAL_DIR

        try:
            with open(manual_path, 'r') as f:
                content = f.read()

            # Basit Arama Mantığı
            results = []
            paragraphs = content.split('\n\n')  # Paragraflara böl

            for p in paragraphs:
                if search_query.lower() in p.lower():
                    results.append(p)

            if not results:
                return "No specific manual entry found for this query. Use standard protocol."

            return "\n---\n".join(results)

        except Exception as e:
            return f"Error reading manual: {str(e)}"