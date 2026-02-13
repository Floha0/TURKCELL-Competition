from crewai.tools import tool
from config.paths import MANUAL_DIR

class AnalysisTools:

    @tool("Calculate Rate of Change")
    def calculate_roc(current_val: float, previous_val: float, time_delta: float):
        """
        Calculates the Rate of Change (RoC) between two sensor readings.
        Useful for detecting rapid spikes in Temperature or Pressure.

        Args:
            current_val (float): The specific value at the current cycle.
            previous_val (float): The value at the previous cycle.
            time_delta (float): The time difference (usually 1.0 for single cycle).

        Returns:
            str: The calculated rate or an error message.
        """
        try:
            # Gelen veriler bazen string olabilir, garantiye alalım
            c_val = float(current_val)
            p_val = float(previous_val)
            t_delta = float(time_delta)

            if t_delta == 0:
                return "Error: Time delta cannot be zero."

            roc = (c_val - p_val) / t_delta
            return f"{roc:.4f}"
        except ValueError:
            return "Error: Invalid numeric input provided."
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