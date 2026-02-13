"""
Agentlar:

    Sensor Analyst Agent: "Sensör 11 ve 12 artarken Sensör 7 düşmüş, bu kompresör arızasına işaret ediyor olabilir." yorumunu yapar.

    Maintenance Planner Agent: "Bu motorun acil bakıma girmesi lazım, şu anki uçuş döngüsü tamamlanınca hangara çekin." aksiyonunu önerir.

Kullanım: LangChain üzerinden bu ajanlar birbirine data paslar ve final raporu oluşturur.
"""
import sys

"""
Agentlar:

    Sensor Analyst Agent: "Sensör 11 ve 12 artarken Sensör 7 düşmüş, bu kompresör arızasına işaret ediyor olabilir." yorumunu yapar.

    Maintenance Planner Agent: "Bu motorun acil bakıma girmesi lazım, şu anki uçuş döngüsü tamamlanınca hangara çekin." aksiyonunu önerir.

Kullanım: LangChain üzerinden bu ajanlar birbirine data paslar ve final raporu oluşturur.
"""
import dotenv

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

dotenv.load_dotenv(ROOT_DIR / '.env.local')

from src.ai_core.tools import AnalysisTools
from config.paths import CONFIG_DIR, AGENTS_FILE, TASKS_FILE
import os
import yaml
from crewai import Agent, Task, Crew, Process


class JetEngineCrew:
    def __init__(self):
        # 1. API Anahtarı Kontrolü
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ WARNING: GROQ_API_KEY not found! Please export it in your terminal.")

        # Model Tanımı (LiteLLM Formatı)
        self.model_name = "groq/llama-3.3-70b-versatile"

        # 2. Konfigürasyon Dosyalarını Yükle

        self.configs_path = CONFIG_DIR
        self.agents_config = self._load_yaml(AGENTS_FILE)
        self.tasks_config = self._load_yaml(TASKS_FILE)

    def _load_yaml(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def run_mission(self, sensor_data, loss_score):
        """
        Orchestrates the AI crew to analyze the failure.
        """

        # --- AGENTLER (YAML'dan gelen verilerle) ---
        sensor_analyst = Agent(
            role=self.agents_config['sensor_analyst']['role'],
            goal=self.agents_config['sensor_analyst']['goal'],
            backstory=self.agents_config['sensor_analyst']['backstory'],
            verbose=True,
            allow_delegation=False,
            llm=self.model_name,
            tools=[
                AnalysisTools.calculate_roc,
                AnalysisTools.fetch_sensor_limits
            ]
        )

        maintenance_commander = Agent(
            role=self.agents_config['maintenance_commander']['role'],
            goal=self.agents_config['maintenance_commander']['goal'],
            backstory=self.agents_config['maintenance_commander']['backstory'],
            verbose=True,
            allow_delegation=False,
            llm=self.model_name
        )

        # --- GÖREVLER (YAML'dan gelen verilerle & Formatlayarak) ---

        # YAML içindeki {variable} kısımlarını gerçek verilerle dolduruyoruz
        formatted_diagnosis_desc = self.tasks_config['diagnosis_task']['description'].format(
            loss_score=loss_score,
            sensor_data=sensor_data
        )

        diagnosis_task = Task(
            description=formatted_diagnosis_desc,
            expected_output=self.tasks_config['diagnosis_task']['expected_output'],
            agent=sensor_analyst
        )

        action_plan_task = Task(
            description=self.tasks_config['action_plan_task']['description'],
            expected_output=self.tasks_config['action_plan_task']['expected_output'],
            agent=maintenance_commander
        )

        # --- EKİBİ KUR ---
        crew = Crew(
            agents=[sensor_analyst, maintenance_commander],
            tasks=[diagnosis_task, action_plan_task],
            process=Process.sequential,
            verbose=True,
            memory=False,  # OpenAI hatasını engeller
        )

        return crew.kickoff()