import sys
import os
import yaml
import dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

dotenv.load_dotenv(ROOT_DIR / '.env.local')
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key or groq_key == "":
    raise ValueError("🚨 GROQ_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")

os.environ["OPENAI_API_KEY"] = groq_key
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

from src.ai_core.tools import AnalysisTools
from config.paths import CONFIG_DIR, AGENTS_FILE, TASKS_FILE
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

class JetEngineCrew:
    def __init__(self):
        # 2. GERÇEK MOTOR (Adres değiştirildiği için hatasız bağlanacak)
        self.llm = ChatGroq(
            api_key=groq_key,
            model_name="llama-3.3-70b-versatile"
        )

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
        sensor_analyst = Agent(
            role=self.agents_config['sensor_analyst']['role'],
            goal=self.agents_config['sensor_analyst']['goal'],
            backstory=self.agents_config['sensor_analyst']['backstory'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            function_calling_llm=self.llm,
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
            llm=self.llm,
            function_calling_llm=self.llm,
            tools=[
                AnalysisTools.consult_manual,
            ]
        )

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

        crew = Crew(
            agents=[sensor_analyst, maintenance_commander],
            tasks=[diagnosis_task, action_plan_task],
            process=Process.sequential,
            verbose=True,
            memory=False
        )

        return crew.kickoff()