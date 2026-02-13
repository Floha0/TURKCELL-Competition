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

import os
from crewai import Agent, Task, Crew, Process


class JetEngineCrew:
    def __init__(self):
        # API Anahtarını kontrol et
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ UYARI: GROQ_API_KEY bulunamadı! Terminalden export ettiğine emin ol.")

        # MODEL ADI (String olarak)
        # CrewAI, "groq/" ön ekini görünce otomatik olarak Groq API'sini kullanır.
        self.model_name = "groq/llama-3.3-70b-versatile"

    def run_mission(self, sensor_data, loss_score):
        # --- 1. AJAN: TEŞHİS UZMANI ---
        diagnostician = Agent(
            role='Kıdemli Jet Motoru Teknisyeni',
            goal='Sensör verilerindeki anomaliyi analiz et ve kök nedeni bul.',
            backstory="""Sen NASA ve GE'de çalışmış, 20 yıllık tecrübesi olan bir motor uzmanısın. 
            Sayısal verileri okuyup motorun hangi parçasının (Fan, Kompresör, Türbin) arızalandığını 
            anında anlarsın.""",
            verbose=True,
            allow_delegation=False,
            llm=self.model_name  # String olarak veriyoruz
        )

        # --- 2. AJAN: KRİZ YÖNETİCİSİ ---
        commander = Agent(
            role='Acil Durum Müdahale Lideri',
            goal='Teşhis raporuna göre en güvenli aksiyon planını oluştur.',
            backstory="""Sen endüstriyel felaketleri önlemekle görevli bir güvenlik liderisin. 
            Risk almazsın. Önceliğin insan hayatı ve ekipman güvenliğidir. 
            Net, kısa ve emir kipiyle konuşursun.""",
            verbose=True,
            allow_delegation=False,
            llm=self.model_name  # String olarak veriyoruz
        )

        # --- GÖREVLER ---
        analysis_task = Task(
            description=f"""
            Şu an bir Jet Motorunda KRİTİK HATA (Priority 4) tespit edildi.
            İstatistiksel Hata Skoru: {loss_score} (Normalin çok üzerinde!)

            Gelen son sensör verisi şudur:
            {sensor_data}

            GÖREVİN:
            1. Verilen sensör değerlerini incele.
            2. Hangi sensörlerin limit dışı olduğunu tespit et.
            3. Bu durumun fiziksel nedenini (Örn: Kompresör stall, yüksek sıcaklık) açıkla.
            """,
            expected_output="Maddeler halinde teknik arıza analizi raporu.",
            agent=diagnostician
        )

        action_task = Task(
            description="""
            Teşhis raporunu oku. Operasyon ekibine 3 maddelik ACİL MÜDAHALE PLANI hazırla.
            Emir kipi kullan (Örn: 'Motoru kapat', 'Vanayı aç').
            """,
            expected_output="3 maddelik acil eylem listesi.",
            agent=commander
        )

        # --- EKİBİ KUR ---
        crew = Crew(
            agents=[diagnostician, commander],
            tasks=[analysis_task, action_task],
            process=Process.sequential,
            verbose=True,
            memory=False,  # <--- İŞTE ÇÖZÜM: Hafızayı kapatıyoruz ki OpenAI aramasın.
        )

        result = crew.kickoff()
        return result