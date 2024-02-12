from crewai import Agent, Task, Crew, Process
from crewai.models import GPT
from crewai.languages import Language

# Create CrewAI GPT instances for the different agents
researcher_gpt = GPT(engine="openhermes", language=Language.ENGLISH)
writer_gpt = GPT(engine="openhermes", language=Language.ENGLISH)
examiner_gpt = GPT(engine="openhermes", language=Language.ENGLISH)

# Researcher Agent
researcher = Agent(
    role='Researcher',
    goal='Develop ideas for teaching someone new to the subject',
    backstory='A knowledgeable expert in the subject',
    verbose=True,
    llm=researcher_gpt
)

# Writer Agent
writer = Agent(
    role='Writer',
    goal='Write a piece of text to explain the topic',
    backstory='An experienced writer skilled at simplifying complex topics',
    verbose=True,
    llm=writer_gpt
)

# Examiner Agent
examiner = Agent(
    role='Examiner',
    goal='Craft test questions to evaluate understanding',
    backstory='An expert in creating challenging questions',
    verbose=True,
    llm=examiner_gpt
)

# Assign tasks
research_task = Task(
    description='Generate ideas for teaching the subject',
    agent=researcher
)

write_task = Task(
    description='Write a piece of text explaining the topic',
    agent=writer
)

examine_task = Task(
    description='Craft test questions to evaluate understanding',
    agent=examiner
)

# Instantiate crew
topic_crew = Crew(
    agents=[researcher, writer, examiner],
    tasks=[research_task, write_task, examine_task],
    process=Process.SEQUENTIAL
)

# Begin the task execution
result = topic_crew.kickoff()
print(result)