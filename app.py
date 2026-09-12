import os
import requests
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool
from google import genai

# 1. Setup global configurations and connect to the upgraded Gemini 3.6 engine
search_tool = SerperDevTool()
gemini_model = LLM(
    model="gemini/gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# 2. Build the Autonomous Team of 3 Distinct Agents
researcher = Agent(
    role="Trend Spotter",
    goal="Identify the single most viral, shocking, or revolutionary AI development in the last 24 hours.",
    backstory="A digital scout obsessed with breaking news, viral tech trends, and massive industry shifts.",
    tools=[search_tool],
    llm=gemini_model,
    verbose=True
)

designer = Agent(
    role="Visual Concept Director",
    goal="Analyze news research data and outline a highly engaging visual art prompt for image generation.",
    backstory="A creative director who translates raw technical text into impactful imagery without text overlay.",
    llm=gemini_model,
    verbose=True
)

publisher = Agent(
    role="Social Media Publisher",
    goal="Package finalized media assets and coordinate secure broadcast to social media webhooks.",
    backstory="A digital distributor responsible for compiling captions and delivering payloads to cloud bridges.",
    llm=gemini_model,
    verbose=True
)

# 3. Define the Sequenced Tasks
research_task = Task(
    description="Scan the web for the single wildest or coolest development in Artificial Intelligence today. Focus on breaking news or mind-blowing tech releases.",
    expected_output="A clean breakdown of the news item with verified facts, figures, and source links.",
    agent=researcher
)

design_task = Task(
    description=(
        "Using the research provided, create high-quality social media content.\n"
        "Generate:\n"
        "1. An Instagram caption containing an intense hook, a short summary line, clean bullet points, and 4 specific hashtags.\n"
        "2. A descriptive, highly detailed prompt for an AI image generator that visually symbolizes this news story (no text/words in the image)."
    ),
    expected_output="An Instagram caption and an image generator prompt written in plain text.",
    agent=designer
)

publish_task = Task(
    description="Package the final structured captions and prepare the asset pipeline payload for the external posting webhook.",
    expected_output="A final structural status confirmation string detailing post readiness.",
    agent=publisher
)

# Assemble and launch your team
content_crew = Crew(
    agents=[researcher, designer, publisher],
    tasks=[research_task, design_task, publish_task],
    process=Process.sequential
)

print("\n🚀 3 Agents are launching! Running the pipeline autonomously...\n")
result = content_crew.kickoff()

output_text = str(result)

# 4. Corrected Image Generation Execution Block (Developer Mode Compatible)
print("\n🎨 Generating matching high-attention visual using Developer-Safe Image model...\n")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    # Uses the developer-accessible multimodal endpoint to safely generate raw visual content blocks
    image_result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            f"Generate a high-quality, eye-catching, cinematic, 1:1 aspect ratio square graphic depicting: {output_text}. "
            f"Photorealistic, vibrant lighting, ultra-detailed, strictly no text, words, letters, or numbers inside the image."
        ]
    )

    # Scans the response payload for inline binary image blocks and saves them locally
    image_saved = False
    for part in image_result.parts:
        if part.inline_data is not None:
            image = part.as_image()
            image.save("post_image.jpg")
            image_saved = True
            print("✅ Image generated successfully and saved as 'post_image.jpg'!")
            break
            
    if not image_saved:
        print("⚠️ Model did not return an image part. Moving forward with text only.")
        
except Exception as e:
    print(f"⚠️ Image generation failed: {e}. Moving forward with text only.")

# Save text caption output to folder
with open("social_media_posts.txt", "w", encoding="utf-8") as file:
    file.write(output_text)

# 5. Route Payload out to Automation Bridge (Make/n8n)
webhook_url = os.getenv("WEBHOOK_URL")
if webhook_url:
    print("\n📦 Pushing data to your automated Instagram posting bridge...")
    payload = {"caption": output_text, "image_status": "Ready in repository"}
    try:
        response = requests.post(webhook_url, json=payload)
        print(f"📡 Bridge Response Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not reach posting bridge: {e}")
else:
    print("\nℹ️ Webhook URL not set. Text and images saved locally in folder.")
