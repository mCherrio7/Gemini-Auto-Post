import os
import requests
import base64
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool
from google import genai

# 1. Setup tools and connect to the upgraded Gemini 3.6 engine
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
        "Generate exactly two elements labeled precisely like this:\n"
        "=== CAPTION START ===\n"
        "An intense hook line, a short summary, 3 bullet points, and 4 specific hashtags. Keep this section under 600 characters total.\n"
        "=== CAPTION END ===\n"
        "=== PROMPT START ===\n"
        "A descriptive, detailed prompt for an AI image generator summarizing this news story (no words in the image).\n"
        "=== PROMPT END ==="
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
crew_output = content_crew.kickoff()

# CRITICAL FIX: Extract ONLY the text from the Designer Agent's task output (removes the noise)
raw_designer_text = content_crew.tasks[1].output.raw

# Parse out the clean caption between the structural tags
final_instagram_caption = "AI Update Today! 🔥"
if "=== CAPTION START ===" in raw_designer_text:
    try:
        final_instagram_caption = raw_designer_text.split("=== CAPTION START ===")[1].split("=== CAPTION END ===")[0].strip()
    except Exception:
        final_instagram_caption = raw_designer_text[:800] # Fallback safety cutoff
else:
    final_instagram_caption = raw_designer_text[:800]

# 4. Native Interactions API Image Call (Nano Banana)
print("\n🎨 Initializing native Interactions API to generate the post graphic...\n")
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    interaction = client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=f"Create a high-quality, eye-catching, cinematic, 1:1 aspect ratio square social media graphic depicting: {raw_designer_text}. Photorealistic, vibrant color layout, ultra-detailed, strictly no text words or letters."
    )
    
    image_saved = False
    if interaction.candidates and interaction.candidates.content.parts:
        for part in interaction.candidates.content.parts:
            if part.inline_data and "image" in part.inline_data.mime_type:
                image_bytes = base64.b64decode(part.inline_data.data)
                with open("post_image.jpg", "wb") as f:
                    f.write(image_bytes)
                print("✅ Success! Image generated natively and saved as 'post_image.jpg'!")
                image_saved = True
                break
                
    if not image_saved:
        print("⚠️ Model did not embed raw image bytes into the content block. Moving forward with text only.")
        
except Exception as e:
    print(f"⚠️ Rebuilt image generation block encountered a system issue: {e}")

# Save text caption output to folder directory
with open("social_media_posts.txt", "w", encoding="utf-8") as file:
    file.write(final_instagram_caption)

# 5. Route Payload out to Automation Bridge (Make/n8n)
webhook_url = os.getenv("WEBHOOK_URL")
if webhook_url:
    print("\n📦 Pushing data to your automated Instagram posting bridge...")
    payload = {"caption": final_instagram_caption, "image_status": "Ready in repository"}
    try:
        response = requests.post(webhook_url, json=payload)
        print(f"📡 Bridge Response Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not reach posting bridge: {e}")
else:
    print("\nℹ️ Webhook URL not set. Text and images saved locally in folder.")
