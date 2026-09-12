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
    goal="Take an expanded news update and shrink it down into a short description under 300 characters total.",
    backstory="A strict editor who deletes long background paragraphs to make text perfect for Instagram limits.",
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
        "2. A short, vivid description of a single cinematic image summarizing this news story (no text/words/letters in the image)."
    ),
    expected_output="An Instagram caption and a visual summary description written in plain text.",
    agent=designer
)

publish_task = Task(
    description=(
        "Look at the text provided by the previous agents. Extract ONLY the Instagram caption part.\n"
        "Rewrite it to be extremely short, clean, and punchy.\n"
        "CRITICAL RULE: The final output must be pure copy text under 300 characters total. "
        "Do not write anything else. No technical notes, no explanations, no labels."
    ),
    expected_output="A single short copy caption under 300 characters with no labels or extra text.",
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

# Force extraction from the raw final task block string
raw_designer_text = str(content_crew.tasks[-1].output.raw).strip()

# HARDCODE FIX: Forcefully slice the text array so it is physically impossible to exceed Meta's limit
clean_short_caption = raw_designer_text[:400]

# 4. Native Interactions API Image Call (Nano Banana)
print("\n🎨 Initializing native Interactions API to generate the post graphic...\n")
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    interaction = client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=f"Create a high-quality, eye-catching, cinematic, 1:1 aspect ratio square social media graphic depicting: {clean_short_caption}. Photorealistic, vibrant color layout, ultra-detailed, strictly no text words or letters."
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
    file.write(clean_short_caption)

# 5. Route Payload out to Automation Bridge (Make/n8n)
webhook_url = os.getenv("WEBHOOK_URL")
if webhook_url:
    print("\n📦 Pushing data to your automated Instagram posting bridge...")
    payload = {"caption": clean_short_caption, "image_status": "Ready in repository"}
    try:
        response = requests.post(webhook_url, json=payload)
        print(f"📡 Bridge Response Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not reach posting bridge: {e}")
else:
    print("\nℹ️ Webhook URL not set. Text and images saved locally in folder.")
