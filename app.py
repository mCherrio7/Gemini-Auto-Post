import os
import requests
import json
import base64
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool

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

# 4. Corrected Direct Prediction Request to Imagen 3 (Bypasses 404 block)
print("\n🎨 Sending optimized prediction request to Imagen 3 for visual construction...\n")
api_key = os.getenv("GEMINI_API_KEY")
imagen_url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict"
params = {"key": api_key}

headers = {"Content-Type": "application/json"}
# Google's updated structural payload wrapper layout format
payload = {
    "instances": [
        {
            "prompt": f"A high-quality, eye-catching, cinematic social media graphic. Photorealistic, vibrant lighting, ultra-detailed, square 1:1 aspect ratio, strictly no text, words, letters, or typos. Theme: {output_text}"
        }
    ],
    "parameters": {
        "sampleCount": 1,
        "aspectRatio": "1:1",
        "outputMimeType": "image/jpeg"
    }
}

try:
    response = requests.post(imagen_url, headers=headers, params=params, data=json.dumps(payload))
    if response.status_code == 200:
        response_data = response.json()
        # Decodes the updated predictions byte block structure
        base64_image_data = response_data["predictions"][0]["bytesBase64Encoded"]
        image_bytes = base64.b64decode(base64_image_data)
        
        with open("post_image.jpg", "wb") as f:
            f.write(image_bytes)
        print("✅ Image generated successfully via Imagen 3 API and saved as 'post_image.jpg'!")
    else:
        print(f"⚠️ Imagen API returned an error code: {response.status_code} - {response.text}")
except Exception as e:
    print(f"⚠️ Image request failed: {e}. Moving forward with text only.")

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
