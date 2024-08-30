from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import time
import random
import re

# Initialize Qdrant client
qdrant_client = QdrantClient(":memory:")  # Replace with your Qdrant server URL or local setup

# Initialize sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize GPT-Neo model
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Add a pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
pad_token_id = tokenizer.pad_token_id

# Sample emergency data with a specific entry for high fever
emergencies = {
    "not breathing": "Start CPR immediately. Push hard and fast in the center of the chest, and give rescue breaths if you are trained.",
    "chest pain": "Chew an aspirin if available and not allergic. Sit down and try to stay calm.",
    "severe bleeding": "Apply pressure to the wound with a clean cloth or bandage. Elevate the injured area if possible.",
    "unconscious": "Check for breathing and pulse. If absent, begin CPR immediately.",
    "burns": "Cool the burn under running water for at least 10 minutes. Cover with a clean cloth.",
    "poisoning": "Try to identify the poison and call the poison control center immediately.",
    "blood clot": "Keep the affected area elevated and apply a cold compress. Seek immediate medical attention.",
    "high fever": "Ensure the patient stays hydrated and rests. Use a damp cloth on the forehead and administer fever-reducing medication if available and appropriate. Seek medical attention if the fever persists or is very high."
}

# Prepare data for Qdrant
points = []
for idx, (emergency, instruction) in enumerate(emergencies.items()):
    embedding = model.encode(emergency).tolist()
    point = PointStruct(
        id=idx,
        vector=embedding,
        payload={"instruction": instruction, "emergency": emergency}
    )
    points.append(point)

# Create a collection in Qdrant and upload points
qdrant_client.recreate_collection(
    collection_name="emergencies",
    vectors_config=VectorParams(size=len(points[0].vector), distance="Cosine")
)

qdrant_client.upsert(collection_name="emergencies", points=points)

# Function to handle emergencies using Qdrant and fallback to GPT-Neo if necessary
def handle_emergency(user_input):
    embedding = model.encode(user_input).tolist()
    result = qdrant_client.search(
        collection_name="emergencies",
        query_vector=embedding,
        limit=1
    )

    if result and result[0].score > 0.8:  # Adjust threshold based on requirements
        instruction = result[0].payload['instruction']
    else:
        instruction = call_gpt_neo_for_solution(user_input)

    return instruction

# Function to call GPT-Neo for a solution when Qdrant doesn't have an appropriate response
def call_gpt_neo_for_solution(user_input):
    prompt = f"A patient is experiencing an emergency described as: '{user_input}'. What are the immediate steps they should take until the doctor arrives?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)  # Get attention mask if available

    outputs = gpt_neo_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_gpt_neo_response(response)

# Function to clean the GPT-Neo response
def clean_gpt_neo_response(response_text):
    # Remove repetitive instructions
    response_text = re.sub(r"(Call the (doctor|police|ambulance|hospital)\.){2,}", "Please call the doctor, police, ambulance, or hospital if needed.", response_text)
    # Check if the response contains generic or irrelevant instructions
    if "take a shower" in response_text.lower():
        return "The response provided does not seem appropriate for your situation. Please follow general emergency guidelines while waiting for the doctor."
    return response_text

def simulate_delay_and_search(user_input, delay=15):
    print("I am checking what you should do immediately, meanwhile, can you tell me which area you are located in right now?")
    time.sleep(delay)
    return handle_emergency(user_input)

def chat_bot():
    print("Hello! Are you experiencing an emergency or would you like to leave a message for Dr. Adrin?")
    user_input = input()

    if "emergency" in user_input.lower():
        print("Please describe the emergency.")
        emergency_input = input()

        # Simulate delay and search for emergency response
        response = simulate_delay_and_search(emergency_input)

        print("Where are you located?")
        location = input()

        # Provide a random estimated time of arrival
        eta = random.randint(5, 20)
        print(f"Dr. Adrin will be coming to your location in approximately {eta} minutes.")

        if eta > 10:
            print("I understand that you are worried that Dr. Adrin will arrive too late.")
            print(response)
        else:
            print(response)
            print("Don't worry, please follow these steps, Dr. Adrin will be with you shortly.")
    elif "message" in user_input.lower():
        print("Please provide your message.")
        message_input = input()
        print("Thanks for the message, we will forward it to Dr. Adrin.")
    else:
        print("I don't understand that. Could you please specify if it's an emergency or if you'd like to leave a message for Dr. Adrin?")
        chat_bot()

# Run the chatbot
chat_bot()
