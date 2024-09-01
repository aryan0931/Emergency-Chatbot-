import asyncio
import random
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

# Initialize Qdrant client and models
qdrant_client = QdrantClient(":memory:")  # Use in-memory DB for this demo
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # For vector embeddings
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Add a pad token for GPT-Neo
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
pad_token_id = tokenizer.pad_token_id

# Function to search Qdrant for the closest matching emergency
async def search_emergency(user_input):
    await asyncio.sleep(15)  # Simulated 15-second delay
    embedding = embedding_model.encode(user_input).tolist()
    result = qdrant_client.search(collection_name="emergencies", query_vector=embedding, limit=1)
    
    if result and result[0].score > 0.8:
        return result[0].payload['instruction']
    else:
        return None

# GPT-Neo to generate emergency instructions based on the user's description
async def generate_emergency_response(user_input):
    prompt = f"A patient is experiencing an emergency described as: '{user_input}'. What are the immediate steps they should take until the doctor arrives?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = gpt_neo_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask', None),
        max_length=150,
        num_return_sequences=1,
        pad_token_id=pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_gpt_neo_response(response)

# Clean GPT-Neo response
def clean_gpt_neo_response(response_text):
    response_text = re.sub(r"(Call the (doctor|ambulance)\.){2,}", "Call the doctor.", response_text)
    return response_text.strip()

# Store the generated response in Qdrant for future use
async def store_emergency_response(user_input, response):
    embedding = embedding_model.encode(user_input).tolist()
    point = PointStruct(id=random.randint(1, 1_000_000), vector=embedding, payload={"instruction": response, "emergency": user_input})
    qdrant_client.upsert(collection_name="emergencies", points=[point])

# Simulate interaction with the user
async def interact_with_user():
    user_input = input("Is this an emergency or would you like to leave a message for Dr. Adrin? ").strip().lower()
    
    if "emergency" in user_input:
        user_input = input("Please describe the emergency: ").strip()
        print("I am checking what you should do immediately. Meanwhile, can you tell me which area you are located right now?")
        location = input("Enter your location: ").strip()

        # Artificial delay while searching the vector database
        existing_response = await search_emergency(user_input)
        
        if existing_response:
            response = existing_response
        else:
            response = await generate_emergency_response(user_input)
            await store_emergency_response(user_input, response)
        
        print(f"Thank you for providing your location: {location}. Dr. Adrin will be coming to your location immediately. Estimated time of arrival: {random.randint(10, 30)} minutes.")
        
        # If the user is concerned about the time
        arrival_concern = input("Are you concerned that Dr. Adrin will arrive too late? (yes/no): ").strip().lower()
        if "yes" in arrival_concern:
            print(f"I understand that you are worried that Dr. Adrin will arrive too late. Meanwhile, we would suggest the following steps: {response}")
        
        print("Don't worry, please follow these steps. Dr. Adrin will be with you shortly.")
    
    else:
        message = input("Please provide your message for Dr. Adrin: ").strip()
        # Here we would capture the message and acknowledge receipt
        print("Thanks for the message, we will forward it to Dr. Adrin.")

# Define the main function that will trigger the interaction
async def main():
    await interact_with_user()

# Initialize Qdrant before usage
async def setup():
    qdrant_client.recreate_collection(
        collection_name="emergencies",
        vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance="Cosine")
    )
    await main()

# Example usage within a running event loop
await setup()  # Use 'await' directly since 'asyncio.run()' isn't allowed in Jupyter
