import asyncio
from supabase import create_client, Client
from realtime.connection import Socket
from dotenv import load_dotenv
load_dotenv()
import os


# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# Create a synchronous Supabase client (Async not natively supported)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

async def listen_for_changes():
    socket = Socket(f"{SUPABASE_URL}/realtime/v1", params={"apikey": SUPABASE_KEY})
    socket.connect()

    channel = socket.set_channel("realtime:public:criminal_faces")

    async def callback(payload):
        print("Change received!", payload)

    channel.on("postgres_changes", {
        "event": "INSERT",
        "schema": "public",
        "table": "criminal_faces"
    }, callback)

    channel.subscribe()
    print("Listening for changes...")

    while True:
        await asyncio.sleep(1)  # Keep the event loop running

# Run the async function
asyncio.run(listen_for_changes())
