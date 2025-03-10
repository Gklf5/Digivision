import os
from supabase import create_client, Client
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

def on_change(payload):
    """Handle real-time changes from Supabase"""
    print("Change received:", payload)

async def subscribe_to_changes():
    """Subscribe to real-time changes"""
    try:
        # Subscribe to the notification_test table
        channel = supabase.realtime.channel('notification_changes')
        channel.on('postgres_changes',
            {
                'event': '*',  # Listen for all events (insert, update, delete)
                'schema': 'public',
                'table': 'notification_test'
            },
            on_change
        )
        await channel.subscribe()
        print("✅ Successfully subscribed to changes")
    except Exception as e:
        print(f"❌ Error subscribing to changes: {e}")

# Run the subscription
asyncio.run(subscribe_to_changes())



