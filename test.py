import asyncio
from gemini_rotate import GeminiRotationClient
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = GeminiRotationClient()

    print("--- Testing Async generate_content ---")
    try:
        response = await client.generate_content("Hi!")
        print(f"Response: {response.text}")
        print(f"Succeeded using: Client={getattr(response, 'client_id', 'Unknown')}, Model={getattr(response, 'model', 'Unknown')}")
    except Exception as e:
        print(f"Async failed: {type(e).__name__}: {e}")

    print("\n--- Testing Sync generate_content_sync ---")
    try:
        response = client.generate_content_sync("Hola!")
        print(f"Response: {response.text}")
        print(f"Succeeded using: Client={getattr(response, 'client_id', 'Unknown')}, Model={getattr(response, 'model', 'Unknown')}")
    except Exception as e:
        print(f"Sync failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())