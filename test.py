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

    print("\n--- Testing Async generate_content_stream ---")
    try:
        print("Response stream: ", end="", flush=True)
        client_id, model = "Unknown", "Unknown"
        async for chunk in client.generate_content_stream("Write a short sentence about space."):
            print(chunk.text or "", end="", flush=True)
            client_id = getattr(chunk, 'client_id', 'Unknown')
            model = getattr(chunk, 'model', 'Unknown')
        print(f"\nSucceeded stream using: Client={client_id}, Model={model}")
    except Exception as e:
        print(f"\nAsync stream failed: {type(e).__name__}: {e}")

    print("\n--- Testing Sync generate_content_stream_sync ---")
    try:
        print("Response stream: ", end="", flush=True)
        client_id, model = "Unknown", "Unknown"
        for chunk in client.generate_content_stream_sync("Write a short sentence about oceans."):
            print(chunk.text or "", end="", flush=True)
            client_id = getattr(chunk, 'client_id', 'Unknown')
            model = getattr(chunk, 'model', 'Unknown')
        print(f"\nSucceeded stream using: Client={client_id}, Model={model}")
    except Exception as e:
        print(f"\nSync stream failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())