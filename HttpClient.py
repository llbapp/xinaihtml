import asyncio
import aiohttp


SERVER_URL = "http://localhost:8000"

async def test_chat():
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{SERVER_URL}/chat", json={"text": "What is AI?"}) as response:
            result = await response.json()
            print("Chat Result:", result)

async def test_batch_chat():
    async with aiohttp.ClientSession() as session:
        queries = ["What is machine learning?", "Explain deep learning", "What are neural networks?"]
        async with session.post(f"{SERVER_URL}/batch_chat", json={"queries": queries}) as response:
            results = await response.json()
            print("Batch Chat Results:")
            for result in results:
                print(f"Query: {result['query']}")
                print(f"Response: {result['response']}")
                print("---")


async def main():
    print("Testing /chat endpoint")
    await test_chat()
    print("\nTesting /batch_chat endpoint")
    await test_batch_chat()


if __name__ == "__main__":
    asyncio.run(main())