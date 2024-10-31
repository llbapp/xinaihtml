import asyncio
import aiohttp
import websockets
import json

WS_URL = "ws://localhost:8000"


async def test_stream_chat():
    async with websockets.connect(f"{WS_URL}/stream_chat") as websocket:
        query = "我喜欢加菲猫吗？"
        await websocket.send(query)
        print(f"Sent query: {query}")
        
        full_response = ""
        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received: {response}")
                full_response += response
            except asyncio.TimeoutError:
                print("Stream completed or timed out")
                break
        
        print("Full streamed response:", full_response)

async def main():
    print("\nTesting /stream_chat WebSocket")
    await test_stream_chat()

if __name__ == "__main__":
    asyncio.run(main())