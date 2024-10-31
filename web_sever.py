import asyncio
import websockets

async def handle_client(websocket, path):
    print("Client connected")
    try:
        while True:
            # 接收来自客户端的消息
            message = await websocket.recv()
            print(f"Received message from client: {message}")

            # 生成响应并发送回客户端（模拟流式响应）
            for i in range(3):  # 发送3条消息片段
                response = f"Server response part {i+1} to: {message}"
                await websocket.send(response)
                await asyncio.sleep(1)  # 模拟延时
            print("Response sent to client")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    server = await websockets.serve(handle_client, "localhost", 8000)
    print("WebSocket server started on ws://localhost:8000")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
