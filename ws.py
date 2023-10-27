import json

from fastapi import APIRouter, WebSocket, Header, Response
from starlette import status
from typing import Dict, Optional, Any

router = APIRouter(prefix="/ws")

ws_connection_to_GPU = None
current_user_id = ""


@router.websocket("/connect")
async def websocket_connect(websocket: WebSocket):
    try:
        await websocket.accept()
    except Exception as e:
        print(f'WebSocket connection failed: {e}')
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    await websocket.send_text("Connected to GPU server！")
    global ws_connection_to_GPU
    ws_connection_to_GPU = websocket
    print('WebSocket connection established')
    print(f'Current connections: {ws_connection_to_GPU}')

    try:
        while True:
            print(ws_connection_to_GPU.client_state)
            if websocket.client_state == "closed":
                break

            data = await websocket.receive_text()
            if data == 'close':
                await websocket.close()
                break
            else:
                print("here!!!!!!!!")
                await websocket.send_text(data)
    except Exception as e:
        print(f'WebSocket connection closed with exception {e}')
    print('WebSocket connection closed')


# only for testing
from pydantic import BaseModel


class WebSocketMessage(BaseModel):
    message: str


@router.post("/send_message")
async def send_message_to_user(message: str) -> Any:
    global ws_connection_to_GPU
    print(ws_connection_to_GPU)
    if ws_connection_to_GPU:
        try:
            await ws_connection_to_GPU.send_text(message)
            return {"status": "message sent"}
        except RuntimeError as e:
            return {"status": "not connected"}
    else:
        return {"status": "not connected"}


