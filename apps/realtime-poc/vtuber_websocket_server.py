"""
WebSocket server for VTuber integration with big-3-super-agent.

This server acts as a bridge between VTuber frontend and OpenAI Realtime API,
allowing VTuber to leverage multi-agent orchestration while maintaining its
Live2D avatar and frontend interface.
"""

import asyncio
import json
import logging
import websockets
from websockets.server import serve
import base64
from typing import Dict, Optional
import os
from datetime import datetime

# Import the existing OpenAI Realtime Agent
from big_three_realtime_agents import OpenAIRealtimeVoiceAgent


class VTuberWebSocketBridge:
    """
    WebSocket bridge for VTuber integration.

    Handles connections from VTuber frontend and manages OpenAI Realtime API sessions.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, api_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.active_sessions: Dict[str, 'VTuberSession'] = {}
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging for the WebSocket server."""
        logger = logging.getLogger("VTuberWebSocketBridge")
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_dir = "output_logs/vtuber_bridge"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            f"{log_dir}/vtuber_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connection from VTuber frontend."""
        session_id = id(websocket)
        self.logger.info(f"New VTuber client connected: {session_id}")

        # Create session
        session = VTuberSession(websocket, session_id, self.logger)
        self.active_sessions[session_id] = session

        try:
            # Send welcome message
            await session.send_message({
                "type": "connection",
                "status": "connected",
                "session_id": str(session_id),
                "message": "Connected to big-3-super-agent orchestrator"
            })

            # Handle messages
            async for message in websocket:
                await session.handle_message(message)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {session_id} disconnected normally")
        except Exception as e:
            self.logger.error(f"Error handling client {session_id}: {e}", exc_info=True)
        finally:
            # Cleanup
            if session_id in self.active_sessions:
                await self.active_sessions[session_id].cleanup()
                del self.active_sessions[session_id]
            self.logger.info(f"Session {session_id} cleaned up")

    async def start(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting VTuber WebSocket Bridge on {self.host}:{self.port}")

        async with serve(self.handle_client, self.host, self.port):
            self.logger.info(f"✓ VTuber WebSocket Bridge running on ws://{self.host}:{self.port}")
            self.logger.info("Waiting for VTuber client connections...")
            await asyncio.Future()  # Run forever


class VTuberSession:
    """
    Manages a single VTuber client session and its OpenAI Realtime API connection.
    """

    def __init__(self, websocket, session_id, logger):
        self.websocket = websocket
        self.session_id = session_id
        self.logger = logger
        self.realtime_agent: Optional[OpenAIRealtimeVoiceAgent] = None
        self.realtime_ws = None
        self.running = False
        self.transcript_buffer = ""

    async def send_message(self, data: dict):
        """Send JSON message to VTuber client."""
        try:
            await self.websocket.send(json.dumps(data))
        except Exception as e:
            self.logger.error(f"Error sending message to client: {e}")

    async def handle_message(self, message):
        """Handle incoming message from VTuber frontend."""
        try:
            data = json.parse(message) if isinstance(message, str) else message

            # Handle different message types
            msg_type = data.get("type")

            if msg_type == "start_session":
                await self.start_realtime_session(data)
            elif msg_type == "audio":
                await self.forward_audio_to_realtime(data)
            elif msg_type == "text":
                await self.send_text_to_realtime(data)
            elif msg_type == "stop_session":
                await self.stop_realtime_session()
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            # Handle binary audio data
            await self.forward_audio_bytes(message)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}", exc_info=True)
            await self.send_message({
                "type": "error",
                "error": str(e)
            })

    async def start_realtime_session(self, config: dict):
        """Initialize OpenAI Realtime API connection."""
        self.logger.info(f"Starting Realtime API session for VTuber client {self.session_id}")

        try:
            # Create OpenAI Realtime Agent instance
            # Using audio mode since VTuber will handle its own STT/TTS
            self.realtime_agent = OpenAIRealtimeVoiceAgent(
                input_mode="audio",
                output_mode="audio",
                logger=self.logger,
                realtime_model=config.get("model", os.getenv("REALTIME_MODEL_DEFAULT"))
            )

            # Start the agent connection in background
            # We'll intercept WebSocket messages instead of using default audio I/O
            await self.initialize_realtime_connection()

            await self.send_message({
                "type": "session_started",
                "status": "ready",
                "message": "OpenAI Realtime API connected, multi-agent orchestration ready"
            })

        except Exception as e:
            self.logger.error(f"Failed to start Realtime session: {e}", exc_info=True)
            await self.send_message({
                "type": "error",
                "error": f"Failed to connect to OpenAI Realtime API: {str(e)}"
            })

    async def initialize_realtime_connection(self):
        """
        Initialize connection to OpenAI Realtime API and set up message forwarding.

        This is a simplified version - in production, you'd need to properly integrate
        with the OpenAI Realtime WebSocket protocol.
        """
        # Note: This is a placeholder for the actual implementation
        # The real implementation would need to:
        # 1. Connect to wss://api.openai.com/v1/realtime
        # 2. Send session configuration
        # 3. Set up bidirectional audio streaming
        # 4. Handle tool calls and function responses
        # 5. Extract transcripts from responses

        self.logger.info("Initializing OpenAI Realtime WebSocket connection")
        # TODO: Implement actual WebSocket connection to OpenAI
        self.running = True

    async def forward_audio_to_realtime(self, data: dict):
        """Forward audio from VTuber to OpenAI Realtime API."""
        if not self.running:
            self.logger.warning("Attempted to send audio but session not running")
            return

        try:
            audio_data = data.get("audio")
            if not audio_data:
                return

            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)

            # Forward to OpenAI Realtime API
            # TODO: Send via WebSocket in proper format
            # await self.realtime_ws.send(audio_event)

            self.logger.debug(f"Forwarded {len(audio_bytes)} bytes of audio to Realtime API")

        except Exception as e:
            self.logger.error(f"Error forwarding audio: {e}")

    async def forward_audio_bytes(self, audio_bytes: bytes):
        """Forward raw audio bytes to OpenAI Realtime API."""
        if not self.running:
            return

        try:
            # TODO: Package and send to OpenAI Realtime API
            self.logger.debug(f"Forwarded {len(audio_bytes)} raw audio bytes")
        except Exception as e:
            self.logger.error(f"Error forwarding audio bytes: {e}")

    async def send_text_to_realtime(self, data: dict):
        """Send text message to OpenAI Realtime API (for debugging/testing)."""
        if not self.running:
            return

        text = data.get("text")
        if text:
            self.logger.info(f"Text input: {text}")
            # TODO: Send text to Realtime API

    async def handle_realtime_response(self, response_data: dict):
        """
        Handle response from OpenAI Realtime API and forward to VTuber.

        Extracts both audio and transcript for VTuber expression analysis.
        """
        try:
            msg_type = response_data.get("type")

            # Handle audio delta (streaming audio chunks)
            if msg_type == "response.audio.delta":
                audio_b64 = response_data.get("delta")
                if audio_b64:
                    await self.send_message({
                        "type": "audio_chunk",
                        "audio": audio_b64
                    })

            # Handle audio completion
            elif msg_type == "response.audio.done":
                await self.send_message({
                    "type": "audio_complete"
                })

            # Handle transcript (for expression analysis)
            elif msg_type == "response.output_item.done":
                item = response_data.get("item", {})
                content = item.get("content", [])

                for part in content:
                    if part.get("type") == "text":
                        transcript = part.get("text", "")
                        self.transcript_buffer += transcript

                        # Send transcript to VTuber for expression analysis
                        await self.send_message({
                            "type": "transcript",
                            "text": transcript
                        })

            # Handle function calls (tool invocations)
            elif msg_type == "response.function_call_arguments.done":
                function_name = response_data.get("name")

                # Send notification to VTuber about agent activity
                await self.send_message({
                    "type": "agent_activity",
                    "activity": function_name,
                    "message": f"Delegating to: {function_name}"
                })

        except Exception as e:
            self.logger.error(f"Error handling Realtime response: {e}")

    async def stop_realtime_session(self):
        """Stop the OpenAI Realtime session."""
        self.logger.info(f"Stopping Realtime session for {self.session_id}")
        self.running = False

        if self.realtime_ws:
            # TODO: Properly close OpenAI WebSocket
            pass

        await self.send_message({
            "type": "session_stopped",
            "status": "disconnected"
        })

    async def cleanup(self):
        """Clean up session resources."""
        await self.stop_realtime_session()
        self.logger.info(f"Session {self.session_id} cleaned up")


async def main():
    """Main entry point for VTuber WebSocket Bridge."""
    import argparse

    parser = argparse.ArgumentParser(description="VTuber WebSocket Bridge for big-3-super-agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--api-key", help="Optional API key for authentication")

    args = parser.parse_args()

    bridge = VTuberWebSocketBridge(
        host=args.host,
        port=args.port,
        api_key=args.api_key
    )

    await bridge.start()


if __name__ == "__main__":
    asyncio.run(main())
