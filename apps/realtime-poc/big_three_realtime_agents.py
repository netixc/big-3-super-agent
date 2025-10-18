#!/usr/bin/env python3
"""
Big Three Realtime Agents - Unified Agent System
=================================================

Unified system combining three powerful agent types:
1. OpenAIRealtimeVoiceAgent - OpenAI Realtime API for voice interactions
2. ClaudeCodeAgenticCoder - Claude Code SDK agents for software development
3. GeminiBrowserAgent - Gemini Computer Use for browser automation

Each agent class is self-contained with all necessary functionality.

Usage:
    # Auto-prompt mode (text only)
    uv run big_three_realtime_agents.py --prompt "Create an agent and have it make changes"

    # Interactive text mode
    uv run big_three_realtime_agents.py --input text --output text

    # Full voice interaction
    uv run big_three_realtime_agents.py --input audio --output audio

    # Use mini model
    uv run big_three_realtime_agents.py --mini --input text --output text

Arguments:
    --input {text,audio}   Input mode (default: text)
    --output {text,audio}  Output mode (default: text)
    --prompt TEXT          Auto-dispatch prompt (forces text mode)
    --mini                 Use mini realtime model
"""

# /// script
# dependencies = [
#     "websocket-client",
#     "pyaudio",
#     "python-dotenv",
#     "rich",
#     "claude-agent-sdk",
#     "google-genai",
#     "playwright",
#     "numpy",
#     "pynput",
# ]
# ///

import os
import json
import base64
import logging
import threading
import argparse
import asyncio
import textwrap
import urllib.request
import urllib.error
import shutil
import time
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import websocket
import pyaudio
import numpy as np
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

# Environment setup
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    raise ImportError(
        "dotenv not found. Please install it with `pip install python-dotenv`"
    )

# Claude CLI configuration - No SDK needed, using CLI with subscription
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", None)

def find_claude_cli() -> str:
    """Find Claude CLI executable."""
    if CLAUDE_CLI_PATH and os.path.isabs(CLAUDE_CLI_PATH):
        return CLAUDE_CLI_PATH

    # Try local install path
    home_path = os.path.join(os.path.expanduser("~"), ".claude", "local", "claude")
    if os.path.exists(home_path):
        return home_path

    # Fallback to system PATH
    return "claude"

# Gemini imports
try:
    from google import genai
    from google.genai import types
    from google.genai.types import Content, Part
except ImportError as exc:
    raise ImportError(
        "google-genai not found. Install with `pip install google-genai`."
    ) from exc

# Playwright imports
try:
    from playwright.sync_api import sync_playwright, Page
except ImportError as exc:
    raise ImportError(
        "playwright not found. Install with `pip install playwright` and run `playwright install`."
    ) from exc

# ================================================================
# Constants
# ================================================================

# OpenAI Realtime API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REALTIME_API_URL = os.environ.get("REALTIME_API_URL", "wss://api.openai.com/v1/realtime")
REALTIME_MODEL_DEFAULT = os.environ.get("REALTIME_MODEL_DEFAULT", "gpt-realtime-2025-08-28")
REALTIME_MODEL_MINI = os.environ.get("REALTIME_MODEL_MINI", "gpt-realtime-mini-2025-10-06")
REALTIME_API_URL_TEMPLATE = f"{REALTIME_API_URL}?model={{model}}"
REALTIME_VOICE_CHOICE = os.environ.get("REALTIME_AGENT_VOICE", "shimmer")
BROWSER_TOOL_STARTING_URL = os.environ.get(
    "BROWSER_TOOL_STARTING_URL", "localhost:3333"
)

# Audio configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000

# Claude Code configuration
DEFAULT_CLAUDE_MODEL = os.environ.get(
    "CLAUDE_AGENT_MODEL", "claude-sonnet-4-5-20250929"
)
ENGINEER_NAME = os.environ.get("ENGINEER_NAME", "Dan")
REALTIME_ORCH_AGENT_NAME = os.environ.get("REALTIME_ORCH_AGENT_NAME", "ada")
CLAUDE_CODE_TOOL = "claude_code"
CLAUDE_CODE_TOOL_SLUG = "claude_code"
AGENTIC_CODING_TYPE = "agentic_coding"

# Gemini Computer Use configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-computer-use-preview-10-2025"
GEMINI_TOOL = "gemini"
GEMINI_TOOL_SLUG = "gemini"
AGENTIC_BROWSERING_TYPE = "agentic_browsering"
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

# Common agent configuration (defined after AGENT_WORKING_DIRECTORY below)
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Agent working directory - set to content-gen app
AGENT_WORKING_DIRECTORY = Path(__file__).parent.parent / "content-gen"

# Set AGENTS_BASE_DIR relative to working directory for consolidated outputs
AGENTS_BASE_DIR = AGENT_WORKING_DIRECTORY / "agents"
CLAUDE_CODE_REGISTRY_PATH = AGENTS_BASE_DIR / CLAUDE_CODE_TOOL_SLUG / "registry.json"
GEMINI_REGISTRY_PATH = AGENTS_BASE_DIR / GEMINI_TOOL_SLUG / "registry.json"

# Console for rich output
console = Console()


# ================================================================
# GeminiBrowserAgent - Browser automation with Gemini Computer Use
# ================================================================


class GeminiBrowserAgent:
    """
    Browser automation agent powered by Gemini Computer Use API.

    Handles web browsing, navigation, and interaction tasks using
    Gemini's vision and action planning capabilities with Playwright.
    """

    def __init__(self, logger=None):
        """Initialize browser agent."""
        self.logger = logger or logging.getLogger("GeminiBrowserAgent")

        # Validate Gemini API key
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

        # Browser automation state
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        # Screenshot session setup - persistent for entire browser session
        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        )
        self.screenshot_dir = Path("output_screenshots") / self.session_id
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_counter = 0

        # Registry management for browser agents
        self.registry_lock = threading.Lock()
        self.agent_registry = self._load_agent_registry()

        self.logger.info(f"Browser session ID: {self.session_id}")
        self.logger.info(f"Screenshots will be saved to: {self.screenshot_dir}")
        self.logger.info("Initialized GeminiBrowserAgent")

    # ------------------------------------------------------------------ #
    # Agent registry helpers
    # ------------------------------------------------------------------ #

    def _load_agent_registry(self) -> Dict[str, Any]:
        """Load agent registry from disk."""
        if not GEMINI_REGISTRY_PATH.exists():
            return {"agents": {}}

        try:
            with GEMINI_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if "agents" not in data:
                    data["agents"] = {}
                return data
        except Exception as exc:
            self.logger.error(f"Failed to load agent registry: {exc}")
            return {"agents": {}}

    def _save_agent_registry(self):
        """Save agent registry to disk."""
        GEMINI_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with GEMINI_REGISTRY_PATH.open("w", encoding="utf-8") as fh:
                json.dump(self.agent_registry, fh, indent=2)
        except Exception as exc:
            self.logger.error(f"Failed to save agent registry: {exc}")

    def _register_agent(self, agent_name: str, metadata: Dict[str, Any]):
        """Register a browser agent in the registry."""
        with self.registry_lock:
            self.agent_registry.setdefault("agents", {})[agent_name] = {
                "tool": GEMINI_TOOL,
                "type": AGENTIC_BROWSERING_TYPE,
                "created_at": metadata.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                ),
                "session_id": metadata.get("session_id", self.session_id),
            }
            self._save_agent_registry()

    def _get_agent_by_name(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by name."""
        return self.agent_registry.get("agents", {}).get(agent_name)

    def _agent_directory(self, agent_name: str) -> Path:
        """Get agent working directory path."""
        return AGENTS_BASE_DIR / GEMINI_TOOL_SLUG / agent_name

    # ------------------------------------------------------------------ #
    # Browser automation
    # ------------------------------------------------------------------ #

    def setup_browser(self):
        """Initialize Playwright browser."""
        try:
            self.logger.info("Initializing browser...")
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=False)
            self.context = self.browser.new_context(
                viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
            )
            self.page = self.context.new_page()
            self.logger.info("Browser ready!")
        except Exception as e:
            self.logger.error(f"Failed to initialize browser: {e}")
            raise

    def cleanup_browser(self):
        """Clean up Playwright browser resources."""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.logger.info("Browser cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Browser cleanup error: {e}")

    def execute_task(
        self, task: str, url: Optional[str] = BROWSER_TOOL_STARTING_URL
    ) -> Dict[str, Any]:
        """
        Execute a browser automation task.

        Args:
            task: Description of the browsing task to perform
            url: Optional starting URL

        Returns:
            Dictionary with ok status and either data or error
        """
        try:
            self.logger.info(f"Task: {task}")
            self.logger.info(f"Starting URL: {url or BROWSER_TOOL_STARTING_URL}")
            self.logger.info(f"Session ID: {self.session_id}")

            # Setup browser if not already done
            if not self.page:
                self.setup_browser()

            # Navigate to starting URL if provided
            if url:
                self.page.goto(url, wait_until="networkidle", timeout=10000)
                self.logger.info(f"Navigated to: {url}")
            else:
                # Start with a search engine
                self.page.goto(
                    "https://www.google.com", wait_until="networkidle", timeout=10000
                )
                self.logger.info("Starting from Google")

            # Run the browser automation loop
            result = self._run_browser_automation_loop(task)

            self.logger.info(
                f"Task completed! Screenshots saved to: {self.screenshot_dir}"
            )

            return {
                "ok": True,
                "data": result,
                "screenshot_dir": str(self.screenshot_dir),
            }

        except Exception as exc:
            self.logger.exception("Browser automation failed")
            return {"ok": False, "error": str(exc)}

    def _run_browser_automation_loop(self, task: str, max_turns: int = 30) -> str:
        """
        Run the Gemini Computer Use agent loop to complete the task.

        Args:
            task: The browsing task to complete
            max_turns: Maximum number of agent turns

        Returns:
            The final result as a string
        """
        # Configure Gemini with Computer Use
        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER
                    )
                )
            ],
        )

        # Initial screenshot
        initial_screenshot = self.page.screenshot(type="png")

        # Save initial screenshot
        timestamp = datetime.now().strftime("%H%M%S")
        screenshot_path = (
            self.screenshot_dir
            / f"step_{self.screenshot_counter:02d}_initial_{timestamp}.png"
        )
        self.page.screenshot(path=str(screenshot_path))
        self.logger.info(f"Saved initial screenshot: {screenshot_path}")
        self.screenshot_counter += 1

        # Build initial contents
        contents = [
            Content(
                role="user",
                parts=[
                    Part(text=task),
                    Part.from_bytes(data=initial_screenshot, mime_type="image/png"),
                ],
            )
        ]

        self.logger.info(f"Starting browser automation loop for task: {task}")

        # Agent loop
        for turn in range(max_turns):
            self.logger.info(f"Turn {turn + 1}/{max_turns}")

            try:
                # Get response from Gemini
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )

                candidate = response.candidates[0]
                contents.append(candidate.content)

                # Check if there are function calls
                has_function_calls = any(
                    part.function_call for part in candidate.content.parts
                )

                if not has_function_calls:
                    # No more actions - extract final text response
                    text_response = " ".join(
                        [part.text for part in candidate.content.parts if part.text]
                    )
                    self.logger.info(f"Agent finished: {text_response}")

                    console.print(Panel(text_response, title="GeminiBrowserAgent"))

                    # Save final screenshot
                    timestamp = datetime.now().strftime("%H%M%S")
                    screenshot_path = (
                        self.screenshot_dir
                        / f"step_{self.screenshot_counter:02d}_final_{timestamp}.png"
                    )
                    self.page.screenshot(path=str(screenshot_path))
                    self.logger.info(f"Saved final screenshot: {screenshot_path}")
                    self.screenshot_counter += 1

                    return text_response

                # Execute function calls
                self.logger.info("Executing browser actions...")
                results = self._execute_gemini_function_calls(candidate)

                # Get function responses with new screenshot
                function_responses = self._get_gemini_function_responses(results)

                # Save screenshot after actions
                timestamp = datetime.now().strftime("%H%M%S")
                screenshot_path = (
                    self.screenshot_dir
                    / f"step_{self.screenshot_counter:02d}_{timestamp}.png"
                )
                self.page.screenshot(path=str(screenshot_path))
                self.logger.info(f"Saved screenshot: {screenshot_path}")
                self.screenshot_counter += 1

                # Add function responses to contents
                contents.append(
                    Content(
                        role="user",
                        parts=[Part(function_response=fr) for fr in function_responses],
                    )
                )

            except Exception as e:
                self.logger.error(f"Error in browser automation loop: {e}")
                raise

        # If we hit max turns, return what we have
        return f"Task reached maximum turns ({max_turns}). Please check browser state."

    def _execute_gemini_function_calls(self, candidate) -> list:
        """Execute Gemini Computer Use function calls using Playwright."""
        results = []
        function_calls = [
            part.function_call for part in candidate.content.parts if part.function_call
        ]

        for function_call in function_calls:
            fname = function_call.name
            args = function_call.args
            self.logger.info(f"Executing Gemini action: {fname}")

            action_result = {}

            try:
                if fname == "open_web_browser":
                    pass  # Already open
                elif fname == "wait_5_seconds":
                    time.sleep(5)
                elif fname == "go_back":
                    self.page.go_back()
                elif fname == "go_forward":
                    self.page.go_forward()
                elif fname == "search":
                    self.page.goto("https://www.google.com")
                elif fname == "navigate":
                    self.page.goto(args["url"], wait_until="networkidle", timeout=10000)
                elif fname == "click_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    self.page.mouse.click(actual_x, actual_y)
                elif fname == "hover_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    self.page.mouse.move(actual_x, actual_y)
                elif fname == "type_text_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    text = args["text"]
                    press_enter = args.get("press_enter", True)
                    clear_before = args.get("clear_before_typing", True)

                    self.page.mouse.click(actual_x, actual_y)
                    if clear_before:
                        self.page.keyboard.press("Meta+A")
                        self.page.keyboard.press("Backspace")
                    self.page.keyboard.type(text)
                    if press_enter:
                        self.page.keyboard.press("Enter")
                elif fname == "key_combination":
                    keys = args["keys"]
                    self.page.keyboard.press(keys)
                elif fname == "scroll_document":
                    direction = args["direction"]
                    if direction == "down":
                        self.page.keyboard.press("PageDown")
                    elif direction == "up":
                        self.page.keyboard.press("PageUp")
                    elif direction == "left":
                        self.page.keyboard.press("ArrowLeft")
                    elif direction == "right":
                        self.page.keyboard.press("ArrowRight")
                elif fname == "scroll_at":
                    actual_x = self._denormalize_x(args["x"])
                    actual_y = self._denormalize_y(args["y"])
                    direction = args["direction"]
                    magnitude = args.get("magnitude", 800)

                    # Scroll by moving to position and using wheel
                    self.page.mouse.move(actual_x, actual_y)
                    scroll_amount = int(magnitude * SCREEN_HEIGHT / 1000)
                    if direction == "down":
                        self.page.mouse.wheel(0, scroll_amount)
                    elif direction == "up":
                        self.page.mouse.wheel(0, -scroll_amount)
                    elif direction == "left":
                        self.page.mouse.wheel(-scroll_amount, 0)
                    elif direction == "right":
                        self.page.mouse.wheel(scroll_amount, 0)
                elif fname == "drag_and_drop":
                    x = self._denormalize_x(args["x"])
                    y = self._denormalize_y(args["y"])
                    dest_x = self._denormalize_x(args["destination_x"])
                    dest_y = self._denormalize_y(args["destination_y"])

                    self.page.mouse.move(x, y)
                    self.page.mouse.down()
                    self.page.mouse.move(dest_x, dest_y)
                    self.page.mouse.up()
                else:
                    self.logger.warning(f"Unimplemented action: {fname}")

                # Wait for potential navigations/renders
                self.page.wait_for_load_state(timeout=5000)
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Error executing {fname}: {e}")
                action_result = {"error": str(e)}

            results.append((fname, action_result))

        return results

    def _get_gemini_function_responses(self, results: list):
        """Generate function responses with current screenshot."""
        screenshot_bytes = self.page.screenshot(type="png")
        current_url = self.page.url
        function_responses = []

        for name, result in results:
            response_data = {"url": current_url}
            response_data.update(result)
            function_responses.append(
                types.FunctionResponse(
                    name=name,
                    response=response_data,
                    parts=[
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type="image/png", data=screenshot_bytes
                            )
                        )
                    ],
                )
            )

        return function_responses

    def _denormalize_x(self, x: int) -> int:
        """Convert normalized x coordinate (0-999) to actual pixel coordinate."""
        return int(x / 1000 * SCREEN_WIDTH)

    def _denormalize_y(self, y: int) -> int:
        """Convert normalized y coordinate (0-999) to actual pixel coordinate."""
        return int(y / 1000 * SCREEN_HEIGHT)


# ================================================================
# ClaudeCodeAgenticCoder - Claude Code agent orchestration
# ================================================================


class ClaudeCodeAgenticCoder:
    """
    Manages Claude Code agents for software development tasks using CLI (subscription-based).

    Handles agent creation, command dispatch, and result retrieval.
    Uses Claude CLI with --dangerously-skip-permissions for cost-effective execution.
    """

    def __init__(self, logger=None, browser_agent=None, completion_callback=None):
        """Initialize agentic coder manager."""
        self.logger = logger or logging.getLogger("ClaudeCodeAgenticCoder")
        self.browser_agent = browser_agent
        self.completion_callback = completion_callback  # Callback when task completes

        self.registry_lock = threading.Lock()
        self.agent_registry = self._load_agent_registry()

        self.background_threads: list[threading.Thread] = []
        self.claude_cli_path = find_claude_cli()

        self.logger.info(f"Initialized ClaudeCodeAgenticCoder (CLI-based using {self.claude_cli_path})")

    # ------------------------------------------------------------------ #
    # Agent registry helpers
    # ------------------------------------------------------------------ #

    def _load_agent_registry(self) -> Dict[str, Any]:
        """Load agent registry from disk."""
        if not CLAUDE_CODE_REGISTRY_PATH.exists():
            return {"agents": {}}

        try:
            with CLAUDE_CODE_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if "agents" not in data:
                    data["agents"] = {}
                return data
        except Exception as exc:
            self.logger.error(f"Failed to load agent registry: {exc}")
            return {"agents": {}}

    def _save_agent_registry(self):
        """Save agent registry to disk."""
        CLAUDE_CODE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with CLAUDE_CODE_REGISTRY_PATH.open("w", encoding="utf-8") as fh:
                json.dump(self.agent_registry, fh, indent=2)
        except Exception as exc:
            self.logger.error(f"Failed to save agent registry: {exc}")

    def _register_agent(
        self, agent_name: str, session_id: str, metadata: Dict[str, Any]
    ):
        """Register an agent in the registry."""
        with self.registry_lock:
            self.agent_registry.setdefault("agents", {})[agent_name] = {
                "session_id": session_id,
                "tool": metadata.get("tool", CLAUDE_CODE_TOOL),
                "type": metadata.get("type", AGENTIC_CODING_TYPE),
                "created_at": metadata.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                ),
                "working_dir": metadata.get(
                    "working_dir", str(AGENT_WORKING_DIRECTORY)
                ),
                "operator_files": metadata.get("operator_files", []),
            }
            self._save_agent_registry()

    def _get_agent_by_name(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by name."""
        return self.agent_registry.get("agents", {}).get(agent_name)

    def _agent_directory(self, agent_name: str) -> Path:
        """Get agent working directory path."""
        return AGENTS_BASE_DIR / CLAUDE_CODE_TOOL_SLUG / agent_name

    # ------------------------------------------------------------------ #
    # CLI execution helper
    # ------------------------------------------------------------------ #

    def _execute_claude_cli(
        self, prompt: str, working_dir: str, timeout: int = 1800
    ) -> Dict[str, Any]:
        """Execute Claude CLI with prompt."""
        try:
            result = subprocess.run(
                [self.claude_cli_path, "--dangerously-skip-permissions", "-p", prompt],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                return {"ok": True, "output": result.stdout, "error": result.stderr}
            else:
                return {
                    "ok": False,
                    "error": f"CLI exited with code {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}",
                }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"CLI execution timed out after {timeout} seconds",
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "error": f"Claude CLI not found at {self.claude_cli_path}. Please install it or set CLAUDE_CLI_PATH environment variable.",
            }
        except Exception as exc:
            return {"ok": False, "error": f"CLI execution failed: {exc}"}

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #

    def _read_prompt(self, relative_path: str) -> str:
        """Read prompt file from super_agent directory."""
        prompt_path = PROMPTS_DIR / "super_agent" / relative_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to read prompt {relative_path}: {exc}") from exc

    def _render_prompt(self, relative_path: str, **kwargs: Any) -> str:
        """Render prompt template with variables."""
        template = self._read_prompt(relative_path)
        if kwargs:
            return template.format(**kwargs)
        return template


    # ------------------------------------------------------------------ #
    # Public API - Tool implementations
    # ------------------------------------------------------------------ #

    def list_agents(self) -> Dict[str, Any]:
        """List all registered agents."""
        agents_payload: list[Dict[str, Any]] = []
        for name, data in sorted(self.agent_registry.get("agents", {}).items()):
            agents_payload.append(
                {
                    "name": name,
                    "session_id": data.get("session_id"),
                    "tool": data.get("tool"),
                    "type": data.get("type"),
                    "created_at": data.get("created_at"),
                    "working_dir": data.get("working_dir"),
                    "operator_files": data.get("operator_files", []),
                }
            )
        return {"ok": True, "agents": agents_payload}

    def create_agent(
        self,
        tool: str = CLAUDE_CODE_TOOL,
        agent_type: str = AGENTIC_CODING_TYPE,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new Claude Code agent."""
        # Validate tool
        if tool != CLAUDE_CODE_TOOL:
            return {
                "ok": False,
                "error": f"Unsupported tool '{tool}'. Only '{CLAUDE_CODE_TOOL}' is supported by this handler.",
            }

        # Validate agent type
        if agent_type != AGENTIC_CODING_TYPE:
            return {
                "ok": False,
                "error": f"Unsupported agent type '{agent_type}'. Only '{AGENTIC_CODING_TYPE}' is supported by this handler.",
            }

        preferred_name = agent_name.strip() if agent_name else None
        if preferred_name and self._get_agent_by_name(preferred_name):
            return {
                "ok": False,
                "error": (
                    f"Agent '{preferred_name}' already exists. Choose a different name or omit agent_name."
                ),
            }

        try:
            agent_info = self._create_new_agent_simple(
                tool=tool, agent_type=agent_type, agent_name=preferred_name
            )
        except Exception as exc:
            self.logger.exception("create_agent failed")
            return {"ok": False, "error": f"Failed to create agent: {exc}"}

        return {
            "ok": True,
            "agent_name": agent_info["name"],
            "session_id": agent_info["session_id"],
        }

    def command_agent(self, agent_name: str, prompt: str) -> Dict[str, Any]:
        """Dispatch command to a Claude Code agent."""
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' not found. Create it first.",
            }

        # Validate this is a Claude Code agent
        if agent.get("type") != AGENTIC_CODING_TYPE:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' is not a {AGENTIC_CODING_TYPE} agent. Wrong handler.",
            }

        # Prepare operator file and dispatch command
        try:
            operator_path = self._prepare_operator_file(name=agent_name, prompt=prompt)
        except Exception as exc:
            self.logger.exception("Failed to prepare operator file")
            return {"ok": False, "error": f"Could not prepare operator log: {exc}"}

        thread = threading.Thread(
            target=self._run_agent_command_thread,
            args=(agent_name, prompt, operator_path),
            daemon=True,
        )
        thread.start()
        self.background_threads.append(thread)

        return {"ok": True, "operator_file": str(operator_path)}

    def check_agent_result(
        self, agent_name: str, operator_file_name: str
    ) -> Dict[str, Any]:
        """Read operator status report."""
        agent_dir = self._agent_directory(agent_name)
        operator_path = agent_dir / operator_file_name

        if not operator_path.exists():
            return {"ok": False, "error": f"Operator file not found: {operator_path}"}

        try:
            content = operator_path.read_text(encoding="utf-8")
        except Exception as exc:
            self.logger.error(f"Failed to read operator file: {exc}")
            return {"ok": False, "error": f"Failed to read operator file: {exc}"}

        return {"ok": True, "content": content}

    def delete_agent(self, agent_name: str) -> Dict[str, Any]:
        """Delete an agent."""
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' not found. Nothing to delete.",
            }

        warnings: list[str] = []

        with self.registry_lock:
            self.agent_registry.get("agents", {}).pop(agent_name, None)
            try:
                self._save_agent_registry()
            except Exception as exc:
                warnings.append(f"Failed to update registry: {exc}")

        agent_dir = self._agent_directory(agent_name)
        if agent_dir.exists():
            try:
                shutil.rmtree(agent_dir)
            except Exception as exc:
                warnings.append(f"Failed to remove directory {agent_dir}: {exc}")

        payload: Dict[str, Any] = {"ok": True, "agent_name": agent_name}
        if warnings:
            payload["warnings"] = warnings
        return payload

    # ------------------------------------------------------------------ #
    # CLI-based agent operations (no SDK, uses subscription)
    # ------------------------------------------------------------------ #

    def _create_new_agent_simple(
        self, tool: str, agent_type: str, agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create new Claude Code agent (CLI-based, no SDK)."""
        existing_names = list(self.agent_registry.get("agents", {}).keys())

        if agent_name:
            final_name = agent_name
        else:
            # Simple name generation without AI
            timestamp = datetime.now(timezone.utc).strftime("%H%M%S")
            final_name = f"Agent{timestamp}"
            # Dedupe
            suffix = 1
            while final_name in existing_names:
                final_name = f"Agent{timestamp}_{suffix}"
                suffix += 1

        agent_dir = self._agent_directory(final_name)
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Generate simple session ID
        session_id = f"{final_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        metadata = {
            "tool": tool,
            "type": agent_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "working_dir": str(AGENT_WORKING_DIRECTORY),
        }
        self._register_agent(final_name, session_id, metadata)

        self.logger.info(f"Created CLI-based agent '{final_name}' - session_id: {session_id}")
        console.print(
            Panel(
                f"Agent '{final_name}' created successfully!",
                title=f"Agent '{final_name}'",
                border_style="green",
            )
        )

        return {
            "name": final_name,
            "session_id": session_id,
            "directory": str(agent_dir),
        }

    def _prepare_operator_file(self, name: str, prompt: str) -> Path:
        """Prepare operator log file for task."""
        agent_dir = self._agent_directory(name)
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Simple filename generation
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        filename = f"task-{timestamp}.md"
        operator_path = agent_dir / filename

        header = textwrap.dedent(
            f"""
            # Operator Log · {name}

            **Task:** {prompt}
            **Created:** {datetime.now(timezone.utc).isoformat()}

            ## Status
            - Pending dispatch to agent.

            ---
            """
        ).strip()
        operator_path.write_text(header + "\n", encoding="utf-8")
        self._record_operator_file(name, operator_path)
        return operator_path

    def _run_agent_command_thread(
        self, agent_name: str, prompt: str, operator_path: Path
    ):
        """Run agent command in background thread using CLI."""
        try:
            self._run_existing_agent_cli(agent_name, prompt, operator_path)
        except Exception as exc:
            self.logger.exception(f"Background command for '{agent_name}' failed")
            failure_note = textwrap.dedent(
                f"""
                ## Operator Update
                - **Status:** Failed to dispatch command.
                - **Error:** {exc}
                - **Timestamp:** {datetime.now(timezone.utc).isoformat()}
                """
            ).strip()
            with operator_path.open("a", encoding="utf-8") as fh:
                fh.write("\n" + failure_note + "\n")

    def _run_existing_agent_cli(
        self, agent_name: str, prompt: str, operator_path: Path
    ):
        """Run command on existing agent using CLI."""
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            raise RuntimeError(f"Agent '{agent_name}' not found in registry.")

        working_dir = agent.get("working_dir", str(AGENT_WORKING_DIRECTORY))

        kickoff_note = textwrap.dedent(
            f"""
            ## Operator Update
            - **Status:** Task dispatched for execution via CLI.
            - **Prompt:** {prompt}
            - **Operator Log:** {operator_path}
            - **Timestamp:** {datetime.now(timezone.utc).isoformat()}
            """
        ).strip()

        with operator_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + kickoff_note + "\n")

        # Execute Claude CLI
        result = self._execute_claude_cli(prompt, working_dir)

        # Update operator log with result
        if result.get("ok"):
            completion_note = textwrap.dedent(
                f"""
                ## Operator Update
                - **Status:** Task completed successfully.
                - **Timestamp:** {datetime.now(timezone.utc).isoformat()}

                ### Output
                ```
                {result.get('output', '')}
                ```

                ### Stderr
                ```
                {result.get('error', '')}
                ```
                """
            ).strip()
            console.print(
                Panel(
                    f"Task completed for agent '{agent_name}'",
                    title=f"Agent '{agent_name}'",
                    border_style="green",
                )
            )

            # Notify via callback if available
            if self.completion_callback:
                self.completion_callback(
                    agent_name=agent_name,
                    status="completed",
                    message=f"Agent '{agent_name}' has completed its task successfully.",
                )
        else:
            completion_note = textwrap.dedent(
                f"""
                ## Operator Update
                - **Status:** Task failed.
                - **Error:** {result.get('error', 'Unknown error')}
                - **Timestamp:** {datetime.now(timezone.utc).isoformat()}
                """
            ).strip()
            console.print(
                Panel(
                    f"Task failed for agent '{agent_name}': {result.get('error')}",
                    title=f"Agent '{agent_name}'",
                    border_style="red",
                )
            )

            # Notify via callback if available
            if self.completion_callback:
                self.completion_callback(
                    agent_name=agent_name,
                    status="failed",
                    message=f"Agent '{agent_name}' task failed: {result.get('error', 'Unknown error')}",
                )

        with operator_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + completion_note + "\n")

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #

    def _record_operator_file(self, agent_name: str, operator_path: Path) -> None:
        """Record operator file in registry."""
        with self.registry_lock:
            agents = self.agent_registry.setdefault("agents", {})
            agent_entry = agents.setdefault(
                agent_name,
                {
                    "operator_files": [],
                    "session_id": None,
                    "tool": CLAUDE_CODE_TOOL,
                    "type": AGENTIC_CODING_TYPE,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "working_dir": str(AGENT_WORKING_DIRECTORY),
                },
            )
            files = agent_entry.setdefault("operator_files", [])
            path_str = str(operator_path)
            if path_str not in files:
                files.append(path_str)
            self._save_agent_registry()


# ================================================================
# OpenAIRealtimeVoiceAgent - Main orchestrator
# ================================================================


class OpenAIRealtimeVoiceAgent:
    """
    OpenAI Realtime Voice Agent with agentic coding and browser automation.

    Orchestrates voice interactions via OpenAI Realtime API and delegates
    tasks to Claude Code agents and Gemini browser automation.
    """

    def __init__(
        self,
        input_mode: str = "text",
        output_mode: str = "text",
        logger=None,
        realtime_model: str | None = None,
        startup_prompt: Optional[str] = None,
        auto_timeout: int = 60,
    ):
        """Initialize the unified voice agent."""
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.logger = logger or logging.getLogger("OpenAIRealtimeVoiceAgent")
        self.realtime_model = realtime_model or REALTIME_MODEL_DEFAULT
        self.ws = None
        self.audio_queue = []
        self.running = False
        self.audio_interface = None
        self.audio_stream = None
        self.console = Console()

        # Validate OpenAI API key
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Auto mode settings
        self.startup_prompt = startup_prompt
        self.auto_mode = startup_prompt is not None
        self.awaiting_auto_close = False
        self.auto_timeout = auto_timeout
        self.auto_start_time = None

        # Tool call tracking
        self.default_output_modalities = (
            ["audio"] if self.output_mode == "audio" else ["text"]
        )
        self.pending_function_arguments: Dict[str, str] = {}
        self.completed_function_calls: set[str] = set()
        self.current_response_function_calls: set[str] = set()  # Track function calls in current response

        # Initialize sub-agents
        self.browser_agent = GeminiBrowserAgent(logger=self.logger)
        self.agentic_coder = ClaudeCodeAgenticCoder(
            logger=self.logger,
            browser_agent=self.browser_agent,
            completion_callback=self._on_agent_task_complete,
        )

        # Build tool specs
        self.tool_specs = self._build_tool_specs()

        # Background threads
        self.background_threads: list[threading.Thread] = []

        # Audio pause control (shift+space toggle)
        self.audio_paused = False  # Manual pause via shift+space
        self.auto_paused_for_response = False  # Auto-pause during agent speech
        self.keyboard_listener = None
        self.shift_pressed = False  # Track shift key state

        # Token tracking and cost analysis
        self.response_count = 0
        self.token_summary_interval = 3  # Show summary every N responses
        self.cumulative_tokens = {
            "total": 0,
            "input": 0,
            "output": 0,
            "input_text": 0,
            "input_audio": 0,
            "output_text": 0,
            "output_audio": 0,
        }
        self.cumulative_cost_usd = 0.0

        # Latency tracking
        self.speech_stopped_timestamp = None
        self.first_audio_delta_timestamp = None

        self.logger.info(
            f"Initialized OpenAIRealtimeVoiceAgent - Input: {input_mode}, Output: {output_mode}"
        )
        self._log_tool_catalog()

    # ------------------------------------------------------------------ #
    # Logging and UI
    # ------------------------------------------------------------------ #

    def _log_panel(
        self,
        message: str,
        *,
        title: str = "Agent",
        style: str = "cyan",
        level: str = "info",
        expand: bool = True,
    ) -> None:
        """Log message to both console panel and file logger."""
        console.print(Panel(message, title=title, border_style=style, expand=expand))
        log_fn = getattr(self.logger, level, None)
        if log_fn:
            log_fn(message)

    def _log_tool_catalog(self) -> None:
        """Display available tools in a panel."""
        if not getattr(self, "tool_specs", None):
            return
        entries: list[str] = []
        for spec in self.tool_specs:
            name = spec.get("name", "unknown_tool")
            properties = spec.get("parameters", {}).get("properties", {}) or {}
            params = ", ".join(properties.keys())
            if params:
                entries.append(f"{name}({params})")
            else:
                entries.append(f"{name}()")

        syntax = Syntax(
            json.dumps(entries, indent=2, ensure_ascii=False),
            "json",
            theme="monokai",
            word_wrap=True,
        )
        console.print(
            Panel(
                syntax,
                title="Tool Catalog",
                border_style="cyan",
                expand=True,
            )
        )
        self.logger.info("Tool catalog loaded with %d tools", len(self.tool_specs))

    def _log_agent_roster_panel(self, agents_payload: list[Dict[str, Any]]) -> None:
        """Display agent roster in a table."""
        if not agents_payload:
            self._log_panel(
                "No registered agents yet. Use create_agent to spin one up.",
                title="Agent Roster",
                style="yellow",
            )
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="bold")
        table.add_column("Session ID", overflow="fold")
        table.add_column("Type")
        table.add_column("Tool")
        table.add_column("Recent File", overflow="fold")
        # table.add_column("All Files", overflow="fold")

        for agent in agents_payload:
            files = agent.get("operator_files") or []

            # Extract relative paths from AGENT_WORKING_DIRECTORY
            relative_paths = []
            if files:
                for f in files:
                    try:
                        rel_path = Path(f).relative_to(AGENT_WORKING_DIRECTORY)
                        relative_paths.append(str(rel_path))
                    except ValueError:
                        # If path is not relative to AGENT_WORKING_DIRECTORY, use filename only
                        relative_paths.append(Path(f).name)

            # Recent operator file (most recent = last in list)
            recent_file = relative_paths[-1] if relative_paths else "—"

            # All operator files (relative paths)
            # all_files_display = "\n".join(relative_paths) if relative_paths else "—"

            table.add_row(
                agent.get("name", "?"),
                agent.get("session_id", "?"),
                agent.get("type", "?"),
                agent.get("tool", "?"),
                recent_file,
                # all_files_display,
            )

        console.print(Panel.fit(table, title="Agent Roster", border_style="cyan"))
        self.logger.debug("Listed %d agents", len(agents_payload))

    def _log_tool_request_panel(
        self, tool_name: str, call_id: str, arguments_str: str
    ) -> None:
        """Display tool request in a panel."""
        try:
            parsed_args = json.loads(arguments_str or "{}")
            syntax = Syntax(
                json.dumps(parsed_args, indent=2, ensure_ascii=False),
                "json",
                theme="monokai",
                word_wrap=True,
            )
        except Exception:
            syntax = arguments_str or "{}"

        console.print(
            Panel(
                syntax,
                title=f"Tool Request · {tool_name}",
                border_style="cyan",
                expand=True,
            )
        )
        self.logger.info(
            "Model requested tool '%s' (call_id=%s) with args=%s",
            tool_name,
            call_id,
            arguments_str,
        )

    def _play_soft_beep(self, frequency: int = 440):
        """Play a soft beep tone (non-blocking)."""
        try:
            duration = 0.12  # seconds
            sample_rate = 24000
            volume = 0.15  # Soft volume

            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beep = (np.sin(frequency * t * 2 * np.pi) * volume * 32767).astype(np.int16)

            # Play using pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True
            )
            stream.write(beep.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            self.logger.debug(f"Beep playback failed: {e}")

    def _on_key_press(self, key):
        """Handle key press to toggle audio pause (shift+space)."""
        try:
            # Track shift key state
            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                self.shift_pressed = True
            # Toggle only when space is pressed while shift is held
            elif key == keyboard.Key.space and self.shift_pressed:
                self.audio_paused = not self.audio_paused
                status = "PAUSED" if self.audio_paused else "LIVE"
                emoji = "⏸️" if self.audio_paused else "🎤"
                color = "yellow" if self.audio_paused else "green"

                # Play beep: higher pitch when resuming, lower when pausing
                beep_freq = 520 if not self.audio_paused else 380
                threading.Thread(
                    target=self._play_soft_beep, args=(beep_freq,), daemon=True
                ).start()

                # Show status panel
                self._log_panel(
                    f"{emoji} {status}",
                    title="Audio Input",
                    style=color,
                    level="info",
                    expand=False,
                )
                self.logger.info(f"Audio input {status}")
        except Exception as e:
            self.logger.error(f"Error handling shift+space: {e}")

    def _on_key_release(self, key):
        """Handle key release to track shift key state."""
        try:
            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                self.shift_pressed = False
        except Exception as e:
            self.logger.error(f"Error handling key release: {e}")

    def _start_keyboard_listener(self):
        """Start keyboard listener for shift+space toggle."""
        if self.input_mode != "audio":
            return

        try:
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press, on_release=self._on_key_release
            )
            self.keyboard_listener.daemon = True
            self.keyboard_listener.start()
            self.logger.info("Keyboard listener started (shift+space to pause/resume)")
        except Exception as e:
            self.logger.warning(f"Could not start keyboard listener: {e}")

    def _calculate_cost_from_usage(self, usage: Dict[str, Any]) -> float:
        """Calculate cost in USD from usage data based on current model."""
        # Official pricing per 1M tokens (text and audio priced separately)
        if "mini" in self.realtime_model.lower():
            # gpt-realtime-mini pricing
            text_input_price = 0.60
            text_output_price = 2.40
            audio_input_price = 10.00
            audio_output_price = 20.00
        else:
            # gpt-realtime (standard) pricing
            text_input_price = 4.00
            text_output_price = 16.00
            audio_input_price = 32.00
            audio_output_price = 64.00

        input_details = usage.get("input_token_details", {})
        output_details = usage.get("output_token_details", {})

        # Extract token counts (skip image tokens as requested)
        input_text_tokens = input_details.get("text_tokens", 0)
        input_audio_tokens = input_details.get("audio_tokens", 0)
        output_text_tokens = output_details.get("text_tokens", 0)
        output_audio_tokens = output_details.get("audio_tokens", 0)

        # Calculate costs separately for text and audio
        text_input_cost = input_text_tokens / 1_000_000 * text_input_price
        text_output_cost = output_text_tokens / 1_000_000 * text_output_price
        audio_input_cost = input_audio_tokens / 1_000_000 * audio_input_price
        audio_output_cost = output_audio_tokens / 1_000_000 * audio_output_price

        return text_input_cost + text_output_cost + audio_input_cost + audio_output_cost

    def _display_token_summary(self):
        """Display token usage and cost summary."""
        tokens = self.cumulative_tokens
        total_cost = self.cumulative_cost_usd

        # Calculate input and output costs separately
        if "mini" in self.realtime_model.lower():
            text_input_price = 0.60
            text_output_price = 2.40
            audio_input_price = 10.00
            audio_output_price = 20.00
        else:
            text_input_price = 4.00
            text_output_price = 16.00
            audio_input_price = 32.00
            audio_output_price = 64.00

        # Input cost breakdown
        input_text_cost = tokens["input_text"] / 1_000_000 * text_input_price
        input_audio_cost = tokens["input_audio"] / 1_000_000 * audio_input_price
        input_cost = input_text_cost + input_audio_cost

        # Output cost breakdown
        output_text_cost = tokens["output_text"] / 1_000_000 * text_output_price
        output_audio_cost = tokens["output_audio"] / 1_000_000 * audio_output_price
        output_cost = output_text_cost + output_audio_cost

        summary = (
            f"Responses: {self.response_count}\n"
            f"Total Tokens: {tokens['total']:,}\n"
            f"├─ Input: {tokens['input']:,} (text: {tokens['input_text']:,}, audio: {tokens['input_audio']:,})\n"
            f"└─ Output: {tokens['output']:,} (text: {tokens['output_text']:,}, audio: {tokens['output_audio']:,})\n"
            f"\n"
            f"Cost Breakdown:\n"
            f"├─ Input: ${input_cost:.4f}\n"
            f"├─ Output: ${output_cost:.4f}\n"
            f"└─ Total: ${total_cost:.4f} USD"
        )

        self._log_panel(
            summary,
            title="Token & Cost Summary",
            style="magenta",
            expand=False,
        )

    # ------------------------------------------------------------------ #
    # System prompt
    # ------------------------------------------------------------------ #

    def load_system_prompt(self) -> str:
        """Load system prompt for the orchestrator."""
        prompt_file = (
            PROMPTS_DIR / "super_agent" / "realtime_super_agent_system_prompt.md"
        )
        try:
            if prompt_file.exists():
                base_prompt = prompt_file.read_text(encoding="utf-8").strip()
                base_prompt = base_prompt.format(
                    AGENT_NAME=REALTIME_ORCH_AGENT_NAME, ENGINEER_NAME=ENGINEER_NAME
                )
            else:
                base_prompt = (
                    "You are a helpful voice assistant with advanced capabilities."
                )
        except Exception as e:
            self.logger.error(f"Error loading prompt file: {e}")
            base_prompt = (
                "You are a helpful voice assistant with advanced capabilities."
            )

        # Append active agent roster
        agents = self.agentic_coder.agent_registry.get("agents", {})
        if agents:
            roster_lines = [
                "\n# Active Agents",
                *[
                    f"- {name} · session {data.get('session_id', 'unknown')}"
                    for name, data in agents.items()
                ],
            ]
            roster_text = '\n'.join(roster_lines)
            base_prompt = f"{base_prompt}\n\n{roster_text}"

        return base_prompt

    # ------------------------------------------------------------------ #
    # Audio setup
    # ------------------------------------------------------------------ #

    def setup_audio(self):
        """Initialize PyAudio for audio input/output."""
        if self.input_mode != "audio" and self.output_mode != "audio":
            return

        self.logger.info("Setting up audio interface...")
        try:
            self.audio_interface = pyaudio.PyAudio()
            self.audio_stream = self.audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
            )
            self.logger.info("Audio interface ready")
        except Exception as e:
            self.logger.error(f"Failed to setup audio: {e}")
            raise

    def cleanup_audio(self):
        """Clean up audio resources."""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()
        self.logger.info("Audio interface cleaned up")

    def base64_encode_audio(self, audio_bytes):
        """Encode audio bytes to base64."""
        return base64.b64encode(audio_bytes).decode("ascii")

    def base64_decode_audio(self, base64_str):
        """Decode base64 audio to bytes."""
        return base64.b64decode(base64_str)

    # ------------------------------------------------------------------ #
    # WebSocket handlers
    # ------------------------------------------------------------------ #

    def on_open(self, ws):
        """WebSocket connection opened."""
        self.logger.info("WebSocket connection established")
        self.running = True

        instructions = self.load_system_prompt()
        output_modalities = self.default_output_modalities

        session_config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.realtime_model,
                "output_modalities": output_modalities,
                "tool_choice": "auto",
                "tools": self.tool_specs,
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "turn_detection": {"type": "semantic_vad"},
                        "transcription": {
                            "model": "gpt-4o-transcribe",
                        },
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "voice": REALTIME_VOICE_CHOICE,
                    },
                },
            },
        }

        self.logger.info("Sending session configuration...")
        ws.send(json.dumps(session_config))

        if self.startup_prompt:
            self._log_panel(
                f"Auto prompt queued: {self.startup_prompt}\nTimeout: {self.auto_timeout}s",
                title="Auto Prompt",
                style="magenta",
            )
            # Record start time for timeout tracking
            self.auto_start_time = time.time()
            threading.Thread(
                target=self._dispatch_text_message,
                args=(self.startup_prompt,),
                daemon=True,
            ).start()
            self.awaiting_auto_close = True
        elif self.input_mode == "text":
            threading.Thread(target=self.text_input_loop, daemon=True).start()
        elif self.input_mode == "audio":
            threading.Thread(target=self.audio_input_loop, daemon=True).start()

    def on_message(self, ws, message):
        """Handle incoming server events."""
        try:
            event = json.loads(message)
            event_type = event.get("type", "unknown")

            # Log ALL events for debugging - CRITICAL EVENTS
            # Debug logging disabled for clean output
            # if event_type in ("response.done", "response.created", "response.output_item.added"):
            #     print(f"[ON_MESSAGE] Received event: {event_type}", flush=True)
            #     self.logger.info(f"[ON_MESSAGE] Received event: {event_type}")

            if event_type == "error":
                self.logger.error(
                    f"ERROR EVENT RECEIVED: {json.dumps(event, indent=2)}"
                )

            # Enhanced conversation logging
            if event_type == "conversation.item.created":
                item = event.get("item", {})
                if item.get("type") == "message":
                    role = item.get("role")
                    text_parts = [
                        part.get("text", "")
                        for part in item.get("content", [])
                        if isinstance(part, dict)
                        and part.get("type") in {"input_text", "output_text"}
                    ]
                    message_text = "\n".join(filter(None, text_parts))
                    if message_text:
                        if role == "user":
                            self._log_panel(
                                message_text, title="User Input", style="blue"
                            )
                        elif role == "assistant":
                            self._log_panel(
                                message_text, title="Assistant", style="green"
                            )

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                if transcript:
                    self._log_panel(
                        transcript, title="User Input (Audio)", style="blue"
                    )

            elif event_type == "input_audio_buffer.speech_stopped":
                # Track when user stopped speaking for latency measurement
                self.speech_stopped_timestamp = time.time()
                self.first_audio_delta_timestamp = None  # Reset for next response
                self.logger.debug("User speech stopped, tracking latency")

            elif event_type == "response.text.delta":
                # Handle streaming text response (Speaches/OpenAI Realtime API)
                delta = event.get("delta", "")
                if delta:
                    # Print delta to console for real-time streaming
                    print(delta, end="", flush=True)
                    self.logger.debug(f"Text delta: {delta}")

            elif event_type == "response.audio_transcript.delta":
                # Handle streaming audio transcript (Speaches/OpenAI Realtime API)
                delta = event.get("delta", "")
                if delta:
                    # Print delta to console for real-time streaming
                    print(delta, end="", flush=True)
                    self.logger.debug(f"Audio transcript delta: {delta}")

            elif event_type == "response.output_text.done":
                final_text = event.get("text", "")
                if final_text:
                    self._log_panel(final_text, title="Assistant", style="green")

            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript", "")
                if transcript:
                    # Calculate latency if we have both timestamps
                    title = "Assistant (Audio)"
                    if (
                        self.speech_stopped_timestamp
                        and self.first_audio_delta_timestamp
                    ):
                        latency = (
                            self.first_audio_delta_timestamp
                            - self.speech_stopped_timestamp
                        )
                        title = f"Assistant (Audio) ({latency:.3f}s)"

                    self._log_panel(transcript, title=title, style="green")

            elif event_type in ("response.output_audio.delta", "response.audio.delta"):
                # Handle audio output from both OpenAI and Speaches formats
                # Track first audio delta for latency measurement
                if self.first_audio_delta_timestamp is None:
                    self.first_audio_delta_timestamp = time.time()
                    if self.speech_stopped_timestamp:
                        latency = (
                            self.first_audio_delta_timestamp
                            - self.speech_stopped_timestamp
                        )
                        self.logger.debug(f"Voice latency: {latency:.3f}s")

                # Auto-pause audio input when agent starts speaking
                if not self.auto_paused_for_response and self.input_mode == "audio":
                    self.auto_paused_for_response = True
                    self.logger.debug("Auto-paused audio input (agent speaking)")

                audio_base64 = event.get("delta", "")
                if audio_base64 and self.output_mode == "audio" and self.audio_stream:
                    audio_bytes = self.base64_decode_audio(audio_base64)
                    self.audio_stream.write(audio_bytes)

            # Handle function calls
            if event_type == "response.function_call_arguments.delta":
                self._handle_function_call_delta(event)
            elif event_type == "response.done":
                # print("[EVENT] Received response.done event", flush=True)
                # self.logger.info("[EVENT] Received response.done event")
                self._handle_response_done(event)

                # Track token usage and cost
                response = event.get("response", {})
                usage = response.get("usage", {})

                if usage:
                    # Increment response count
                    self.response_count += 1

                    # Update cumulative tokens
                    self.cumulative_tokens["total"] += usage.get("total_tokens", 0)
                    self.cumulative_tokens["input"] += usage.get("input_tokens", 0)
                    self.cumulative_tokens["output"] += usage.get("output_tokens", 0)

                    input_details = usage.get("input_token_details", {})
                    output_details = usage.get("output_token_details", {})

                    self.cumulative_tokens["input_text"] += input_details.get(
                        "text_tokens", 0
                    )
                    self.cumulative_tokens["input_audio"] += input_details.get(
                        "audio_tokens", 0
                    )
                    self.cumulative_tokens["output_text"] += output_details.get(
                        "text_tokens", 0
                    )
                    self.cumulative_tokens["output_audio"] += output_details.get(
                        "audio_tokens", 0
                    )

                    # Calculate and accumulate cost
                    response_cost = self._calculate_cost_from_usage(usage)
                    self.cumulative_cost_usd += response_cost

                    # Log token usage every N responses (no UI pollution)
                    if self.response_count % self.token_summary_interval == 0:
                        tokens = self.cumulative_tokens
                        self.logger.info(
                            f"[Token Summary] Responses: {self.response_count}, "
                            f"Total: {tokens['total']:,}, "
                            f"Input: {tokens['input']:,} (text: {tokens['input_text']:,}, audio: {tokens['input_audio']:,}), "
                            f"Output: {tokens['output']:,} (text: {tokens['output_text']:,}, audio: {tokens['output_audio']:,}), "
                            f"Cost: ${self.cumulative_cost_usd:.4f} USD"
                        )

                # Resume audio input after agent finishes speaking
                if self.auto_paused_for_response and self.input_mode == "audio":
                    self.auto_paused_for_response = False
                    self.logger.debug("Resumed audio input (agent done speaking)")

                if self.auto_mode and self.awaiting_auto_close:
                    # Check elapsed time since auto-prompt started
                    elapsed = (
                        time.time() - self.auto_start_time
                        if self.auto_start_time
                        else 0
                    )

                    # Only consider closing if this response has NO function calls
                    response = event.get("response", {})
                    output_items = response.get("output", [])
                    has_function_calls = any(
                        item.get("type") == "function_call" for item in output_items
                    )

                    # Close if: text-only response AND timeout reached
                    if not has_function_calls and elapsed >= self.auto_timeout:
                        self.awaiting_auto_close = False

                        def _close():
                            self._log_panel(
                                f"Auto prompt complete after {elapsed:.1f}s (timeout: {self.auto_timeout}s); closing WebSocket.",
                                title="Auto Prompt",
                                style="magenta",
                            )
                            try:
                                ws.close()
                            except Exception as exc:
                                self._log_panel(
                                    f"Error closing WebSocket: {exc}",
                                    title="WebSocket Error",
                                    style="red",
                                    level="error",
                                )

                        threading.Timer(2.0, _close).start()
                    elif not has_function_calls:
                        # Text-only response but timeout not reached - log progress
                        self.logger.info(
                            f"Auto-prompt: Text response received. Waiting for timeout ({elapsed:.1f}s / {self.auto_timeout}s elapsed)"
                        )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}", exc_info=True)

    def on_error(self, ws, error):
        """WebSocket error handler."""
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed."""
        self.logger.info(
            f"WebSocket connection closed: {close_status_code} - {close_msg}"
        )
        self.running = False

        # Cleanup browser
        if self.browser_agent:
            self.browser_agent.cleanup_browser()

        # Check background threads
        for thread in self.background_threads:
            if thread.is_alive():
                self.logger.debug(f"Background task still running: {thread.name}")

    # ------------------------------------------------------------------ #
    # Input loops
    # ------------------------------------------------------------------ #

    def text_input_loop(self):
        """Handle text input from user."""
        self.logger.info(
            "Text input mode active. Type your messages (or 'quit' to exit):"
        )

        while self.running:
            try:
                user_input = input("\nYou: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    self.logger.info("User requested exit")
                    self.ws.close()
                    break

                if not user_input.strip():
                    continue

                self._dispatch_text_message(user_input)

            except EOFError:
                self.logger.info("EOF received, closing connection")
                break
            except Exception as e:
                self.logger.error(f"Error in text input loop: {e}")
                break

    def audio_input_loop(self):
        """Handle audio input from microphone."""
        self.logger.info(
            "Audio input mode active. Speak into your microphone (press SHIFT+SPACE to pause/resume):"
        )

        while self.running:
            try:
                # Skip audio capture when paused (manual or auto)
                if self.audio_paused or self.auto_paused_for_response:
                    time.sleep(0.1)
                    continue

                audio_data = self.audio_stream.read(
                    CHUNK_SIZE, exception_on_overflow=False
                )
                audio_base64 = self.base64_encode_audio(audio_data)
                event = {"type": "input_audio_buffer.append", "audio": audio_base64}
                self.ws.send(json.dumps(event))

            except Exception as e:
                self.logger.error(f"Error in audio input loop: {e}")
                break

    def _on_agent_task_complete(self, agent_name: str, status: str, message: str):
        """Callback when an agent task completes."""
        self.logger.info(f"Agent task completion: {agent_name} - {status}")

        # Send notification message to user via WebSocket
        if self.ws and self.running:
            self._dispatch_text_message(message)

    def _dispatch_text_message(self, text: str):
        """Send a text message and request a response."""
        if not self.ws:
            self._log_panel(
                "WebSocket unavailable; cannot dispatch text message.",
                title="Dispatch Error",
                style="red",
                level="error",
            )
            return

        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            },
        }

        self.logger.info(f"Sending text message: {text}")
        self.ws.send(json.dumps(event))

        response_event = {
            "type": "response.create",
            "response": {"output_modalities": self.default_output_modalities},
        }
        self.ws.send(json.dumps(response_event))

    # ------------------------------------------------------------------ #
    # Function call handling
    # ------------------------------------------------------------------ #

    def _handle_function_call_delta(self, event: Dict[str, Any]):
        """Handle streaming function call arguments."""
        call_id = event.get("call_id")
        delta = event.get("delta", "")
        if not call_id or not delta:
            return

        self.pending_function_arguments[call_id] = (
            self.pending_function_arguments.get(call_id, "") + delta
        )

    def _handle_response_done(self, event: Dict[str, Any]):
        """Handle completed responses with function calls."""
        # print(f"[RESPONSE_DONE] Called with event", flush=True)
        response = event.get("response", {})
        output_items = response.get("output", [])
        # print(f"[RESPONSE_DONE] Processing response with {len(output_items)} output items", flush=True)
        # self.logger.info(f"[RESPONSE_DONE] Processing response with {len(output_items)} output items")
        if not output_items:
            # print("[RESPONSE_DONE] No output items, returning early", flush=True)
            # self.logger.info("[RESPONSE_DONE] No output items, returning early")
            return

        # Collect all function call IDs from this response
        function_call_ids = []
        # print(f"[RESPONSE_DONE] Iterating through {len(output_items)} output items", flush=True)
        for item in output_items:
            item_type = item.get("type")
            # print(f"[RESPONSE_DONE] Item type: {item_type}", flush=True)
            if item.get("type") != "function_call":
                continue

            call_id = item.get("call_id")
            # print(f"[RESPONSE_DONE] Found function_call with call_id: {call_id}", flush=True)
            # print(f"[RESPONSE_DONE] Already completed: {call_id in self.completed_function_calls}", flush=True)
            if not call_id or call_id in self.completed_function_calls:
                continue

            function_call_ids.append(call_id)

        # Update current response function calls set
        # print(f"[RESPONSE_DONE] Collected {len(function_call_ids)} function call IDs: {function_call_ids}", flush=True)
        if function_call_ids:
            self.current_response_function_calls = set(function_call_ids)
            # print(f"[BATCH] Starting batch of {len(function_call_ids)} function calls: {function_call_ids}", flush=True)
            # self.logger.info(f"[BATCH] Starting batch of {len(function_call_ids)} function calls: {function_call_ids}")

        # Execute all function calls
        for item in output_items:
            if item.get("type") != "function_call":
                continue

            call_id = item.get("call_id")
            if not call_id or call_id in self.completed_function_calls:
                continue

            tool_name = item.get("name") or "unknown"
            arguments_str = item.get(
                "arguments"
            ) or self.pending_function_arguments.pop(call_id, "")
            self._log_tool_request_panel(tool_name, call_id, arguments_str)
            self._execute_tool_call(
                call_id=call_id, tool_name=tool_name, arguments_str=arguments_str
            )

    def _execute_tool_call(
        self, call_id: str, tool_name: Optional[str], arguments_str: str
    ):
        """Execute a tool call and send the result back."""
        if not self.ws:
            self._log_panel(
                "WebSocket connection unavailable; cannot satisfy tool call.",
                title="Tool Error",
                style="red",
                level="error",
            )
            return

        parsed_args: Dict[str, Any] = {}
        if arguments_str:
            try:
                parsed_args = json.loads(arguments_str)
            except json.JSONDecodeError as exc:
                self._log_panel(
                    f"Failed to parse tool arguments: {exc}",
                    title="Tool Error",
                    style="red",
                    level="error",
                )
                payload = json.dumps(
                    {"ok": False, "error": f"Could not parse arguments: {exc}"}
                )
                self._send_function_output(call_id, payload)
                self.completed_function_calls.add(call_id)
                return

        handler_map = {
            "list_agents": self._tool_list_agents,
            "create_agent": self._tool_create_agent,
            "command_agent": self._tool_command_agent,
            "check_agent_result": self._tool_check_agent_result,
            "delete_agent": self._tool_delete_agent,
            "browser_use": self._tool_browser_use,
            "open_file": self._tool_open_file,
            "read_file": self._tool_read_file,
            "report_costs": self._tool_report_costs,
        }

        handler = handler_map.get(tool_name or "")
        if not handler:
            error_msg = f"Tool '{tool_name}' is not implemented on the server."
            self._log_panel(
                error_msg,
                title="Tool Error",
                style="red",
                level="error",
            )
            payload = json.dumps({"ok": False, "error": error_msg})
            self._send_function_output(call_id, payload)
            self.completed_function_calls.add(call_id)
            return

        try:
            result = handler(**parsed_args)
            payload = json.dumps(result)
        except Exception as exc:
            self._log_panel(
                f"Tool '{tool_name}' failed: {exc}",
                title="Tool Error",
                style="red",
                level="error",
            )
            self.logger.exception(f"Tool '{tool_name}' failed")
            payload = json.dumps({"ok": False, "error": f"Tool failed: {exc}"})

        self._send_function_output(call_id, payload)
        self.completed_function_calls.add(call_id)

        # print(f"[EXECUTE_TOOL] Completed tool call: {call_id}", flush=True)
        # print(f"[EXECUTE_TOOL] current_response_function_calls: {self.current_response_function_calls}", flush=True)
        # print(f"[EXECUTE_TOOL] completed_function_calls: {self.completed_function_calls}", flush=True)

        # Check if all function calls from current response are completed
        if self.current_response_function_calls:
            # print(f"[EXECUTE_TOOL] Checking batch completion...", flush=True)
            remaining = self.current_response_function_calls - self.completed_function_calls
            # print(f"[BATCH] Function call {call_id} completed. Remaining in batch: {remaining}", flush=True)
            # self.logger.info(f"[BATCH] Function call {call_id} completed. Remaining in batch: {remaining}")
            # self.logger.info(f"[BATCH] Current batch: {self.current_response_function_calls}")
            # self.logger.info(f"[BATCH] Completed so far: {self.completed_function_calls}")

            if not remaining:
                # print(f"[BATCH] INSIDE if not remaining block - remaining is: {remaining}", flush=True)
                # All function calls completed, send response.create
                # print(f"[BATCH] About to send delayed response.create", flush=True)
                # self.logger.info(f"[BATCH] All function calls in batch completed, sending response.create after brief delay")
                self.current_response_function_calls = set()  # Reset for next batch

                # Add a small delay to ensure server has processed all conversation items
                # This prevents race conditions where response.create arrives before all
                # conversation.item.create events have been fully processed
                def send_response_after_delay():
                    # print(f"[BATCH] send_response_after_delay called, sleeping 50ms", flush=True)
                    time.sleep(0.05)  # 50ms delay
                    # print(f"[BATCH] Woke up from sleep, checking ws={self.ws is not None} running={self.running}", flush=True)
                    try:
                        if self.ws and self.running:
                            response_event = {
                                "type": "response.create",
                                "response": {"output_modalities": self.default_output_modalities},
                            }
                            # print(f"[BATCH] Sending response.create event", flush=True)
                            self.ws.send(json.dumps(response_event))
                            # print(f"[BATCH] Successfully sent response.create", flush=True)
                            # self.logger.info("[BATCH] Sent response.create after batch completion (delayed)")
                    except Exception as e:
                        # print(f"[BATCH] EXCEPTION in send_response_after_delay: {e}", flush=True)
                        self.logger.error(f"[BATCH] Failed to send delayed response.create: {e}")

                # Send in a background thread to avoid blocking
                # print(f"[BATCH] Starting background thread for delayed response.create", flush=True)
                threading.Thread(target=send_response_after_delay, daemon=True).start()
                # print(f"[BATCH] Background thread started", flush=True)
        else:
            # Fallback for single function call (no batch tracking)
            self.logger.info(f"[FALLBACK] Sending response.create immediately (current_response_function_calls is empty)")
            response_event = {
                "type": "response.create",
                "response": {"output_modalities": self.default_output_modalities},
            }
            self.ws.send(json.dumps(response_event))

    def _send_function_output(self, call_id: str, output_payload: str):
        """Send function output back to the model."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_payload,
            },
        }
        self.ws.send(json.dumps(event))
        self.logger.debug(f"Emitted function_call_output for call_id={call_id}")

    # ------------------------------------------------------------------ #
    # Tool specifications
    # ------------------------------------------------------------------ #

    def _build_tool_specs(self) -> list[Dict[str, Any]]:
        """Build tool specifications for OpenAI Realtime API."""
        return [
            {
                "type": "function",
                "name": "list_agents",
                "description": "List all registered agents (both coding and browser automation) with session details and operator files.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "create_agent",
                "description": (
                    "Create and register a new agent. Two tool/type combinations available:\n"
                    f"1. tool='{CLAUDE_CODE_TOOL}' + type='{AGENTIC_CODING_TYPE}' for software development tasks\n"
                    f"2. tool='{GEMINI_TOOL}' + type='{AGENTIC_BROWSERING_TYPE}' for browser automation tasks"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": [CLAUDE_CODE_TOOL, GEMINI_TOOL],
                            "description": f"Tool to use: '{CLAUDE_CODE_TOOL}' for coding agents, '{GEMINI_TOOL}' for browser automation",
                            "default": CLAUDE_CODE_TOOL,
                        },
                        "type": {
                            "type": "string",
                            "enum": [AGENTIC_CODING_TYPE, AGENTIC_BROWSERING_TYPE],
                            "description": f"Agent type: '{AGENTIC_CODING_TYPE}' for software development, '{AGENTIC_BROWSERING_TYPE}' for browser automation",
                            "default": AGENTIC_CODING_TYPE,
                        },
                        "agent_name": {
                            "type": "string",
                            "description": (
                                "Optional explicit codename for the agent. "
                                "If omitted, a unique name is generated."
                            ),
                        },
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "command_agent",
                "description": (
                    "Dispatch an asynchronous task to an existing agent (coding or browser automation). "
                    "For coding agents, returns the operator log path. For browser agents, executes the task directly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Codename of the agent to command.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Task prompt to send to the agent.",
                        },
                    },
                    "required": ["agent_name", "prompt"],
                },
            },
            {
                "type": "function",
                "name": "delete_agent",
                "description": (
                    "Remove a registered agent (coding or browser) and delete its working directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Codename of the agent to delete.",
                        },
                    },
                    "required": ["agent_name"],
                },
            },
            {
                "type": "function",
                "name": "check_agent_result",
                "description": (
                    "Read the operator status report for a given agent and file name."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Codename of the agent whose log to read.",
                        },
                        "operator_file_name": {
                            "type": "string",
                            "description": "Filename (including .md) inside the agent's directory.",
                        },
                    },
                    "required": ["agent_name", "operator_file_name"],
                },
            },
            {
                "type": "function",
                "name": "browser_use",
                "description": (
                    "Automate web browsing tasks using advanced AI. "
                    "Can navigate websites, search for information, interact with web pages, "
                    "extract data, and perform complex multi-step browsing tasks. "
                    "Provide a clear task description and optionally a starting URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": (
                                "A clear description of the browsing task to perform. "
                                "Be specific about what information to gather or actions to take."
                            ),
                        },
                        "url": {
                            "type": "string",
                            "description": (
                                "Optional starting URL. If not provided, will start from "
                                "a search engine or appropriate starting point."
                            ),
                        },
                    },
                    "required": ["task"],
                },
            },
            {
                "type": "function",
                "name": "open_file",
                "description": (
                    "Open a file in VS Code or the default system application. "
                    "Uses 'code' command for text/code files and 'open' command for media files (audio/video). "
                    "File path is automatically prefixed with the agent working directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Relative path to the file (e.g., 'frontend/src/App.vue' or 'backend/main.py'). "
                                "Will be prefixed with the working directory automatically."
                            ),
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "type": "function",
                "name": "read_file",
                "description": (
                    "Read and return the contents of a file from the working directory. "
                    "Useful for reviewing code, checking configurations, or gathering context before directing agents. "
                    "File path is automatically prefixed with the agent working directory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Relative path to the file (e.g., 'backend/main.py' or 'specs/plan.md'). "
                                "Will be prefixed with the working directory automatically."
                            ),
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "type": "function",
                "name": "report_costs",
                "description": (
                    "Display current token usage and cost summary for this session. "
                    "Shows cumulative token counts (text/audio breakdown) and total cost in USD. "
                    "Use when the user asks about costs, usage, or spending."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

    # ------------------------------------------------------------------ #
    # Tool implementations
    # ------------------------------------------------------------------ #

    def _tool_list_agents(self) -> Dict[str, Any]:
        """List all registered agents from both registries."""
        # Get Claude Code agents
        claude_result = self.agentic_coder.list_agents()
        claude_agents = claude_result.get("agents", [])

        # Get Gemini browser agents
        browser_agents_list = []
        for name, data in sorted(
            self.browser_agent.agent_registry.get("agents", {}).items()
        ):
            browser_agents_list.append(
                {
                    "name": name,
                    "session_id": data.get("session_id"),
                    "tool": data.get("tool"),
                    "type": data.get("type"),
                    "created_at": data.get("created_at"),
                }
            )

        # Combine both lists
        all_agents = claude_agents + browser_agents_list
        self._log_agent_roster_panel(all_agents)
        return {"ok": True, "agents": all_agents}

    def _create_browser_agent(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Create and register a browser agent."""
        # Generate unique name if not provided
        browser_name = (
            agent_name
            or f"BrowserAgent_{datetime.now(timezone.utc).strftime('%H%M%S')}"
        )

        # Check if name already exists
        if self.browser_agent._get_agent_by_name(browser_name):
            return {
                "ok": False,
                "error": f"Browser agent '{browser_name}' already exists. Choose a different name.",
            }

        # Register in browser registry
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.browser_agent.session_id,
        }
        self.browser_agent._register_agent(browser_name, metadata)

        return {
            "ok": True,
            "agent_name": browser_name,
            "session_id": self.browser_agent.session_id,
            "type": AGENTIC_BROWSERING_TYPE,
        }

    def _tool_create_agent(
        self,
        tool: str = CLAUDE_CODE_TOOL,
        type: str = AGENTIC_CODING_TYPE,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new agent - routes to appropriate handler based on tool/type."""
        # Route to browser agent handler
        if tool == GEMINI_TOOL and type == AGENTIC_BROWSERING_TYPE:
            return self._create_browser_agent(agent_name)

        # Route to Claude Code agent handler
        elif tool == CLAUDE_CODE_TOOL and type == AGENTIC_CODING_TYPE:
            return self.agentic_coder.create_agent(
                tool=tool, agent_type=type, agent_name=agent_name
            )

        # Invalid combination
        else:
            return {
                "ok": False,
                "error": f"Invalid tool/type combination: tool='{tool}', type='{type}'. Valid combinations: ('{CLAUDE_CODE_TOOL}', '{AGENTIC_CODING_TYPE}') or ('{GEMINI_TOOL}', '{AGENTIC_BROWSERING_TYPE}')",
            }

    def _tool_command_agent(self, agent_name: str, prompt: str) -> Dict[str, Any]:
        """Command an agent - routes to appropriate handler based on agent type."""
        # Check both registries to find the agent
        claude_agent = self.agentic_coder._get_agent_by_name(agent_name)
        browser_agent = self.browser_agent._get_agent_by_name(agent_name)

        # Route to Claude Code agent handler
        if claude_agent:
            return self.agentic_coder.command_agent(
                agent_name=agent_name, prompt=prompt
            )

        # Route to browser agent handler
        elif browser_agent:
            try:
                result = self.browser_agent.execute_task(task=prompt)
                return result
            except Exception as exc:
                self.logger.exception("Browser agent command failed")
                return {"ok": False, "error": f"Browser task failed: {exc}"}

        # Agent not found in either registry
        else:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' not found in either registry. Create it first.",
            }

    def _tool_check_agent_result(
        self, agent_name: str, operator_file_name: str
    ) -> Dict[str, Any]:
        """Check agent result."""
        return self.agentic_coder.check_agent_result(
            agent_name=agent_name, operator_file_name=operator_file_name
        )

    def _tool_delete_agent(self, agent_name: str) -> Dict[str, Any]:
        """Delete an agent - routes to appropriate handler based on agent type."""
        # Check both registries to find the agent
        claude_agent = self.agentic_coder._get_agent_by_name(agent_name)
        browser_agent_data = self.browser_agent._get_agent_by_name(agent_name)

        # Route to Claude Code agent handler
        if claude_agent:
            return self.agentic_coder.delete_agent(agent_name=agent_name)

        # Route to browser agent handler
        elif browser_agent_data:
            warnings: list[str] = []

            with self.browser_agent.registry_lock:
                self.browser_agent.agent_registry.get("agents", {}).pop(
                    agent_name, None
                )
                try:
                    self.browser_agent._save_agent_registry()
                except Exception as exc:
                    warnings.append(f"Failed to update registry: {exc}")

            agent_dir = self.browser_agent._agent_directory(agent_name)
            if agent_dir.exists():
                try:
                    shutil.rmtree(agent_dir)
                except Exception as exc:
                    warnings.append(f"Failed to remove directory {agent_dir}: {exc}")

            payload: Dict[str, Any] = {"ok": True, "agent_name": agent_name}
            if warnings:
                payload["warnings"] = warnings
            return payload

        # Agent not found in either registry
        else:
            return {
                "ok": False,
                "error": f"Agent '{agent_name}' not found in either registry. Nothing to delete.",
            }

    def _tool_browser_use(self, task: str, url: Optional[str] = None) -> Dict[str, Any]:
        """Execute browser automation task."""
        try:
            self._log_panel(
                f"Task: {task}\nStarting URL: {url or 'Auto-detect'}",
                title="Browser Use - Starting Task",
                style="cyan",
            )

            result = self.browser_agent.execute_task(task, url)

            if result.get("ok"):
                self._log_panel(
                    f"Task completed!\n\nResult:\n{result.get('data')}\n\nScreenshots: {result.get('screenshot_dir')}",
                    title="Browser Use - Complete",
                    style="green",
                )
            else:
                self._log_panel(
                    f"Browser automation failed: {result.get('error')}",
                    title="Browser Use - Error",
                    style="red",
                    level="error",
                )

            return result

        except Exception as exc:
            self._log_panel(
                f"Browser automation failed: {exc}",
                title="Browser Use - Error",
                style="red",
                level="error",
            )
            self.logger.exception("Browser automation failed")
            return {"ok": False, "error": str(exc)}

    def _tool_open_file(self, file_path: str) -> Dict[str, Any]:
        """Open a file in VS Code or default system application."""
        try:
            # Prefix with working directory
            full_path = AGENT_WORKING_DIRECTORY / file_path

            # Check if file exists
            if not full_path.exists():
                return {"ok": False, "error": f"File not found: {file_path}"}

            # Determine if it's a media file (audio/video)
            media_extensions = {
                ".mp3",
                ".mp4",
                ".wav",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".mov",
                ".avi",
                ".mkv",
                ".webm",
                ".wmv",
                ".flv",
                ".m4v",
            }

            file_ext = full_path.suffix.lower()
            is_media = file_ext in media_extensions

            # Choose command based on file type
            if is_media:
                command = f'open "{full_path}"'
                app_name = "default system application"
            else:
                command = f'code "{full_path}"'
                app_name = "VS Code"

            self._log_panel(
                f"Opening: {file_path}\nFull path: {full_path}\nUsing: {app_name}",
                title="Open File",
                style="cyan",
            )

            # Execute command
            import subprocess

            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                self._log_panel(
                    f"Successfully opened {file_path} in {app_name}",
                    title="Open File - Success",
                    style="green",
                )
                return {
                    "ok": True,
                    "file_path": str(full_path),
                    "opened_with": app_name,
                }
            else:
                error_msg = result.stderr or "Unknown error"
                self._log_panel(
                    f"Failed to open file: {error_msg}",
                    title="Open File - Error",
                    style="red",
                    level="error",
                )
                return {"ok": False, "error": error_msg}

        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Command timed out after 5 seconds"}
        except Exception as exc:
            self._log_panel(
                f"Error opening file: {exc}",
                title="Open File - Error",
                style="red",
                level="error",
            )
            self.logger.exception("Failed to open file")
            return {"ok": False, "error": str(exc)}

    def _tool_read_file(self, file_path: str) -> Dict[str, Any]:
        """Read and return the contents of a file."""
        try:
            # Prefix with working directory
            full_path = AGENT_WORKING_DIRECTORY / file_path

            # Check if file exists
            if not full_path.exists():
                return {"ok": False, "error": f"File not found: {file_path}"}

            # Check if it's a directory
            if full_path.is_dir():
                return {
                    "ok": False,
                    "error": f"Path is a directory, not a file: {file_path}",
                }

            # Read file contents
            try:
                content = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try reading as binary for non-text files
                return {"ok": False, "error": f"File is not a text file: {file_path}"}

            self._log_panel(
                f"File: {file_path}\nSize: {len(content)} characters\nLines: {len(content.splitlines())}",
                title="Read File",
                style="cyan",
            )

            return {
                "ok": True,
                "file_path": str(full_path),
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines()),
            }

        except Exception as exc:
            self._log_panel(
                f"Error reading file: {exc}",
                title="Read File - Error",
                style="red",
                level="error",
            )
            self.logger.exception("Failed to read file")
            return {"ok": False, "error": str(exc)}

    def _tool_report_costs(self) -> Dict[str, Any]:
        """Display token usage and cost summary."""
        try:
            # Display the summary panel
            self._display_token_summary()

            # Return data for the tool result
            return {
                "ok": True,
                "responses": self.response_count,
                "total_tokens": self.cumulative_tokens["total"],
                "input_tokens": self.cumulative_tokens["input"],
                "output_tokens": self.cumulative_tokens["output"],
                "cost_usd": self.cumulative_cost_usd,
            }
        except Exception as exc:
            self._log_panel(
                f"Error generating cost report: {exc}",
                title="Report Costs - Error",
                style="red",
                level="error",
            )
            self.logger.exception("Failed to generate cost report")
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self):
        """Connect to OpenAI Realtime API."""
        self.logger.info("Connecting to OpenAI Realtime API...")

        # Setup audio if needed
        if self.input_mode == "audio" or self.output_mode == "audio":
            self.setup_audio()

        # Start keyboard listener for audio pause control
        self._start_keyboard_listener()

        # Create WebSocket connection
        websocket_url = REALTIME_API_URL_TEMPLATE.format(model=self.realtime_model)

        # Only add Authorization header if connecting to OpenAI's API
        headers = []
        if "api.openai.com" in websocket_url and OPENAI_API_KEY:
            headers = [f"Authorization: Bearer {OPENAI_API_KEY}"]

        self.ws = websocket.WebSocketApp(
            websocket_url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        try:
            # Run WebSocket connection (blocking)
            self.ws.run_forever()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            # Stop keyboard listener
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None

            if self.input_mode == "audio" or self.output_mode == "audio":
                self.cleanup_audio()
            self.logger.info("Connection closed")


# ================================================================
# Setup logging
# ================================================================


def setup_logging():
    """Setup logging to file only (no stdout)."""
    logger = logging.getLogger("BigThreeAgents")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    logger.propagate = False

    now = datetime.now()
    log_dir = Path(__file__).parent / "output_logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"{now.strftime('%Y-%m-%d_%H')}.log"
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


# ================================================================
# Main entry point
# ================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Big Three Realtime Agents - Unified agent system with voice, coding, and browser automation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        choices=["text", "audio"],
        default="text",
        help="Input mode: 'text' for typing, 'audio' for microphone",
    )
    parser.add_argument(
        "--output",
        choices=["text", "audio"],
        default="text",
        help="Output mode: 'text' for text responses, 'audio' for voice responses",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice mode (sets both input and output to audio)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional text prompt to auto-dispatch (forces text input/output).",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help=f"Use the mini realtime model ({REALTIME_MODEL_MINI}) instead of the default",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Auto-prompt mode timeout in seconds (default: 300). Keeps session alive for background agents to complete work.",
    )

    args = parser.parse_args()

    startup_prompt = None
    input_mode = args.input
    output_mode = args.output
    realtime_model = REALTIME_MODEL_MINI if args.mini else REALTIME_MODEL_DEFAULT

    # Voice flag overrides input/output settings
    if args.voice:
        input_mode = "audio"
        output_mode = "audio"

    # Prompt flag forces text mode (overrides --voice if both are set)
    if args.prompt:
        startup_prompt = args.prompt
        input_mode = "text"
        output_mode = "text"

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Big Three Realtime Agents")
    logger.info("=" * 60)
    logger.info(f"Input: {input_mode}, Output: {output_mode}")
    logger.info(f"Realtime model: {realtime_model}")
    logger.info(f"Gemini model: {GEMINI_MODEL}")
    logger.info(f"Claude model: {DEFAULT_CLAUDE_MODEL}")
    logger.info(f"Agent working directory: {AGENT_WORKING_DIRECTORY}")
    if startup_prompt:
        logger.info(f"Auto prompt enabled: {startup_prompt}")

    config_message = (
        f"Input: {input_mode}\n"
        f"Output: {output_mode}\n"
        f"Realtime model: {realtime_model}\n"
        f"Gemini model: {GEMINI_MODEL}\n"
        f"Claude model: {DEFAULT_CLAUDE_MODEL}\n"
        f"Working dir: {AGENT_WORKING_DIRECTORY}"
    )
    console.print(
        Panel(config_message, title="Launch Configuration", border_style="cyan")
    )
    if startup_prompt:
        console.print(
            Panel(
                f"Auto prompt enabled:\n{startup_prompt}",
                title="Auto Prompt",
                border_style="magenta",
            )
        )

    try:
        agent = OpenAIRealtimeVoiceAgent(
            input_mode=input_mode,
            output_mode=output_mode,
            logger=logger,
            startup_prompt=startup_prompt,
            realtime_model=realtime_model,
            auto_timeout=args.timeout,
        )
        agent.connect()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        return 1

    logger.info("Agent terminated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
