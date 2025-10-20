"""
Text-based HTTP API server for big-3-super-agent orchestration.

This server provides a text-in/text-out API that uses the same multi-agent
orchestration as the Realtime API version, but works with traditional
STT/TTS pipelines (like Open-LLM-VTuber).

Uses OpenAI Chat Completions API with function calling for tool dispatch.
"""

# /// script
# dependencies = [
#     "fastapi>=0.118.2",
#     "uvicorn[standard]>=0.37.0",
#     "pydantic>=2.0.0",
#     "python-dotenv",
#     "websocket-client",
#     "pyaudio",
#     "rich",
#     "openai",
#     "claude-agent-sdk",
#     "google-genai",
#     "playwright",
#     "numpy",
#     "pynput",
# ]
# ///


import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI

# Import agents from big_three_realtime_agents
from big_three_realtime_agents import (
    GeminiBrowserAgent,
    AgentZeroAgent,
    ClaudeCodeAgenticCoder,
    OPENAI_API_KEY,
    REALTIME_API_URL,
    REALTIME_MODEL_DEFAULT,
    AGENT_WORKING_DIRECTORY,
    setup_logging,
)


# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agents_used: List[str] = []
    tool_calls: List[str] = []


# Session management
class SessionManager:
    """Manages conversation sessions with context history."""

    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.logger = logging.getLogger("SessionManager")

    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, List[Dict[str, str]]]:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return session_id, self.sessions[session_id]

        # Create new session
        new_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.sessions)}"
        self.sessions[new_id] = []
        self.logger.info(f"Created new session: {new_id}")
        return new_id, self.sessions[new_id]

    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get session history."""
        return self.sessions.get(session_id, [])


class Big3Orchestrator:
    """
    Text-based orchestrator for big-3-super-agent.

    Uses OpenAI Chat Completions API with function calling to dispatch
    tasks to Claude Code, Gemini Browser, and Agent Zero agents.
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("Big3Orchestrator")

        # Configure OpenAI client for self-hosted endpoint
        # Convert WebSocket URL (ws://...:8000/v1/realtime) to HTTP (http://...:8000/v1)
        if REALTIME_API_URL:
            base_url = REALTIME_API_URL.replace("ws://", "http://").replace("wss://", "https://")
            # Remove /realtime suffix if present
            if base_url.endswith("/realtime"):
                base_url = base_url[:-9]  # Remove "/realtime"
            self.logger.info(f"Using self-hosted OpenAI-compatible endpoint: {base_url}")
        else:
            base_url = "https://api.openai.com/v1"
            self.logger.info("Using OpenAI cloud API")

        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY or "dummy-key",
            base_url=base_url
        )
        self.model = REALTIME_MODEL_DEFAULT or "gpt-4o"
        self.logger.info(f"Using model: {self.model}")

        # Initialize sub-agents
        self.browser_agent = GeminiBrowserAgent(logger=self.logger)
        self.agent_zero = None
        try:
            self.agent_zero = AgentZeroAgent(logger=self.logger)
        except ValueError as e:
            self.logger.warning(f"Agent Zero not available: {e}")

        self.agentic_coder = ClaudeCodeAgenticCoder(
            logger=self.logger,
            browser_agent=self.browser_agent,
        )

        # Build tool specifications
        self.tools = self._build_tool_specs()

        # System prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt for the orchestrator."""
        return f"""You are an advanced AI assistant orchestrating multiple specialized agents to help with complex tasks.

You have access to three powerful agent types:
1. **Claude Code Agent** - Expert software developer for writing, debugging, and analyzing code
2. **Gemini Browser Agent** - Web automation specialist for browsing, data extraction, and web interactions
3. **Agent Zero** - General-purpose AI for miscellaneous tasks

Current working directory: {AGENT_WORKING_DIRECTORY}

## CRITICAL EXECUTION RULES:

1. **IMMEDIATE AND COMPLETE EXECUTION**:
   - Execute ALL required tools in a SINGLE response
   - Do NOT say "I will now...", "let me...", "one moment", or "next I will..."
   - Do NOT wait for user confirmation between steps
   - If a task requires multiple tools (e.g., create agent + command agent), call BOTH tools immediately

2. **Multi-Step Tasks**:
   - Example: "create an agent and tell it to turn off the light"
   - CORRECT: Call create_agent, then IMMEDIATELY call command_agent in the SAME response
   - WRONG: Call create_agent, say "I will now command it", wait for user input

3. **Workflow**:
   - Identify ALL tools needed for the request
   - Execute ALL tools immediately in your response
   - After ALL tools complete, briefly report what was done
   - NEVER split multi-step tasks across multiple responses

4. **Tool Usage**:
   - Use `list_agents` to check existing agents before creating duplicates
   - Call `create_agent` when you need a new agent
   - Use `command_agent` to delegate tasks to agents
   - Call `delete_agent` to remove agents when explicitly requested
   - Use `browser_use` for web automation tasks

Be proactive, efficient, and collaborative. Break down complex tasks and leverage each agent's strengths.

* ALWAYS use emotion tags in your responses: [joy] [sadness] [anger] [surprise] [fear] [disgust] [smirk] [neutral]
* NO emojis - use emotion tags instead
* NEVER mention or apologize about emotion tags - just use them silently
* Don't say "I'll use emotion tags" or "Thanks for the reminder" - the user knows you use them.

Examples:

User: "how are you?" Good: [joy] I'm doing great! [neutral] How about you? Bad: ❌ 😊 I'm doing great! (don't use emojis)."""

    def _build_tool_specs(self) -> List[Dict[str, Any]]:
        """Build OpenAI function calling tool specifications."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_agent",
                    "description": "Create a new specialized agent (Claude Code for coding, Gemini Browser for web tasks, Agent Zero for general tasks)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "enum": ["claude-code", "gemini-browser", "agent-zero"],
                                "description": "The tool/agent type to create"
                            },
                            "agent_name": {
                                "type": "string",
                                "description": "Descriptive name for this agent instance"
                            },
                            "lifetime_hours": {
                                "type": "number",
                                "description": "How long the agent should persist (hours)",
                                "default": 24
                            }
                        },
                        "required": ["tool", "agent_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "command_agent",
                    "description": "Send a task/command to a specific agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": "Name of the agent to command"
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The task or instruction to give the agent"
                            }
                        },
                        "required": ["agent_name", "prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "description": "List all active agents across all types",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_agent",
                    "description": "Delete/remove an agent by name and clean up its resources",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": "Name of the agent to delete (e.g., 'MyAgent' or 'agent-zero:MyAgent'). Use the exact name from list_agents."
                            }
                        },
                        "required": ["agent_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_use",
                    "description": "Direct browser automation task (quick one-off web tasks)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The browser task to perform"
                            },
                            "url": {
                                "type": "string",
                                "description": "Starting URL for the task"
                            }
                        },
                        "required": ["task"]
                    }
                }
            }
        ]

    def process_message(self, message: str, history: List[Dict[str, str]]) -> tuple[str, List[str], List[str]]:
        """
        Process a message and return response.

        Returns:
            tuple: (response_text, agents_used, tool_calls)
        """
        # Build messages array with history
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        agents_used = []
        tool_calls_made = []

        print(f"\n=== PROCESSING MESSAGE ===")
        print(f"User: {message}")
        print(f"History length: {len(history)}")

        try:
            # Allow multiple rounds of tool calling (max 5 rounds)
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")

                # Make API call
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                print(f"Tool calls: {len(tool_calls) if tool_calls else 0}")

                # If no tool calls, we're done
                if not tool_calls:
                    final_text = response_message.content
                    print(f"Final response (no tools): {final_text}")
                    break

                # Add assistant's response to messages
                messages.append(response_message)

                # Process all tool calls
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    tool_calls_made.append(function_name)

                    import json
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"Tool call: {function_name}")
                    print(f"Args: {function_args}")
                    self.logger.info(f"Tool call: {function_name} with args: {function_args}")

                    # Execute the function
                    function_result = self._execute_tool(function_name, function_args, agents_used)

                    # Add function result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(function_result)
                    })

                # Continue loop to allow AI to make more tool calls or finish
            else:
                # Hit max iterations
                final_text = "Task completed (reached maximum tool calling iterations)."

            return final_text, agents_used, tool_calls_made

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
            return f"Error: {str(e)}", agents_used, tool_calls_made

    def _execute_tool(self, function_name: str, function_args: Dict[str, Any], agents_used: List[str]) -> str:
        """Execute a tool call and return the result."""
        try:
            if function_name == "create_agent":
                tool = function_args["tool"]
                agent_name = function_args["agent_name"]
                agents_used.append(f"{tool}:{agent_name}")

                if tool == "claude-code":
                    result = self.agentic_coder.create_agent(agent_name=agent_name)
                    if result.get("ok"):
                        return f"Created Claude Code agent '{agent_name}'"
                    else:
                        return f"Error creating Claude Code agent: {result.get('error')}"

                elif tool == "gemini-browser":
                    # Create browser agent by registering it
                    if self.browser_agent._get_agent_by_name(agent_name):
                        return f"Error: Browser agent '{agent_name}' already exists"

                    metadata = {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "session_id": self.browser_agent.session_id,
                    }
                    self.browser_agent._register_agent(agent_name, metadata)
                    return f"Created Gemini Browser agent '{agent_name}'"

                elif tool == "agent-zero" and self.agent_zero:
                    result = self.agent_zero.create_agent(agent_name=agent_name)
                    if result.get("ok"):
                        return f"Created Agent Zero '{agent_name}'"
                    else:
                        return f"Error creating Agent Zero: {result.get('error')}"
                else:
                    return f"Error: Agent type '{tool}' not available"

            elif function_name == "command_agent":
                agent_name = function_args["agent_name"]
                prompt = function_args["prompt"]
                agents_used.append(agent_name)

                # Strip agent type prefix if present (e.g., "claude-code:MyAgent" -> "MyAgent")
                clean_agent_name = agent_name
                if ":" in agent_name:
                    clean_agent_name = agent_name.split(":", 1)[1]

                # Try each agent type
                # Check Claude Code
                if clean_agent_name in self.agentic_coder.agent_registry.get("agents", {}):
                    result = self.agentic_coder.command_agent(clean_agent_name, prompt)
                    if result.get("ok"):
                        operator_file = result.get("operator_file_name", "unknown")
                        return f"Claude Code agent '{clean_agent_name}' is working on the task. Check operator file: {operator_file}"
                    else:
                        return f"Error commanding Claude Code agent '{clean_agent_name}': {result.get('error')}"

                # Check Gemini Browser
                elif clean_agent_name in self.browser_agent.agent_registry.get("agents", {}):
                    result = self.browser_agent.execute_task(task=prompt)
                    if result.get("ok"):
                        return f"Gemini Browser agent '{clean_agent_name}' executed task successfully"
                    else:
                        return f"Gemini Browser agent '{clean_agent_name}' error: {result.get('error')}"

                # Check Agent Zero
                elif self.agent_zero and clean_agent_name in self.agent_zero.agent_registry.get("agents", {}):
                    self.logger.info(f"Sending message to Agent Zero '{clean_agent_name}': {prompt}")
                    result = self.agent_zero.send_message(clean_agent_name, prompt)
                    self.logger.info(f"Agent Zero response: {result}")
                    if result.get("ok"):
                        response_text = result.get("response", "Task sent successfully")
                        return f"Agent Zero '{clean_agent_name}': {response_text}"
                    else:
                        return f"Error commanding Agent Zero '{clean_agent_name}': {result.get('error')}"

                else:
                    return f"Error: Agent '{clean_agent_name}' not found"

            elif function_name == "list_agents":
                all_agents = []

                # Claude Code agents
                for name in self.agentic_coder.agent_registry.get("agents", {}).keys():
                    all_agents.append(f"claude-code:{name}")

                # Browser agents
                for name in self.browser_agent.agent_registry.get("agents", {}).keys():
                    all_agents.append(f"gemini-browser:{name}")

                # Agent Zero agents
                if self.agent_zero:
                    for name in self.agent_zero.agent_registry.get("agents", {}).keys():
                        all_agents.append(f"agent-zero:{name}")

                return f"Active agents: {', '.join(all_agents) if all_agents else 'None'}"

            elif function_name == "delete_agent":
                agent_name = function_args["agent_name"]

                # Strip agent type prefix if present (e.g., "agent-zero:MyAgent" -> "MyAgent")
                if ":" in agent_name:
                    agent_name = agent_name.split(":", 1)[1]

                # Try to find and delete agent from each registry
                # Check Claude Code
                if agent_name in self.agentic_coder.agent_registry.get("agents", {}):
                    result = self.agentic_coder.delete_agent(agent_name)
                    if result.get("ok"):
                        return f"Claude Code agent '{agent_name}' deleted successfully"
                    else:
                        return f"Error deleting Claude Code agent '{agent_name}': {result.get('error')}"

                # Check Gemini Browser
                elif agent_name in self.browser_agent.agent_registry.get("agents", {}):
                    result = self.browser_agent.delete_agent(agent_name)
                    if result.get("ok"):
                        return f"Gemini Browser agent '{agent_name}' deleted successfully"
                    else:
                        return f"Error deleting Gemini Browser agent '{agent_name}': {result.get('error')}"

                # Check Agent Zero
                elif self.agent_zero and agent_name in self.agent_zero.agent_registry.get("agents", {}):
                    result = self.agent_zero.delete_agent(agent_name)
                    if result.get("ok"):
                        return f"Agent Zero '{agent_name}' deleted successfully"
                    else:
                        return f"Error deleting Agent Zero '{agent_name}': {result.get('error')}"

                else:
                    # List available agents to help debug
                    available = []
                    for name in self.agentic_coder.agent_registry.get("agents", {}).keys():
                        available.append(f"claude-code:{name}")
                    for name in self.browser_agent.agent_registry.get("agents", {}).keys():
                        available.append(f"gemini-browser:{name}")
                    if self.agent_zero:
                        for name in self.agent_zero.agent_registry.get("agents", {}).keys():
                            available.append(f"agent-zero:{name}")

                    available_str = ", ".join(available) if available else "None"
                    return f"Error: Agent '{agent_name}' not found. Available agents: {available_str}"

            elif function_name == "browser_use":
                task = function_args["task"]
                url = function_args.get("url", "")
                agents_used.append("browser-direct")

                result = self.browser_agent.quick_task(task, url)
                return f"Browser task completed: {result}"

            else:
                return f"Error: Unknown function '{function_name}'"

        except Exception as e:
            self.logger.error(f"Error executing tool {function_name}: {e}", exc_info=True)
            return f"Error executing {function_name}: {str(e)}"


# FastAPI app
app = FastAPI(
    title="Big-3 Super Agent Text API",
    description="Text-based API for multi-agent orchestration with Claude Code, Gemini Browser, and Agent Zero",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
logger = setup_logging()
session_manager = SessionManager()
orchestrator = Big3Orchestrator(logger=logger)


@app.get("/")
async def root():
    """API info endpoint."""
    return {
        "name": "Big-3 Super Agent Text API",
        "version": "1.0.0",
        "status": "running",
        "working_directory": AGENT_WORKING_DIRECTORY
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message with multi-agent orchestration.

    This endpoint accepts text input and returns text output,
    using the big-3 orchestration system to dispatch tasks to
    specialized agents as needed.
    """
    try:
        # Get or create session
        session_id, history = session_manager.get_or_create_session(request.session_id)

        logger.info(f"Processing message for session {session_id}: {request.message[:100]}...")

        # Use provided context or session history
        context = request.context if request.context else [{"role": msg["role"], "content": msg["content"]} for msg in history]

        # Process the message
        response_text, agents_used, tool_calls = orchestrator.process_message(
            request.message,
            context
        )

        # Update session history
        session_manager.add_message(session_id, "user", request.message)
        session_manager.add_message(session_id, "assistant", response_text)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            agents_used=agents_used,
            tool_calls=tool_calls
        )

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history."""
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


def main():
    """Main entry point for the text API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Big-3 Super Agent Text API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Big-3 Super Agent Text API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Working Directory: {AGENT_WORKING_DIRECTORY}")
    logger.info("=" * 60)

    uvicorn.run(
        "text_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
