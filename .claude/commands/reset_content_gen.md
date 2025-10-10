---
model: claude-sonnet-4-5-20250929
description: Reset apps/content-gen to fresh state for new voice-based agentic coding experiments
---

# Purpose

Reset the apps/content-gen application to a clean starter state with fresh backend (FastAPI), frontend (Vue + TypeScript), and cleared experimental artifacts, preparing for the next round of voice-based agentic coding experiments.

## Codebase Structure

```
apps/content-gen/
├── backend/          # Python FastAPI backend (will be reset)
├── frontend/         # Vue TypeScript frontend (will be reset)
├── specs/            # Specifications (will be cleared)
├── agents/           # Agent artifacts (will be reset)
├── ai_docs/          # AI documentation (preserved)
├── .claude/          # Claude configuration (preserved)
├── start.sh          # Startup script (preserved/recreated)
└── README.md         # Project readme (will be reset to starter)
```

## Instructions

- **CRITICAL**: Use exact paths for all rm -rf commands to prevent accidental deletion
- Remove backend and frontend directories completely before recreating
- Initialize backend with astral uv and FastAPI health check endpoint
- Initialize frontend with Vite + Vue + TypeScript
- Clear specs directory contents but keep the directory
- Clear agents directory contents but keep the directory
- Preserve ai_docs and .claude directories completely
- Preserve start.sh if it exists, recreate if missing
- Reset README.md to starter template with current structure preserved

## Workflow

### Step 1: Safety verification
Confirm we're in the correct working directory:
```bash
pwd
# Should output: .../voice-to-agents/apps/content-gen
```

### Step 2: Remove existing backend and frontend
**CRITICAL: Use exact full paths**
```bash
rm -rf /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/backend
rm -rf /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/frontend
```

### Step 3: Clear specs and agents directories
```bash
rm -rf /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/specs/*
rm -rf /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/agents/*
```

### Step 4: Initialize fresh backend with FastAPI
```bash
cd /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen
mkdir -p backend/src/content_gen_backend
cd backend
uv init --name content-gen-backend --no-readme
```

The structure should now be:
```
backend/
├── pyproject.toml
└── src/
    └── content_gen_backend/
```

Create `src/content_gen_backend/main.py` with FastAPI health check:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Content Gen Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3333"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "content-gen-backend"}
```

Add FastAPI dependency:
```bash
uv add fastapi uvicorn[standard]
```

### Step 5: Initialize fresh frontend with Vite + Vue + TypeScript
```bash
cd /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen
npm create vite@latest frontend -- --template vue-ts
cd frontend
npm install
```

### Step 6: Reset main README.md
```bash
cd /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen
```

Create fresh README.md:
```markdown
# Content Generation Application

> Voice-based agentic coding experiment sandbox

## Codebase Structure

```
backend/          # FastAPI backend service
frontend/         # Vue + TypeScript frontend
specs/            # Project specifications
agents/           # Agent working directory
ai_docs/          # AI documentation
.claude/          # Claude Code configuration
```

## Quick Start

### Start Both Services (Recommended)
```bash
./start.sh
```

### Or Start Individually

**Backend:**
```bash
cd backend
uv sync
uv run uvicorn src.content_gen_backend.main:app --reload --port 4444
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev -- --port 3333
```

## Services

- Backend: http://localhost:4444
- Frontend: http://localhost:3333
- Health Check: http://localhost:4444/health
```

Save this to README.md.

### Step 7: Ensure start.sh exists
Check if start.sh exists, and create it if missing:
```bash
cd /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen

if [ ! -f start.sh ]; then
    echo "Creating start.sh..."
    cat > start.sh << 'EOF'
#!/bin/bash
# Start both backend and frontend services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill processes on ports if needed
echo "🧹 Cleaning up ports..."
lsof -ti:4444 | xargs kill -9 2>/dev/null || true
lsof -ti:3333 | xargs kill -9 2>/dev/null || true

echo ""
echo "🚀 Starting services..."
echo ""

# Start backend
echo "📦 Starting backend on http://localhost:4444"
cd backend
uv run uvicorn src.content_gen_backend.main:app --reload --port 4444 &
BACKEND_PID=$!

# Start frontend
echo "🎨 Starting frontend on http://localhost:3333"
cd ../frontend
npm run dev -- --port 3333 &
FRONTEND_PID=$!

echo ""
echo "✅ Services started!"
echo ""
echo "Backend:  http://localhost:4444"
echo "Frontend: http://localhost:3333"
echo "Health:   http://localhost:4444/health"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait
EOF
    chmod +x start.sh
    echo "✅ start.sh created"
else
    echo "✅ start.sh already exists (preserved)"
fi
```

### Step 8: Verify the reset
Run these verification commands:
```bash
# Verify backend exists and has health endpoint
ls -la /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/backend/src/content_gen_backend/main.py

# Verify frontend exists with Vite setup
ls -la /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/frontend/vite.config.ts

# Verify specs and agents are empty
ls -la /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/specs/
ls -la /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/agents/

# Verify preserved directories still exist
ls -la /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/ai_docs/
ls -la /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/.claude/

# Verify start.sh exists and is executable
ls -lah /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/start.sh

# Verify README.md was updated
cat /Users/indydevdan/Documents/projects/experimental/voice-to-agents/apps/content-gen/README.md
```

### Step 9: Copy .env into backend
> CD back into application root directory and copy .env.ready into backend directory

```bash
cp .env.ready apps/content-gen/backend/.env.ready
```

## Report

Provide a summary report with:

1. **Reset Status**: Confirm successful reset of backend, frontend, specs, and agents
2. **Preserved Items**: Confirm ai_docs and .claude directories remain intact
3. **Backend Verification**:
   - Confirm uv project initialized
   - Confirm FastAPI health endpoint created
   - Show FastAPI dependencies added
4. **Frontend Verification**:
   - Confirm Vite + Vue + TypeScript project created
   - Confirm package.json exists with correct dependencies
5. **Startup Script**: Confirm start.sh exists and is executable
6. **README Status**: Confirm main README.md updated to starter template

Format the report clearly with status indicators (✅ for success, ❌ for issues).
