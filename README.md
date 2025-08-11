# Skull

An agentic AI tool for web scraping and summarization using Model Context Protocol (MCP).

## Quick Start

```bash
# Build the interactive agent
go build -o skull-agent ./cmd/agent-cli

# Set up your LLM endpoint (example with DeepSeek)
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_MODEL="deepseek-chat"

# Run the agent
./skull-agent
```

## What it does

- Natural language interface for web scraping
- LLM-powered content summarization
- MCP protocol integration for AI assistants

## Example

```
🤖 You: Please search and summarize what is MCP from https://modelcontextprotocol.io
🧠 Agent: I'll scrape that webpage and create a summary for you.
✅ Result: The Model Context Protocol (MCP) is an open protocol designed to 
standardize how applications provide context to large language models...
```
