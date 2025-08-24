package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/HeidiZHH/skull/internal/agent"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
)

// AgentCLI provides an interactive command-line interface
type AgentCLI struct {
	agent  *agent.Agent
	logger zerolog.Logger
}

// NewAgentCLI creates a new CLI instance
func NewAgentCLI(logger zerolog.Logger) (*AgentCLI, error) {
	// Require OPENAI_API_KEY; used for OpenAI-compatible providers (including DeepSeek)
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is required")
	}

	// Initialize agent
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		// Default to DeepSeek's OpenAI-compatible endpoint if not provided
		baseURL = "https://api.deepseek.com/v1"
	}
	model := getEnvDefault("OPENAI_MODEL", "gpt-3.5-turbo")
	if strings.Contains(strings.ToLower(baseURL), "deepseek.com") && os.Getenv("OPENAI_MODEL") == "" {
		model = "deepseek-chat"
	}
	agentConfig := agent.Config{
		Provider:    "openai",
		APIKey:      apiKey,
		BaseURL:     baseURL, // Supports DeepSeek or custom endpoints
		Model:       model,
		MaxTokens:   1000,
		Temperature: 0.2,
		MCPServer:   os.Getenv("MCP_SERVER"), // e.g. http://localhost:8080
	}
	agentService := agent.NewAgent(agentConfig, logger)

	return &AgentCLI{
		agent:  agentService,
		logger: logger,
	}, nil
}

// Run starts the interactive CLI
func (cli *AgentCLI) Run(ctx context.Context) error {
	fmt.Println("🧠 Skull AI Agent - Web Scraping & Summarization Assistant")
	fmt.Println("=========================================================")
	fmt.Println()
	fmt.Println("I can help you with:")
	fmt.Println("• Scraping content from websites")
	fmt.Println("• Summarizing text content")
	fmt.Println("• Getting summaries of web pages")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("• \"Scrape content from https://example.com\"")
	fmt.Println("• \"Summarize this webpage: https://news.example.com\"")
	fmt.Println("• \"Get me a summary of the latest news from https://blog.example.com\"")
	fmt.Println()
	fmt.Println("Type 'exit' or 'quit' to stop.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("🤖 You: ")

		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())

		if userInput == "" {
			continue
		}

		if strings.ToLower(userInput) == "exit" || strings.ToLower(userInput) == "quit" {
			fmt.Println("👋 Goodbye!")
			break
		}

		// Process the user input with the agent
		if err := cli.processUserInput(ctx, userInput); err != nil {
			fmt.Printf("❌ Error: %v\n\n", err)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scanner error: %w", err)
	}

	return nil
}

// processUserInput handles a single user input
func (cli *AgentCLI) processUserInput(ctx context.Context, userInput string) error {
	fmt.Printf("🤔 Thinking...\n")

	// Let the agent analyze the input
	response, err := cli.agent.ProcessInput(ctx, userInput)
	if err != nil {
		return fmt.Errorf("agent processing failed: %w", err)
	}

	// Show the agent's understanding
	fmt.Printf("🧠 Agent: %s\n", response.Message)

	if response.Confidence < 0.5 {
		fmt.Printf("⚠️  Confidence: %.1f%% - I'm not very confident about this interpretation.\n", response.Confidence*100)
	}

	// If no tools should be called, we're done
	if !response.ShouldCall || len(response.ToolCalls) == 0 {
		fmt.Printf("💭 %s\n\n", response.Explanation)
		return nil
	}

	// Execute tool calls
	fmt.Printf("🔧 Executing %d tool(s)...\n", len(response.ToolCalls))

	// Aggregate raw outputs to feed into post-processing
	var aggregated []string

	for i, toolCall := range response.ToolCalls {
		fmt.Printf("\n🛠️  Tool %d/%d: %s\n", i+1, len(response.ToolCalls), toolCall.Name)
		fmt.Printf("📝 Reasoning: %s\n", toolCall.Reasoning)

		// Validate the tool call
		if err := cli.agent.ValidateToolCall(toolCall); err != nil {
			fmt.Printf("❌ Validation failed: %v\n", err)
			continue
		}

		// Execute the tool call
		result, err := cli.executeToolCall(ctx, toolCall)
		if err != nil {
			fmt.Printf("❌ Execution failed: %v\n", err)
			continue
		}

		fmt.Printf("✅ Result: %s\n", result)
		if strings.TrimSpace(result) != "" {
			aggregated = append(aggregated, result)
		}
	}

	// Generic post-processing step if requested by the agent
	if response.PostProcess != "" {
		// Use aggregated tool outputs for post-processing
		content := strings.TrimSpace(strings.Join(aggregated, "\n\n"))
		if content != "" {
			fmt.Printf("\n🧪 Post-processing: %s...\n", response.PostProcess)
			final, err := cli.agent.PostProcess(ctx, response.PostProcess, userInput, content)
			if err != nil {
				fmt.Printf("⚠️  Post-process failed: %v\n\n", err)
			} else {
				fmt.Printf("\n🧾 Final Output:\n%s\n\n", final)
			}
		}
	}

	fmt.Println()
	return nil
}

// executeToolCall executes a specific tool call
func (cli *AgentCLI) executeToolCall(ctx context.Context, toolCall agent.ToolCall) (string, error) {
	// Route all tool calls to the agent's reusable MCP session
	res, err := cli.agent.CallToolRemote(ctx, toolCall.Name, toolCall.Arguments)
	if err != nil {
		return "", err
	}
	var msgParts []string
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			msgParts = append(msgParts, tc.Text)
		}
	}
	return strings.Join(msgParts, "\n\n"), nil
}

// executeRemoteTool removed; we rely on the agent's CallToolRemote

// Helper functions
func getEnvDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// truncateText was removed; previews are no longer constructed locally

func main() {
	// Create logger
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).With().Timestamp().Logger()

	// Optional single-run input flag for non-interactive testing
	input := flag.String("input", "", "Process a single input then exit (non-interactive mode)")
	flag.Parse()

	// Create CLI
	cli, err := NewAgentCLI(logger)
	if err != nil {
		log.Fatalf("Failed to create CLI: %v", err)
	}

	// Run the interactive CLI
	ctx := context.Background()
	if *input != "" {
		// Non-interactive single-run
		if err := cli.processUserInput(ctx, *input); err != nil {
			log.Fatalf("CLI failed: %v", err)
		}
		return
	}

	if err := cli.Run(ctx); err != nil {
		log.Fatalf("CLI failed: %v", err)
	}
}
