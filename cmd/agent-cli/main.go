package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/HeidiZHH/skull/internal/agent"
	"github.com/HeidiZHH/skull/internal/mcp"
	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
)

// AgentCLI provides an interactive command-line interface
type AgentCLI struct {
	agent     *agent.Agent
	mcpClient *gomcp.ClientSession
	logger    zerolog.Logger
}

// NewAgentCLI creates a new CLI instance that uses MCP
func NewAgentCLI(logger zerolog.Logger, mcpSession *gomcp.ClientSession, apiKey string) (*AgentCLI, error) {
	// Initialize agent with MCP tools
	agentConfig := agent.Config{
		Provider:    "openai",
		APIKey:      apiKey,
		BaseURL:     os.Getenv("OPENAI_BASE_URL"), // Optional custom endpoint
		Model:       getEnvDefault("OPENAI_MODEL", "gpt-3.5-turbo"),
		MaxTokens:   1000,
		Temperature: 0.2,
	}

	// Create agent
	agentService := agent.NewAgent(agentConfig, logger)

	return &AgentCLI{
		agent:     agentService,
		mcpClient: mcpSession,
		logger:    logger,
	}, nil
}

// buildMCPServer builds the MCP server binary
func buildMCPServer(workspaceRoot string) error {
	cmd := exec.Command("go", "build", "-o", "mcp-server", "./cmd/mcp-server")
	cmd.Dir = workspaceRoot
	return cmd.Run()
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

// Close properly shuts down the CLI and its resources
func (cli *AgentCLI) Close() error {
	if cli.mcpClient != nil {
		return cli.mcpClient.Close()
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
	}

	fmt.Println()
	return nil
}

// executeToolCall executes a specific tool call using MCP
func (cli *AgentCLI) executeToolCall(ctx context.Context, toolCall gomcp.CallToolParamsFor[any]) (string, error) {
	// All tool calls now go through the MCP client
	result, err := cli.mcpClient.CallTool(ctx, toolCall.Name, toolCall.Arguments)
	if err != nil {
		return "", fmt.Errorf("MCP tool call failed: %w", err)
	}

	if !result.Success {
		return "", fmt.Errorf("tool execution failed: %s", result.Error)
	}

	// Join all content strings
	if len(result.Content) > 0 {
		return strings.Join(result.Content, "\n"), nil
	}

	return "Tool executed successfully", nil
}

// Helper functions
func getEnvDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen] + "..."
}

func main() {
	// Create logger
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).With().Timestamp().Logger()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is required")
		return
	}

	serverConfig := &mcp.Config{
		Server: mcp.ServerConfig{
			Name:    "Skull MCP Server",
			Version: "1.0.0",
		},
		Tools: mcp.ToolsConfig{
			Scraper: mcp.ScraperConfig{
				UserAgent:  "skull-agent/1.0",
				Timeout:    30 * time.Second,
				MaxRetries: 3,
				RateLimit:  1 * time.Second,
			},
			Summarizer: mcp.SummarizerConfig{
				Provider:  "deepseek",
				Model:     os.Getenv("OPENAI_MODEL"),
				MaxTokens: 500,
				APIKey:    apiKey,
			},
		},
	}
	// Create and start the MCP server
	server, err := mcp.NewServer(serverConfig, logger)
	if err != nil {
		log.Fatalf("Failed to create MCP server: %v", err)
	}
	clientTransport, serverTransport := gomcp.NewInMemoryTransports()
	ctx := context.Background()
	// Start the MCP server in a separate goroutine
	go func() {
		if err := server.Start(ctx, serverTransport); err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}()
	// Initialize MCP client
	mcpClient := gomcp.NewClient(&gomcp.Implementation{
		Name:    "Skull MCP Client",
		Version: "1.0.0",
	}, nil,
	)

	// Connect the MCP client to the server
	mcpSession, err := mcpClient.Connect(ctx, clientTransport)
	if err != nil {
		log.Fatalf("Failed to connect MCP client: %v", err)
		return
	}
	// Create CLI
	cli, err := NewAgentCLI(logger, mcpSession, apiKey)
	if err != nil {
		log.Fatalf("Failed to create CLI: %v", err)
		return
	}
	// Ensure proper cleanup
	defer func() {
		if err := cli.Close(); err != nil {
			logger.Error().Err(err).Msg("Failed to close CLI properly")
		}
	}()
	// Run the interactive CLI
	if err := cli.Run(ctx); err != nil {
		log.Fatalf("CLI failed: %v", err)
	}
}
