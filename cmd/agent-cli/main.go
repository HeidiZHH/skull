package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
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
	agentService := agent.NewAgent(logger, mcpSession, agentConfig)

	return &AgentCLI{
		agent:     agentService,
		mcpClient: mcpSession,
		logger:    logger,
	}, nil
}

// getEnvDefault retrieves an environment variable or returns a default value
func getEnvDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}

// Run starts the interactive CLI
func (cli *AgentCLI) Run(ctx context.Context) error {
	if err := cli.agent.RefreshTools(ctx); err != nil {
		return fmt.Errorf("failed to refresh tools: %w", err)
	}

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
			fmt.Println("❌ Please enter a valid query.")
			continue
		}

		if strings.ToLower(userInput) == "exit" || strings.ToLower(userInput) == "quit" {
			fmt.Println("👋 Goodbye!")
			break
		}

		// Process the user query
		fmt.Print("🤔 Thinking... ")
		response, err := cli.agent.ProcessQuery(ctx, userInput)
		if err != nil {
			fmt.Printf("❌ Error: %v\n\n", err)
			continue
		}

		// Clear the "Thinking..." line
		fmt.Print("\r")

		// Display the response
		if response.Response != "" {
			fmt.Printf("🤖 Agent: %s\n", response.Response)
		}

		// If tools were used, optionally show tool information
		if len(response.ToolCalls) > 0 {
			fmt.Printf("🔧 Used %d tool(s) to answer your question.\n", len(response.ToolCalls))
		}

		fmt.Println()
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

func main() {
	// Create logger
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).With().Timestamp().Logger()
	ctx := context.Background()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is required")
		return
	}

	serverConfig := &mcp.Config{
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
	server, err := mcp.NewServer(ctx, serverConfig, logger)
	if err != nil {
		log.Fatalf("Failed to create MCP server: %v", err)
	}
	clientTransport, serverTransport := gomcp.NewInMemoryTransports()

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
