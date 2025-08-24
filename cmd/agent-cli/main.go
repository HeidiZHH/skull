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
	"github.com/HeidiZHH/skull/internal/scraper"
	"github.com/HeidiZHH/skull/internal/summarizer"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
)

// AgentCLI provides an interactive command-line interface
type AgentCLI struct {
	agent              *agent.Agent
	scraperService     *scraper.Service
	summarizerService  *summarizer.Service
	logger             zerolog.Logger
	lastScrapedContent string
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

	// Initialize scraper service
	scraperConfig := scraper.Config{
		UserAgent:   "skull-agent/1.0",
		Timeout:     30000000000, // 30 seconds
		MaxRetries:  3,
		RateLimit:   1000000000,       // 1 second
		MaxBodySize: 10 * 1024 * 1024, // 10MB
	}
	scraperService := scraper.NewService(scraperConfig, logger)

	// Initialize summarizer service
	summarizerConfig := summarizer.Config{
		Provider:  "openai",
		APIKey:    apiKey,
		BaseURL:   baseURL,
		Model:     model,
		MaxTokens: 500,
	}
	summarizerService := summarizer.NewService(summarizerConfig, logger)

	return &AgentCLI{
		agent:             agentService,
		scraperService:    scraperService,
		summarizerService: summarizerService,
		logger:            logger,
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

// executeToolCall executes a specific tool call
func (cli *AgentCLI) executeToolCall(ctx context.Context, toolCall agent.ToolCall) (string, error) {
	switch toolCall.Name {
	case "scrape_url":
		return cli.executeScrapeURL(ctx, toolCall.Arguments)
	case "summarize_content":
		return cli.executeSummarizeContent(ctx, toolCall.Arguments)
	case "scrape_and_summarize":
		return cli.executeScrapeAndSummarize(ctx, toolCall.Arguments)
	default:
		return "", fmt.Errorf("unknown tool: %s", toolCall.Name)
	}
}

// executeScrapeURL executes the scrape_url tool
func (cli *AgentCLI) executeScrapeURL(ctx context.Context, args map[string]interface{}) (string, error) {
	url, ok := args["url"].(string)
	if !ok {
		return "", fmt.Errorf("url parameter is required and must be a string")
	}

	selector, _ := args["selector"].(string)

	// Execute remotely via MCP server
	endpoint := os.Getenv("MCP_SERVER")
	if endpoint == "" {
		return "", fmt.Errorf("MCP_SERVER is not set; cannot call remote tool")
	}

	client := mcp.NewClient(&mcp.Implementation{Name: "skull-agent-cli"}, nil)
	transport := &mcp.SSEClientTransport{Endpoint: endpoint}
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return "", fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	defer session.Close()

	params := &mcp.CallToolParams{
		Name:      "scrape_url",
		Arguments: map[string]any{"url": url, "selector": selector},
	}
	res, err := session.CallTool(ctx, params)
	if err != nil {
		return "", fmt.Errorf("remote tool call failed: %w", err)
	}
	// Extract human-readable message from content
	var msgParts []string
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			msgParts = append(msgParts, tc.Text)
		}
	}
	// Capture structured content if available
	var preview string
	if res.StructuredContent != nil {
		if m, ok := res.StructuredContent.(map[string]any); ok {
			if s, ok := m["content"].(string); ok {
				cli.lastScrapedContent = s
				preview = truncateText(s, 300)
			}
		}
	}
	out := strings.Join(msgParts, "\n\n")
	if preview != "" {
		out += "\n\nContent preview:\n" + preview
	}
	return out, nil
}

// executeSummarizeContent executes the summarize_content tool
func (cli *AgentCLI) executeSummarizeContent(ctx context.Context, args map[string]interface{}) (string, error) {
	content, ok := args["content"].(string)
	if !ok {
		return "", fmt.Errorf("content parameter is required and must be a string")
	}

	maxLength := 200
	if ml, ok := args["max_length"].(float64); ok {
		maxLength = int(ml)
	}

	// Validate content
	if err := cli.summarizerService.ValidateContent(content); err != nil {
		return "", fmt.Errorf("content validation failed: %w", err)
	}

	req := summarizer.Request{
		Content:   content,
		MaxLength: maxLength,
		Style:     "concise",
	}

	result, err := cli.summarizerService.Summarize(ctx, req)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	return fmt.Sprintf("Summary (using %s, %d tokens):\n\n%s\n\nOriginal: %d chars → Summary: %d chars",
		result.Model,
		result.TokensUsed,
		result.Summary,
		result.OriginalSize,
		result.SummarySize,
	), nil
}

// executeScrapeAndSummarize executes the scrape_and_summarize tool
func (cli *AgentCLI) executeScrapeAndSummarize(ctx context.Context, args map[string]interface{}) (string, error) {
	url, ok := args["url"].(string)
	if !ok {
		return "", fmt.Errorf("url parameter is required and must be a string")
	}

	maxLength := 200
	if ml, ok := args["max_length"].(float64); ok {
		maxLength = int(ml)
	}

	selector, _ := args["selector"].(string)

	// First scrape
	scrapeResult, err := cli.scraperService.ScrapeURL(ctx, url, selector)
	if err != nil {
		return "", fmt.Errorf("scraping failed: %w", err)
	}

	// Validate content
	if err := cli.summarizerService.ValidateContent(scrapeResult.CleanText); err != nil {
		return "", fmt.Errorf("content validation failed: %w", err)
	}

	// Then summarize
	req := summarizer.Request{
		Content:   scrapeResult.CleanText,
		MaxLength: maxLength,
		Style:     "concise",
	}

	summaryResult, err := cli.summarizerService.Summarize(ctx, req)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	return fmt.Sprintf("Scraped and summarized: %s\n\nTitle: %s\nSummary (using %s, %d tokens):\n\n%s\n\nOriginal: %d chars → Summary: %d chars",
		scrapeResult.URL,
		scrapeResult.Title,
		summaryResult.Model,
		summaryResult.TokensUsed,
		summaryResult.Summary,
		summaryResult.OriginalSize,
		summaryResult.SummarySize,
	), nil
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
