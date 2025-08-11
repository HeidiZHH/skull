package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/HeidiZHH/skull/internal/scraper"
	"github.com/HeidiZHH/skull/internal/summarizer"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
)

// MCPServer wraps the official MCP server with our business logic
type MCPServer struct {
	logger            zerolog.Logger
	mcpServer         *mcp.Server
	scraperService    *scraper.Service
	summarizerService *summarizer.Service
}

// NewMCPServer creates a new MCP server instance using the official SDK
func NewMCPServer(logger zerolog.Logger) (*MCPServer, error) {
	// Create the official MCP server
	impl := &mcp.Implementation{
		Name:    "Skull Web Scraper & Summarizer",
		Version: "1.0.0",
	}

	serverOpts := &mcp.ServerOptions{}
	mcpServer := mcp.NewServer(impl, serverOpts)

	// Initialize scraper service
	scraperConfig := scraper.Config{
		UserAgent:   "skull-agent/1.0",
		Timeout:     30 * time.Second,
		MaxRetries:  3,
		RateLimit:   1 * time.Second,
		MaxBodySize: 10 * 1024 * 1024, // 10MB
	}
	scraperService := scraper.NewService(scraperConfig, logger)

	// Initialize summarizer service
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is required")
	}

	summarizerConfig := summarizer.Config{
		Provider:  "openai",
		APIKey:    apiKey,
		BaseURL:   "", // Use default OpenAI endpoint
		Model:     "gpt-3.5-turbo",
		MaxTokens: 500,
	}
	summarizerService := summarizer.NewService(summarizerConfig, logger)

	server := &MCPServer{
		logger:            logger,
		mcpServer:         mcpServer,
		scraperService:    scraperService,
		summarizerService: summarizerService,
	}

	// Register our tools with the MCP server
	if err := server.registerTools(); err != nil {
		return nil, fmt.Errorf("failed to register tools: %w", err)
	}

	return server, nil
}

// Parameter types for our tools
type ScrapeURLParams struct {
	URL      string `json:"url"`
	Selector string `json:"selector,omitempty"`
}

type SummarizeParams struct {
	Content   string `json:"content"`
	MaxLength int    `json:"max_length,omitempty"`
}

type ScrapeAndSummarizeParams struct {
	URL       string `json:"url"`
	MaxLength int    `json:"max_length,omitempty"`
	Selector  string `json:"selector,omitempty"`
}

// registerTools registers all our tools with the MCP server
func (s *MCPServer) registerTools() error {
	// Register scrape_url tool
	scrapeURLTool := &mcp.Tool{
		Name:        "scrape_url",
		Description: "Scrape content from a single URL and extract text, title, and metadata",
	}
	mcp.AddTool(s.mcpServer, scrapeURLTool, s.handleScrapeURL)

	// Register summarize_content tool
	summarizeTool := &mcp.Tool{
		Name:        "summarize_content",
		Description: "Generate a summary from text content using LLM",
	}
	mcp.AddTool(s.mcpServer, summarizeTool, s.handleSummarizeContent)

	// Register combined scrape_and_summarize tool
	combinedTool := &mcp.Tool{
		Name:        "scrape_and_summarize",
		Description: "Scrape a URL and generate a summary in one step",
	}
	mcp.AddTool(s.mcpServer, combinedTool, s.handleScrapeAndSummarize)

	return nil
}

// Tool handler implementations
func (s *MCPServer) handleScrapeURL(
	ctx context.Context,
	session *mcp.ServerSession,
	params *mcp.CallToolParamsFor[ScrapeURLParams],
) (*mcp.CallToolResultFor[map[string]interface{}], error) {
	s.logger.Info().
		Str("url", params.Arguments.URL).
		Str("selector", params.Arguments.Selector).
		Msg("Scraping URL")

	// Use the actual scraper service
	result, err := s.scraperService.ScrapeURL(ctx, params.Arguments.URL, params.Arguments.Selector)
	if err != nil {
		return &mcp.CallToolResultFor[map[string]interface{}]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error scraping URL: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	responseData := map[string]interface{}{
		"url":          result.URL,
		"title":        result.Title,
		"content":      result.CleanText,
		"links_count":  len(result.Links),
		"images_count": len(result.Images),
		"status_code":  result.StatusCode,
		"content_type": result.ContentType,
	}

	return &mcp.CallToolResultFor[map[string]interface{}]{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: fmt.Sprintf("Successfully scraped %s\n\nTitle: %s\n\nContent Preview:\n%s",
					result.URL, result.Title, result.CleanText[:min(len(result.CleanText), 500)]+"..."),
			},
		},
		StructuredContent: responseData,
	}, nil
}

func (s *MCPServer) handleSummarizeContent(
	ctx context.Context,
	session *mcp.ServerSession,
	params *mcp.CallToolParamsFor[SummarizeParams],
) (*mcp.CallToolResultFor[map[string]interface{}], error) {
	s.logger.Info().
		Int("content_length", len(params.Arguments.Content)).
		Int("max_length", params.Arguments.MaxLength).
		Msg("Summarizing content")

	// Validate content
	if err := s.summarizerService.ValidateContent(params.Arguments.Content); err != nil {
		return &mcp.CallToolResultFor[map[string]interface{}]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Content validation error: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	// Use the actual summarizer service
	req := summarizer.Request{
		Content:   params.Arguments.Content,
		MaxLength: params.Arguments.MaxLength,
		Style:     "concise",
	}

	if req.MaxLength == 0 {
		req.MaxLength = 200
	}

	summaryResult, err := s.summarizerService.Summarize(ctx, req)
	if err != nil {
		return &mcp.CallToolResultFor[map[string]interface{}]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error generating summary: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	responseData := map[string]interface{}{
		"summary":       summaryResult.Summary,
		"original_size": summaryResult.OriginalSize,
		"summary_size":  summaryResult.SummarySize,
		"model":         summaryResult.Model,
		"tokens_used":   summaryResult.TokensUsed,
	}

	return &mcp.CallToolResultFor[map[string]interface{}]{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: fmt.Sprintf("Summary (using %s, %d tokens):\n\n%s",
					summaryResult.Model, summaryResult.TokensUsed, summaryResult.Summary),
			},
		},
		StructuredContent: responseData,
	}, nil
}

func (s *MCPServer) handleScrapeAndSummarize(
	ctx context.Context,
	session *mcp.ServerSession,
	params *mcp.CallToolParamsFor[ScrapeAndSummarizeParams],
) (*mcp.CallToolResultFor[map[string]interface{}], error) {
	s.logger.Info().
		Str("url", params.Arguments.URL).
		Int("max_length", params.Arguments.MaxLength).
		Str("selector", params.Arguments.Selector).
		Msg("Scraping and summarizing URL")

	// First, scrape the URL
	scrapeResult, err := s.scraperService.ScrapeURL(ctx, params.Arguments.URL, params.Arguments.Selector)
	if err != nil {
		return &mcp.CallToolResultFor[map[string]interface{}]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error scraping URL: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	// Validate scraped content
	if err := s.summarizerService.ValidateContent(scrapeResult.CleanText); err != nil {
		return &mcp.CallToolResultFor[map[string]interface{}]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Scraped content validation error: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	// Then, summarize the content
	req := summarizer.Request{
		Content:   scrapeResult.CleanText,
		MaxLength: params.Arguments.MaxLength,
		Style:     "concise",
	}

	if req.MaxLength == 0 {
		req.MaxLength = 200
	}

	summaryResult, err := s.summarizerService.Summarize(ctx, req)
	if err != nil {
		return &mcp.CallToolResultFor[map[string]interface{}]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error generating summary: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	responseData := map[string]interface{}{
		"url":           scrapeResult.URL,
		"title":         scrapeResult.Title,
		"summary":       summaryResult.Summary,
		"original_size": summaryResult.OriginalSize,
		"summary_size":  summaryResult.SummarySize,
		"model":         summaryResult.Model,
		"tokens_used":   summaryResult.TokensUsed,
		"links_count":   len(scrapeResult.Links),
		"images_count":  len(scrapeResult.Images),
	}

	return &mcp.CallToolResultFor[map[string]interface{}]{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: fmt.Sprintf("Scraped and summarized: %s\n\nTitle: %s\n\nSummary (using %s, %d tokens):\n%s",
					scrapeResult.URL, scrapeResult.Title, summaryResult.Model, summaryResult.TokensUsed, summaryResult.Summary),
			},
		},
		StructuredContent: responseData,
	}, nil
}

// Start starts the MCP server using stdio transport (most common)
func (s *MCPServer) Start(ctx context.Context) error {
	s.logger.Info().Msg("Starting MCP server with OpenAI integration")
	s.logger.Info().Msg("Available tools: scrape_url, summarize_content, scrape_and_summarize")

	// Use stdio transport - this is the standard for MCP servers
	transport := mcp.NewStdioTransport()

	// Run the server - this blocks until the client disconnects
	if err := s.mcpServer.Run(ctx, transport); err != nil {
		return fmt.Errorf("MCP server failed: %w", err)
	}

	return nil
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Main function
func main() {
	// Create logger
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).With().Timestamp().Logger()

	// Create and start the MCP server
	server, err := NewMCPServer(logger)
	if err != nil {
		log.Fatalf("Failed to create MCP server: %v", err)
	}

	ctx := context.Background()
	if err := server.Start(ctx); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
