package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/HeidiZHH/skull/internal/scraper"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
)

// MCPServer wraps the official MCP server with our business logic
type MCPServer struct {
	logger         zerolog.Logger
	mcpServer      *mcp.Server
	scraperService *scraper.Service
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

	server := &MCPServer{
		logger:         logger,
		mcpServer:      mcpServer,
		scraperService: scraperService,
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

// Removed summarization-oriented tools. The agent/caller will perform summarization.

// registerTools registers all our tools with the MCP server
func (s *MCPServer) registerTools() error {
	// Register scrape_url tool
	scrapeURLTool := &mcp.Tool{
		Name:        "scrape_url",
		Description: "Scrape content from a single URL and extract text, title, and metadata",
		InputSchema: &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"url": {
					Type:        "string",
					Description: "The URL to scrape",
				},
				"selector": {
					Type:        "string",
					Description: "Optional CSS selector to extract specific content from the page",
				},
			},
			Required: []string{"url"},
		},
	}
	mcp.AddTool(s.mcpServer, scrapeURLTool, s.handleScrapeURL)

	return nil
}

// Tool handler implementations
func (s *MCPServer) handleScrapeURL(
	ctx context.Context,
	req *mcp.CallToolRequest,
	args ScrapeURLParams,
) (*mcp.CallToolResult, any, error) {
	s.logger.Info().
		Str("url", args.URL).
		Str("selector", args.Selector).
		Msg("Scraping URL")

	// Use the actual scraper service
	result, err := s.scraperService.ScrapeURL(ctx, args.URL, args.Selector)
	if err != nil {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error scraping URL: %v", err),
				},
			},
			IsError: true,
		}, nil, nil
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

	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: fmt.Sprintf("Successfully scraped %s\n\nTitle: %s\n\nContent Preview:\n%s",
					result.URL, result.Title, result.CleanText[:min(len(result.CleanText), 500)]+"..."),
			},
			// Include raw content for clients that want to post-process (e.g., summarization)
			&mcp.TextContent{
				Text: "RAW_CONTENT:\n" + result.CleanText,
			},
		},
	}, responseData, nil
}

// Start starts the MCP server using stdio transport (most common)
func (s *MCPServer) Start(ctx context.Context) error {
	s.logger.Info().Msg("Starting MCP server with OpenAI integration")
	s.logger.Info().Msg("Available tools: scrape_url")

	// Use stdio transport - this is the standard for MCP servers
	transport := &mcp.StdioTransport{}

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
	// Add flag for HTTP transport
	httpAddr := flag.String("http", "", "Serve MCP server over HTTP at the given address (e.g. :8080)")
	flag.Parse()

	// Create logger
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).With().Timestamp().Logger()

	// Create and start the MCP server
	server, err := NewMCPServer(logger)
	if err != nil {
		log.Fatalf("Failed to create MCP server: %v", err)
	}

	ctx := context.Background()
	if *httpAddr != "" {
		logger.Info().Str("http", *httpAddr).Msg("Starting MCP server with HTTP transport")
		h := mcp.NewSSEHandler(func(r *http.Request) *mcp.Server { return server.mcpServer })
		if err := http.ListenAndServe(*httpAddr, h); err != nil {
			log.Fatalf("HTTP server failed: %v", err)
		}
		return
	}

	logger.Info().Msg("Starting MCP server with stdio transport")
	transport := &mcp.StdioTransport{}
	if err := server.mcpServer.Run(ctx, transport); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
