package mcp

import (
	"context"
	"fmt"
	"time"

	"github.com/HeidiZHH/skull/internal/scraper"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
)

// Config represents the server configuration
type Config struct {
	Server ServerConfig `yaml:"server"`
	Tools  ToolsConfig  `yaml:"tools"`
}

// ServerConfig represents server-specific configuration
type ServerConfig struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
}

// ToolsConfig represents tools configuration
type ToolsConfig struct {
	Scraper ScraperConfig `yaml:"scraper"`
}

// ScraperConfig represents web scraper configuration
type ScraperConfig struct {
	UserAgent   string        `yaml:"userAgent"`
	Timeout     time.Duration `yaml:"timeout"`
	MaxRetries  int           `yaml:"maxRetries"`
	RateLimit   time.Duration `yaml:"rateLimit"`
	MaxBodySize int64         `yaml:"maxBodySize"`
}

// Server represents the MCP server using the official SDK
type Server struct {
	config         *Config
	logger         zerolog.Logger
	server         *mcp.Server
	scraperService *scraper.Service
}

// NewServer creates a new MCP server instance using the official SDK
func NewServer(config *Config, logger zerolog.Logger) (*Server, error) {
	// Create the implementation with our tools
	impl := &mcp.Implementation{}

	// Create the official MCP server
	mcpServer := mcp.NewServer(impl, &mcp.ServerOptions{})

	// Initialize services
	scraperConfig := scraper.Config{
		UserAgent:   config.Tools.Scraper.UserAgent,
		Timeout:     config.Tools.Scraper.Timeout,
		MaxRetries:  config.Tools.Scraper.MaxRetries,
		RateLimit:   config.Tools.Scraper.RateLimit,
		MaxBodySize: config.Tools.Scraper.MaxBodySize,
	}
	scraperService := scraper.NewService(scraperConfig, logger)

	s := &Server{
		config:         config,
		logger:         logger,
		server:         mcpServer,
		scraperService: scraperService,
	}

	// Register tools
	s.registerTools()

	return s, nil
}

// registerTools registers all available tools with the MCP server
func (s *Server) registerTools() {
	// Register scrape_url tool
	scrapeURLTool := &mcp.Tool{
		Name:        "scrape_url",
		Description: "Scrape content from a single URL",
	}
	mcp.AddTool(s.server, scrapeURLTool, s.handleScrapeURL)

}

// Tool handlers - these implement the actual tool functionality

// ScrapeURLParams represents the parameters for scrape_url tool
type ScrapeURLParams struct {
	URL      string `json:"url"`
	Selector string `json:"selector,omitempty"`
}

// ScrapeURLResult represents the result of scrape_url tool
type ScrapeURLResult struct {
	Content string `json:"content"`
	URL     string `json:"url"`
}

// handleScrapeURL handles the scrape_url tool
func (s *Server) handleScrapeURL(ctx context.Context, session *mcp.ServerSession, params *mcp.CallToolParamsFor[ScrapeURLParams]) (*mcp.CallToolResultFor[ScrapeURLResult], error) {
	url := params.Arguments.URL
	selector := params.Arguments.Selector

	s.logger.Info().Str("url", url).Str("selector", selector).Msg("Scraping URL")

	// Use the actual scraper service
	result, err := s.scraperService.ScrapeURL(ctx, url, selector)
	if err != nil {
		return &mcp.CallToolResultFor[ScrapeURLResult]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error scraping URL: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	return &mcp.CallToolResultFor[ScrapeURLResult]{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: fmt.Sprintf("Successfully scraped %s\n\nTitle: %s\n\nContent:\n%s",
					result.URL, result.Title, result.CleanText),
			},
		},
		StructuredContent: ScrapeURLResult{
			Content: result.CleanText,
			URL:     result.URL,
		},
	}, nil
}

// Start starts the MCP server using stdio transport (standard for MCP)
func (s *Server) Start(ctx context.Context) error {
	s.logger.Info().Str("name", s.config.Server.Name).Str("version", s.config.Server.Version).Msg("Starting MCP server")

	// Create stdio transport (standard for MCP servers)
	transport := mcp.NewStdioTransport()

	// Connect the server to the transport
	conn, err := s.server.Connect(ctx, transport)
	if err != nil {
		return fmt.Errorf("failed to connect server: %w", err)
	}

	// Wait for the connection to close
	return conn.Wait()
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
