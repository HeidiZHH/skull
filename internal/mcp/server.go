package mcp

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/HeidiZHH/skull/internal/scraper"
	"github.com/HeidiZHH/skull/internal/summarizer"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
)

// Config represents the server configuration
type Config struct {
	Tools ToolsConfig `yaml:"tools"`
}

// ToolsConfig represents tools configuration
type ToolsConfig struct {
	Scraper    ScraperConfig    `yaml:"scraper"`
	Summarizer SummarizerConfig `yaml:"summarizer"`
}

// ScraperConfig represents web scraper configuration
type ScraperConfig struct {
	UserAgent   string        `yaml:"userAgent"`
	Timeout     time.Duration `yaml:"timeout"`
	MaxRetries  int           `yaml:"maxRetries"`
	RateLimit   time.Duration `yaml:"rateLimit"`
	MaxBodySize int64         `yaml:"maxBodySize"`
}

// SummarizerConfig represents summarizer configuration
type SummarizerConfig struct {
	Provider  string `yaml:"provider"`
	APIKey    string `yaml:"apiKey"`
	BaseURL   string `yaml:"baseURL"`
	Model     string `yaml:"model"`
	MaxTokens int    `yaml:"maxTokens"`
}

// Server represents the MCP server using the official SDK
type Server struct {
	config            *Config
	logger            zerolog.Logger
	server            *mcp.Server
	scraperService    *scraper.Service
	summarizerService *summarizer.Service
}

// NewServer creates a new MCP server instance using the official SDK
func NewServer(ctx context.Context, config *Config, logger zerolog.Logger) (*Server, error) {
	// Create the implementation with our tools
	impl := &mcp.Implementation{
		Name:    "Skull Web Scraper & Summarizer",
		Version: "1.0.0",
	}

	// Create the official MCP server
	mcpServer := mcp.NewServer(impl, nil)

	// Initialize services
	scraperConfig := scraper.Config{
		UserAgent:   config.Tools.Scraper.UserAgent,
		Timeout:     config.Tools.Scraper.Timeout,
		MaxRetries:  config.Tools.Scraper.MaxRetries,
		RateLimit:   config.Tools.Scraper.RateLimit,
		MaxBodySize: config.Tools.Scraper.MaxBodySize,
	}
	scraperService := scraper.NewService(scraperConfig, logger)

	// Get API key from environment if not set
	apiKey := config.Tools.Summarizer.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	summarizerConfig := summarizer.Config{
		Provider:  config.Tools.Summarizer.Provider,
		APIKey:    apiKey,
		BaseURL:   config.Tools.Summarizer.BaseURL,
		Model:     config.Tools.Summarizer.Model,
		MaxTokens: config.Tools.Summarizer.MaxTokens,
	}
	summarizerService := summarizer.NewService(summarizerConfig, logger)

	s := &Server{
		config:            config,
		logger:            logger,
		server:            mcpServer,
		scraperService:    scraperService,
		summarizerService: summarizerService,
	}

	// Register tools
	s.registerTools(mcpServer)

	return s, nil
}

// registerTools registers all available tools with the MCP server
func (s *Server) registerTools(server *mcp.Server) {
	// Register scrape_url tool
	scrapeURLTool := &mcp.Tool{
		Name:        "scrape_url",
		Description: "Scrape content from a single URL",
	}
	mcp.AddTool(server, scrapeURLTool, s.handleScrapeURL)

	// Register summarize_content tool
	summarizeTool := &mcp.Tool{
		Name:        "summarize_content",
		Description: "Generate a summary of the provided text content",
	}
	mcp.AddTool(server, summarizeTool, s.handleSummarizeContent)

	// Register combined scrape_and_summarize tool
	scrapeAndSummarizeTool := &mcp.Tool{
		Name:        "scrape_and_summarize",
		Description: "Scrape a URL and generate a summary of its content",
	}
	mcp.AddTool(server, scrapeAndSummarizeTool, s.handleScrapeAndSummarize)
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

// SummarizeContentParams represents the parameters for summarize_content tool
type SummarizeContentParams struct {
	Content   string `json:"content"`
	MaxLength int    `json:"max_length,omitempty"`
}

// SummarizeContentResult represents the result of summarize_content tool
type SummarizeContentResult struct {
	Summary   string `json:"summary"`
	Length    int    `json:"length"`
	MaxLength int    `json:"max_length"`
}

// handleSummarizeContent handles the summarize_content tool
func (s *Server) handleSummarizeContent(ctx context.Context, session *mcp.ServerSession, params *mcp.CallToolParamsFor[SummarizeContentParams]) (*mcp.CallToolResultFor[SummarizeContentResult], error) {
	content := params.Arguments.Content
	maxLength := params.Arguments.MaxLength
	if maxLength == 0 {
		maxLength = 200
	}

	s.logger.Info().Int("max_length", maxLength).Msg("Summarizing content")

	// Validate content
	if err := s.summarizerService.ValidateContent(content); err != nil {
		return &mcp.CallToolResultFor[SummarizeContentResult]{
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
		Content:   content,
		MaxLength: maxLength,
		Style:     "concise",
	}

	summaryResult, err := s.summarizerService.Summarize(ctx, req)
	if err != nil {
		return &mcp.CallToolResultFor[SummarizeContentResult]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error generating summary: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	return &mcp.CallToolResultFor[SummarizeContentResult]{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: fmt.Sprintf("Summary (using %s, %d tokens):\n\n%s",
					summaryResult.Model, summaryResult.TokensUsed, summaryResult.Summary),
			},
		},
		StructuredContent: SummarizeContentResult{
			Summary:   summaryResult.Summary,
			Length:    summaryResult.SummarySize,
			MaxLength: maxLength,
		},
	}, nil
}

// ScrapeAndSummarizeParams represents the parameters for scrape_and_summarize tool
type ScrapeAndSummarizeParams struct {
	URL       string `json:"url"`
	MaxLength int    `json:"max_length,omitempty"`
	Selector  string `json:"selector,omitempty"`
}

// ScrapeAndSummarizeResult represents the result of scrape_and_summarize tool
type ScrapeAndSummarizeResult struct {
	Summary   string `json:"summary"`
	URL       string `json:"url"`
	Length    int    `json:"length"`
	MaxLength int    `json:"max_length"`
}

// handleScrapeAndSummarize handles the combined scrape_and_summarize tool
func (s *Server) handleScrapeAndSummarize(ctx context.Context, session *mcp.ServerSession, params *mcp.CallToolParamsFor[ScrapeAndSummarizeParams]) (*mcp.CallToolResultFor[ScrapeAndSummarizeResult], error) {
	url := params.Arguments.URL
	maxLength := params.Arguments.MaxLength
	if maxLength == 0 {
		maxLength = 200
	}
	selector := params.Arguments.Selector

	s.logger.Info().
		Str("url", url).
		Str("selector", selector).
		Int("max_length", maxLength).
		Msg("Scraping and summarizing URL")

	// First, scrape the URL
	scrapeResult, err := s.scraperService.ScrapeURL(ctx, url, selector)
	if err != nil {
		return &mcp.CallToolResultFor[ScrapeAndSummarizeResult]{
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
		return &mcp.CallToolResultFor[ScrapeAndSummarizeResult]{
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
		MaxLength: maxLength,
		Style:     "concise",
	}

	summaryResult, err := s.summarizerService.Summarize(ctx, req)
	if err != nil {
		return &mcp.CallToolResultFor[ScrapeAndSummarizeResult]{
			Content: []mcp.Content{
				&mcp.TextContent{
					Text: fmt.Sprintf("Error generating summary: %v", err),
				},
			},
			IsError: true,
		}, nil
	}

	result := fmt.Sprintf("Scraped and summarized: %s\n\nTitle: %s\n\nSummary (using %s, %d tokens):\n%s",
		url, scrapeResult.Title, summaryResult.Model, summaryResult.TokensUsed, summaryResult.Summary)

	return &mcp.CallToolResultFor[ScrapeAndSummarizeResult]{
		Content: []mcp.Content{
			&mcp.TextContent{
				Text: result,
			},
		},
		StructuredContent: ScrapeAndSummarizeResult{
			Summary:   summaryResult.Summary,
			URL:       url,
			Length:    summaryResult.SummarySize,
			MaxLength: maxLength,
		},
	}, nil
}

// Start starts the MCP server using stdio transport (standard for MCP)
func (s *Server) Start(ctx context.Context, transport mcp.Transport) error {

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
