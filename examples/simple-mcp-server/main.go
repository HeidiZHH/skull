package main

import (
	"context"
	"fmt"
	"log"

	"github.com/modelcontextprotocol/go-sdk/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// Parameter types for our tools
type ScrapeURLParams struct {
	URL     string `json:"url"`
	Timeout int    `json:"timeout,omitempty"`
}

type SummarizeParams struct {
	Content   string `json:"content"`
	MaxLength int    `json:"max_length,omitempty"`
}

// Tool handlers
func handleScrapeURL(
	ctx context.Context,
	session *mcp.ServerSession,
	params *mcp.CallToolParamsFor[ScrapeURLParams],
) (*mcp.CallToolResultFor[any], error) {
	log.Printf("Scraping URL: %s", params.Arguments.URL)

	// TODO: Implement actual scraping logic
	result := fmt.Sprintf("Successfully scraped content from %s", params.Arguments.URL)

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{
			&mcp.TextContent{Text: result},
		},
	}, nil
}

func handleSummarizeContent(
	ctx context.Context,
	session *mcp.ServerSession,
	params *mcp.CallToolParamsFor[SummarizeParams],
) (*mcp.CallToolResultFor[any], error) {
	log.Printf("Summarizing content of length: %d", len(params.Arguments.Content))

	// TODO: Implement actual summarization logic
	summary := "This is a placeholder summary of the provided content."

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{
			&mcp.TextContent{Text: summary},
		},
	}, nil
}

func main() {
	// Create MCP server
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "Web Scraper & Summarizer",
		Version: "1.0.0",
	}, nil)

	// Register scrape_url tool
	scrapeURLTool := &mcp.Tool{
		Name:        "scrape_url",
		Description: "Scrape content from a single URL",
		InputSchema: &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"url": {
					Type:        "string",
					Description: "The URL to scrape",
					Format:      "uri",
				},
				"timeout": {
					Type:        "integer",
					Description: "Timeout in seconds (default: 30)",
				},
			},
			Required: []string{"url"},
		},
	}

	mcp.AddTool(server, scrapeURLTool, handleScrapeURL)

	// Register summarize_content tool
	summarizeTool := &mcp.Tool{
		Name:        "summarize_content",
		Description: "Generate a summary from text content",
		InputSchema: &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"content": {
					Type:        "string",
					Description: "Text content to summarize",
				},
				"max_length": {
					Type:        "integer",
					Description: "Maximum summary length in words (default: 100)",
				},
			},
			Required: []string{"content"},
		},
	}

	mcp.AddTool(server, summarizeTool, handleSummarizeContent)

	log.Println("Starting MCP server...")

	// Use stdio transport (standard for MCP servers)
	transport := mcp.NewStdioTransport()

	// Run the server
	if err := server.Run(context.Background(), transport); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
