package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/HeidiZHH/skull/internal/agent"
	"github.com/HeidiZHH/skull/internal/scraper"
	"github.com/HeidiZHH/skull/internal/summarizer"
	"github.com/rs/zerolog"
)

// This example shows how to use the Skull AI agent programmatically
func main() {
	// Create logger
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr}).With().Timestamp().Logger()

	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Initialize services
	agentConfig := agent.Config{
		Provider:    "openai",
		APIKey:      apiKey,
		Model:       "gpt-3.5-turbo",
		MaxTokens:   1000,
		Temperature: 0.2,
	}
	agentService := agent.NewAgent(agentConfig, logger)

	scraperConfig := scraper.Config{
		UserAgent:   "skull-example/1.0",
		Timeout:     30000000000, // 30 seconds
		MaxRetries:  3,
		RateLimit:   1000000000, // 1 second
		MaxBodySize: 10 * 1024 * 1024,
	}
	scraperService := scraper.NewService(scraperConfig, logger)

	summarizerConfig := summarizer.Config{
		Provider:  "openai",
		APIKey:    apiKey,
		Model:     "gpt-3.5-turbo",
		MaxTokens: 500,
	}
	summarizerService := summarizer.NewService(summarizerConfig, logger)

	ctx := context.Background()

	// Example 1: Process natural language request
	fmt.Println("Example 1: Natural Language Processing")
	fmt.Println("=====================================")

	userRequests := []string{
		"Can you scrape the content from https://example.com?",
		"I need a summary of this text: Go is a programming language developed by Google. It's known for its simplicity, efficiency, and strong support for concurrent programming. Go is statically typed and compiled, making it fast and reliable for building scalable applications.",
		"Please get me a summary of the webpage https://golang.org",
	}

	for i, request := range userRequests {
		fmt.Printf("\n🤖 Request %d: %s\n", i+1, request)

		response, err := agentService.ProcessInput(ctx, request)
		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
			continue
		}

		fmt.Printf("🧠 Agent Response: %s\n", response.Message)
		fmt.Printf("🎯 Confidence: %.1f%%\n", response.Confidence*100)
		fmt.Printf("🔧 Should Execute Tools: %v\n", response.ShouldCall)

		if response.ShouldCall && len(response.ToolCalls) > 0 {
			for _, toolCall := range response.ToolCalls {
				fmt.Printf("   📋 Tool: %s\n", toolCall.Name)
				fmt.Printf("   💭 Reasoning: %s\n", toolCall.Reasoning)

				// Execute the tool call
				result, err := executeToolCall(ctx, toolCall, scraperService, summarizerService)
				if err != nil {
					fmt.Printf("   ❌ Execution Error: %v\n", err)
				} else {
					fmt.Printf("   ✅ Result: %s\n", truncate(result, 200))
				}
			}
		}
	}

	// Example 2: Direct service usage
	fmt.Println("\n\nExample 2: Direct Service Usage")
	fmt.Println("===============================")

	// Direct web scraping
	fmt.Println("\n🌐 Direct Scraping:")
	scrapeResult, err := scraperService.ScrapeURL(ctx, "https://example.com", "")
	if err != nil {
		fmt.Printf("❌ Scraping error: %v\n", err)
	} else {
		fmt.Printf("✅ Scraped: %s\n", scrapeResult.Title)
		fmt.Printf("   Content length: %d characters\n", len(scrapeResult.CleanText))
	}

	// Direct summarization
	fmt.Println("\n📝 Direct Summarization:")
	summaryReq := summarizer.Request{
		Content:   "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. AI can be categorized into narrow AI, which is designed to perform a narrow task, and general AI, which has generalized human cognitive abilities.",
		MaxLength: 50,
		Style:     "concise",
	}

	summaryResult, err := summarizerService.Summarize(ctx, summaryReq)
	if err != nil {
		fmt.Printf("❌ Summarization error: %v\n", err)
	} else {
		fmt.Printf("✅ Summary: %s\n", summaryResult.Summary)
		fmt.Printf("   Model: %s, Tokens: %d\n", summaryResult.Model, summaryResult.TokensUsed)
	}

	fmt.Println("\n🎉 Example completed!")
}

// executeToolCall executes a tool call using the appropriate service
func executeToolCall(ctx context.Context, toolCall agent.ToolCall, scraperService *scraper.Service, summarizerService *summarizer.Service) (string, error) {
	switch toolCall.Name {
	case "scrape_url":
		url, ok := toolCall.Arguments["url"].(string)
		if !ok {
			return "", fmt.Errorf("invalid URL parameter")
		}
		selector, _ := toolCall.Arguments["selector"].(string)

		result, err := scraperService.ScrapeURL(ctx, url, selector)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Scraped %s: %s (%d chars)", result.URL, result.Title, len(result.CleanText)), nil

	case "summarize_content":
		content, ok := toolCall.Arguments["content"].(string)
		if !ok {
			return "", fmt.Errorf("invalid content parameter")
		}

		maxLength := 200
		if ml, ok := toolCall.Arguments["max_length"].(float64); ok {
			maxLength = int(ml)
		}

		req := summarizer.Request{
			Content:   content,
			MaxLength: maxLength,
			Style:     "concise",
		}

		result, err := summarizerService.Summarize(ctx, req)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Summary (%d tokens): %s", result.TokensUsed, result.Summary), nil

	case "scrape_and_summarize":
		url, ok := toolCall.Arguments["url"].(string)
		if !ok {
			return "", fmt.Errorf("invalid URL parameter")
		}

		maxLength := 200
		if ml, ok := toolCall.Arguments["max_length"].(float64); ok {
			maxLength = int(ml)
		}
		selector, _ := toolCall.Arguments["selector"].(string)

		// Scrape first
		scrapeResult, err := scraperService.ScrapeURL(ctx, url, selector)
		if err != nil {
			return "", fmt.Errorf("scraping failed: %w", err)
		}

		// Then summarize
		req := summarizer.Request{
			Content:   scrapeResult.CleanText,
			MaxLength: maxLength,
			Style:     "concise",
		}

		summaryResult, err := summarizerService.Summarize(ctx, req)
		if err != nil {
			return "", fmt.Errorf("summarization failed: %w", err)
		}

		return fmt.Sprintf("Scraped and summarized %s: %s", scrapeResult.URL, summaryResult.Summary), nil

	default:
		return "", fmt.Errorf("unknown tool: %s", toolCall.Name)
	}
}

// truncate helper function
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
