package scraper

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gocolly/colly/v2"
	"github.com/rs/zerolog"
)

// Service handles web scraping operations
type Service struct {
	config Config
	logger zerolog.Logger
	client *http.Client
}

// Config represents scraper configuration
type Config struct {
	UserAgent   string
	Timeout     time.Duration
	MaxRetries  int
	RateLimit   time.Duration
	MaxBodySize int64
}

// Result represents a scraping result
type Result struct {
	URL         string            `json:"url"`
	Title       string            `json:"title"`
	Content     string            `json:"content"`
	CleanText   string            `json:"clean_text"`
	Links       []string          `json:"links"`
	Images      []string          `json:"images"`
	Metadata    map[string]string `json:"metadata"`
	StatusCode  int               `json:"status_code"`
	ContentType string            `json:"content_type"`
}

// NewService creates a new scraper service
func NewService(config Config, logger zerolog.Logger) *Service {
	client := &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:       10,
			IdleConnTimeout:    30 * time.Second,
			DisableCompression: false,
		},
	}

	return &Service{
		config: config,
		logger: logger.With().Str("component", "scraper").Logger(),
		client: client,
	}
}

// ScrapeURL scrapes content from a single URL
func (s *Service) ScrapeURL(ctx context.Context, url string, selector string) (*Result, error) {
	s.logger.Info().Str("url", url).Str("selector", selector).Msg("Starting scrape")

	// Create collector with configuration
	c := colly.NewCollector(
		colly.UserAgent(s.config.UserAgent),
	)

	// Set limits
	c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: 1,
		Delay:       s.config.RateLimit,
	})

	// Set timeout
	c.SetRequestTimeout(s.config.Timeout)

	result := &Result{
		URL:      url,
		Links:    []string{},
		Images:   []string{},
		Metadata: make(map[string]string),
	}

	// Handle errors
	c.OnError(func(r *colly.Response, err error) {
		s.logger.Error().Err(err).Str("url", r.Request.URL.String()).Msg("Scraping error")
	})

	// Handle responses
	c.OnResponse(func(r *colly.Response) {
		result.StatusCode = r.StatusCode
		result.ContentType = r.Headers.Get("Content-Type")
		s.logger.Debug().Int("status", r.StatusCode).Str("content-type", result.ContentType).Msg("Received response")
	})

	// Parse HTML content
	c.OnHTML("html", func(e *colly.HTMLElement) {
		// Extract title
		result.Title = e.ChildText("title")

		// Extract metadata
		e.ForEach("meta", func(i int, meta *colly.HTMLElement) {
			name := meta.Attr("name")
			property := meta.Attr("property")
			content := meta.Attr("content")

			if name != "" && content != "" {
				result.Metadata[name] = content
			}
			if property != "" && content != "" {
				result.Metadata[property] = content
			}
		})

		// Extract links
		e.ForEach("a[href]", func(i int, link *colly.HTMLElement) {
			href := link.Attr("href")
			if href != "" {
				absoluteURL := e.Request.AbsoluteURL(href)
				result.Links = append(result.Links, absoluteURL)
			}
		})

		// Extract images
		e.ForEach("img[src]", func(i int, img *colly.HTMLElement) {
			src := img.Attr("src")
			if src != "" {
				absoluteURL := e.Request.AbsoluteURL(src)
				result.Images = append(result.Images, absoluteURL)
			}
		})

		// Extract content based on selector or default strategy
		if selector != "" {
			// Use custom selector
			result.Content = e.ChildText(selector)
			result.CleanText = strings.TrimSpace(result.Content)
		} else {
			// Default content extraction strategy
			s.extractDefaultContent(e, result)
		}
	})

	// Visit the URL
	err := c.Visit(url)
	if err != nil {
		return nil, fmt.Errorf("failed to scrape URL %s: %w", url, err)
	}

	// Wait for completion
	c.Wait()

	s.logger.Info().
		Str("url", url).
		Int("status", result.StatusCode).
		Int("content_length", len(result.CleanText)).
		Int("links", len(result.Links)).
		Int("images", len(result.Images)).
		Msg("Scraping completed")

	return result, nil
}

// extractDefaultContent extracts content using a default strategy
func (s *Service) extractDefaultContent(e *colly.HTMLElement, result *Result) {
	// Priority selectors for main content
	contentSelectors := []string{
		"main",
		"article",
		"[role=main]",
		".content",
		".post-content",
		".entry-content",
		".article-content",
		"#content",
		".main-content",
	}

	// Try each selector to find main content
	for _, selector := range contentSelectors {
		content := e.ChildText(selector)
		if content != "" && len(strings.TrimSpace(content)) > 100 {
			result.Content = content
			result.CleanText = s.cleanText(content)
			return
		}
	}

	// Fallback: extract from body but exclude common non-content elements
	excludeSelectors := []string{
		"nav", "header", "footer", "aside", ".sidebar", "#sidebar",
		".menu", ".navigation", ".breadcrumb", ".social", ".share",
		"script", "style", "noscript",
	}

	// Clone the selection and remove unwanted elements
	doc := e.DOM
	bodyContent := doc.Find("body").Clone()

	for _, excludeSelector := range excludeSelectors {
		bodyContent.Find(excludeSelector).Remove()
	}

	content := bodyContent.Text()
	result.Content = content
	result.CleanText = s.cleanText(content)
}

// cleanText cleans and normalizes extracted text
func (s *Service) cleanText(text string) string {
	// Remove extra whitespace and normalize
	lines := strings.Split(text, "\n")
	var cleanLines []string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			cleanLines = append(cleanLines, line)
		}
	}

	return strings.Join(cleanLines, "\n")
}

// ScrapeMultiple scrapes multiple URLs concurrently
func (s *Service) ScrapeMultiple(ctx context.Context, urls []string, selector string) ([]*Result, error) {
	results := make([]*Result, len(urls))
	errChan := make(chan error, len(urls))

	// Create a semaphore to limit concurrent requests
	semaphore := make(chan struct{}, 3) // Max 3 concurrent requests

	for i, url := range urls {
		go func(index int, u string) {
			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			result, err := s.ScrapeURL(ctx, u, selector)
			if err != nil {
				errChan <- fmt.Errorf("failed to scrape %s: %w", u, err)
				return
			}

			results[index] = result
			errChan <- nil
		}(i, url)
	}

	// Wait for all to complete
	var errors []error
	for i := 0; i < len(urls); i++ {
		if err := <-errChan; err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return results, fmt.Errorf("scraping errors: %v", errors)
	}

	return results, nil
}
