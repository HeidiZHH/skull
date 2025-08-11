package summarizer

import (
	"context"
	"fmt"
	"strings"

	"github.com/rs/zerolog"
	"github.com/sashabaranov/go-openai"
)

// Service handles text summarization using LLMs
type Service struct {
	client *openai.Client
	config Config
	logger zerolog.Logger
}

// Config represents summarizer configuration
type Config struct {
	Provider  string // "openai", "custom", etc.
	APIKey    string
	BaseURL   string // For custom OpenAI-compatible endpoints
	Model     string
	MaxTokens int
}

// Request represents a summarization request
type Request struct {
	Content   string `json:"content"`
	MaxLength int    `json:"max_length,omitempty"`
	Style     string `json:"style,omitempty"` // "concise", "detailed", "bullet_points"
	Language  string `json:"language,omitempty"`
}

// Response represents a summarization response
type Response struct {
	Summary      string            `json:"summary"`
	OriginalSize int               `json:"original_size"`
	SummarySize  int               `json:"summary_size"`
	Model        string            `json:"model"`
	TokensUsed   int               `json:"tokens_used"`
	Metadata     map[string]string `json:"metadata"`
}

// NewService creates a new summarizer service
func NewService(config Config, logger zerolog.Logger) *Service {
	clientConfig := openai.DefaultConfig(config.APIKey)

	// Support custom OpenAI-compatible endpoints
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
	}

	client := openai.NewClientWithConfig(clientConfig)

	return &Service{
		client: client,
		config: config,
		logger: logger.With().Str("component", "summarizer").Logger(),
	}
}

// Summarize generates a summary of the provided content
func (s *Service) Summarize(ctx context.Context, req Request) (*Response, error) {
	if req.MaxLength == 0 {
		req.MaxLength = 200
	}
	if req.Style == "" {
		req.Style = "concise"
	}

	s.logger.Info().
		Int("content_length", len(req.Content)).
		Int("max_length", req.MaxLength).
		Str("style", req.Style).
		Msg("Starting summarization")

	// Build the prompt based on style
	prompt := s.buildPrompt(req)

	// Prepare the chat completion request
	chatReq := openai.ChatCompletionRequest{
		Model: s.config.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You are a helpful assistant that creates clear, accurate summaries of text content.",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens:   s.config.MaxTokens,
		Temperature: 0.3, // Lower temperature for more consistent summaries
	}

	// Call the LLM
	resp, err := s.client.CreateChatCompletion(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create chat completion: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned")
	}

	summary := resp.Choices[0].Message.Content
	summary = strings.TrimSpace(summary)

	response := &Response{
		Summary:      summary,
		OriginalSize: len(req.Content),
		SummarySize:  len(summary),
		Model:        resp.Model,
		TokensUsed:   resp.Usage.TotalTokens,
		Metadata: map[string]string{
			"style":             req.Style,
			"language":          req.Language,
			"prompt_tokens":     fmt.Sprintf("%d", resp.Usage.PromptTokens),
			"completion_tokens": fmt.Sprintf("%d", resp.Usage.CompletionTokens),
		},
	}

	s.logger.Info().
		Int("original_size", response.OriginalSize).
		Int("summary_size", response.SummarySize).
		Int("tokens_used", response.TokensUsed).
		Str("model", response.Model).
		Msg("Summarization completed")

	return response, nil
}

// buildPrompt constructs the summarization prompt based on the request
func (s *Service) buildPrompt(req Request) string {
	var promptBuilder strings.Builder

	// Base instruction
	promptBuilder.WriteString("Please summarize the following text")

	// Add length constraint
	if req.MaxLength > 0 {
		promptBuilder.WriteString(fmt.Sprintf(" in approximately %d words or less", req.MaxLength))
	}

	// Add style instructions
	switch req.Style {
	case "detailed":
		promptBuilder.WriteString(". Provide a detailed summary that captures the main points, key arguments, and important details")
	case "bullet_points":
		promptBuilder.WriteString(". Format the summary as bullet points, highlighting the key information")
	case "concise":
		promptBuilder.WriteString(". Provide a concise summary focusing on the most important information")
	default:
		promptBuilder.WriteString(". Provide a clear and informative summary")
	}

	// Add language instruction if specified
	if req.Language != "" {
		promptBuilder.WriteString(fmt.Sprintf(" in %s", req.Language))
	}

	promptBuilder.WriteString(":\n\n")
	promptBuilder.WriteString(req.Content)

	return promptBuilder.String()
}

// SummarizeWithKeywords generates a summary and extracts keywords
func (s *Service) SummarizeWithKeywords(ctx context.Context, req Request) (*Response, []string, error) {
	// First get the regular summary
	response, err := s.Summarize(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	// Extract keywords with a separate request
	keywordPrompt := fmt.Sprintf(`Extract 5-10 key terms or phrases from the following text. Return only the keywords, separated by commas:

%s`, req.Content)

	keywordReq := openai.ChatCompletionRequest{
		Model: s.config.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You extract key terms and phrases from text. Return only the keywords separated by commas.",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: keywordPrompt,
			},
		},
		MaxTokens:   100,
		Temperature: 0.2,
	}

	keywordResp, err := s.client.CreateChatCompletion(ctx, keywordReq)
	if err != nil {
		s.logger.Warn().Err(err).Msg("Failed to extract keywords")
		return response, []string{}, nil
	}

	if len(keywordResp.Choices) > 0 {
		keywordsText := keywordResp.Choices[0].Message.Content
		keywords := strings.Split(keywordsText, ",")

		// Clean up keywords
		var cleanKeywords []string
		for _, keyword := range keywords {
			cleaned := strings.TrimSpace(keyword)
			if cleaned != "" {
				cleanKeywords = append(cleanKeywords, cleaned)
			}
		}

		return response, cleanKeywords, nil
	}

	return response, []string{}, nil
}

// ValidateContent checks if content is suitable for summarization
func (s *Service) ValidateContent(content string) error {
	if content == "" {
		return fmt.Errorf("content cannot be empty")
	}

	words := len(strings.Fields(content))
	if words < 10 {
		return fmt.Errorf("content too short (minimum 10 words, got %d)", words)
	}

	if words > 10000 {
		return fmt.Errorf("content too long (maximum 10000 words, got %d)", words)
	}

	return nil
}
