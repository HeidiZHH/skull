package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/rs/zerolog"
	"github.com/sashabaranov/go-openai"
)

// Agent handles natural language interpretation and tool calling
type Agent struct {
	client *openai.Client
	config Config
	logger zerolog.Logger
	tools  []ToolDefinition
}

// Config represents agent configuration
type Config struct {
	Provider    string // "openai", "custom", etc.
	APIKey      string
	BaseURL     string // For custom OpenAI-compatible endpoints
	Model       string
	MaxTokens   int
	Temperature float32
}

// ToolDefinition represents a tool that the agent can call
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToolCall represents a tool call decision made by the agent
type ToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
	Reasoning string                 `json:"reasoning"`
}

// Response represents the agent's response to user input
type Response struct {
	Message     string     `json:"message"`
	ToolCalls   []ToolCall `json:"tool_calls,omitempty"`
	ShouldCall  bool       `json:"should_call"`
	Confidence  float64    `json:"confidence"`
	Explanation string     `json:"explanation"`
}

// NewAgent creates a new agent instance
func NewAgent(config Config, logger zerolog.Logger) *Agent {
	clientConfig := openai.DefaultConfig(config.APIKey)

	// Support custom OpenAI-compatible endpoints
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
	}

	client := openai.NewClientWithConfig(clientConfig)

	agent := &Agent{
		client: client,
		config: config,
		logger: logger.With().Str("component", "agent").Logger(),
		tools:  []ToolDefinition{},
	}

	// Register default tools
	agent.registerDefaultTools()

	return agent
}

// registerDefaultTools registers the web scraping and summarization tools
func (a *Agent) registerDefaultTools() {
	// Scrape URL tool
	a.tools = append(a.tools, ToolDefinition{
		Name:        "scrape_url",
		Description: "Scrape content from a single URL and extract text, title, and metadata. Use this when the user wants to get content from a specific webpage.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "The URL to scrape",
				},
				"selector": map[string]interface{}{
					"type":        "string",
					"description": "Optional CSS selector to extract specific content from the page",
				},
			},
			"required": []string{"url"},
		},
	})

	// Summarize content tool
	a.tools = append(a.tools, ToolDefinition{
		Name:        "summarize_content",
		Description: "Generate a summary from text content using LLM. Use this when the user wants to summarize existing text content.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"content": map[string]interface{}{
					"type":        "string",
					"description": "The text content to summarize",
				},
				"max_length": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum length of the summary in words (default: 200)",
				},
			},
			"required": []string{"content"},
		},
	})

	// Combined scrape and summarize tool
	a.tools = append(a.tools, ToolDefinition{
		Name:        "scrape_and_summarize",
		Description: "Scrape a URL and generate a summary in one step. Use this when the user wants to get a summary of a webpage's content.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "The URL to scrape and summarize",
				},
				"max_length": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum length of the summary in words (default: 200)",
				},
				"selector": map[string]interface{}{
					"type":        "string",
					"description": "Optional CSS selector to extract specific content from the page",
				},
			},
			"required": []string{"url"},
		},
	})
}

// ProcessInput analyzes user input and determines what tools to call
func (a *Agent) ProcessInput(ctx context.Context, userInput string) (*Response, error) {
	a.logger.Info().Str("input", userInput).Msg("Processing user input")

	// Create system prompt that teaches the agent about tools
	systemPrompt := a.buildSystemPrompt()

	// Create user message that asks for tool analysis
	userPrompt := fmt.Sprintf(`Analyze this user request and determine if any tools should be called:

User Request: "%s"

Instructions:
1. Determine if the request requires using any of the available tools
2. If tools should be used, specify which ones and with what parameters
3. Provide reasoning for your decisions
4. Respond in JSON format as specified in the system prompt

Consider these scenarios:
- If user wants to get content from a URL → use scrape_url
- If user wants to summarize existing text → use summarize_content  
- If user wants to summarize a webpage → use scrape_and_summarize
- If user asks general questions without needing web content → no tools needed`, userInput)

	// Prepare the chat completion request
	chatReq := openai.ChatCompletionRequest{
		Model: a.config.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: userPrompt,
			},
		},
		MaxTokens:   a.config.MaxTokens,
		Temperature: a.config.Temperature,
	}

	// Call the LLM
	resp, err := a.client.CreateChatCompletion(ctx, chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create chat completion: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned")
	}

	content := resp.Choices[0].Message.Content
	content = strings.TrimSpace(content)

	// Handle JSON wrapped in markdown code blocks
	if strings.HasPrefix(content, "```json") {
		// Extract JSON from markdown code block
		lines := strings.Split(content, "\n")
		var jsonLines []string
		inCodeBlock := false
		for _, line := range lines {
			if strings.HasPrefix(line, "```json") {
				inCodeBlock = true
				continue
			}
			if strings.HasPrefix(line, "```") && inCodeBlock {
				break
			}
			if inCodeBlock {
				jsonLines = append(jsonLines, line)
			}
		}
		content = strings.Join(jsonLines, "\n")
	}

	// Parse the JSON response
	var response Response
	if err := json.Unmarshal([]byte(content), &response); err != nil {
		// If JSON parsing fails, create a fallback response
		a.logger.Warn().Err(err).Str("content", content).Msg("Failed to parse agent response as JSON")
		return &Response{
			Message:     "I understand your request, but I had trouble determining the best approach. Could you please rephrase your request?",
			ShouldCall:  false,
			Confidence:  0.1,
			Explanation: "Failed to parse agent decision",
		}, nil
	}

	a.logger.Info().
		Bool("should_call", response.ShouldCall).
		Int("tool_calls", len(response.ToolCalls)).
		Float64("confidence", response.Confidence).
		Msg("Agent analysis complete")

	return &response, nil
}

// buildSystemPrompt creates the system prompt that defines the agent's behavior
func (a *Agent) buildSystemPrompt() string {
	toolsJSON, _ := json.MarshalIndent(a.tools, "", "  ")

	return fmt.Sprintf(`You are an intelligent agent that helps users with web scraping and content summarization tasks. You have access to the following tools:

%s

Your job is to analyze user requests and determine:
1. Whether any tools should be called to fulfill the request
2. Which specific tools to call and with what parameters
3. The reasoning behind your decisions

IMPORTANT: You must respond in valid JSON format with this exact structure:
{
  "message": "A helpful message to the user about what you understand from their request",
  "tool_calls": [
    {
      "name": "tool_name",
      "arguments": {
        "param1": "value1",
        "param2": "value2"
      },
      "reasoning": "Why this tool call is needed"
    }
  ],
  "should_call": true/false,
  "confidence": 0.0-1.0,
  "explanation": "Detailed explanation of your analysis and decisions"
}

Guidelines:
- If the user wants to get content from a specific URL, use "scrape_url"
- If the user wants to summarize existing text they provide, use "summarize_content"
- If the user wants to get a summary of a webpage, use "scrape_and_summarize"
- If the user asks general questions that don't require web content, set "should_call" to false
- Be conservative: only call tools when clearly needed
- Provide clear reasoning for your decisions
- Extract URLs, text content, and parameters accurately from user input
- Set confidence based on how clear the user's intent is`, string(toolsJSON))
}

// ValidateToolCall checks if a tool call is valid
func (a *Agent) ValidateToolCall(toolCall ToolCall) error {
	// Find the tool definition
	var toolDef *ToolDefinition
	for _, tool := range a.tools {
		if tool.Name == toolCall.Name {
			toolDef = &tool
			break
		}
	}

	if toolDef == nil {
		return fmt.Errorf("unknown tool: %s", toolCall.Name)
	}

	// Validate required parameters
	params, ok := toolDef.Parameters["properties"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid tool definition for %s", toolCall.Name)
	}

	required, ok := toolDef.Parameters["required"].([]string)
	if ok {
		for _, reqParam := range required {
			if _, exists := toolCall.Arguments[reqParam]; !exists {
				return fmt.Errorf("missing required parameter '%s' for tool '%s'", reqParam, toolCall.Name)
			}
		}
	}

	// Validate parameter types (basic validation)
	for paramName, paramValue := range toolCall.Arguments {
		if paramDef, exists := params[paramName]; exists {
			if paramDefMap, ok := paramDef.(map[string]interface{}); ok {
				expectedType, ok := paramDefMap["type"].(string)
				if ok {
					if err := a.validateParameterType(paramName, paramValue, expectedType); err != nil {
						return err
					}
				}
			}
		}
	}

	return nil
}

// validateParameterType validates a parameter's type
func (a *Agent) validateParameterType(paramName string, value interface{}, expectedType string) error {
	switch expectedType {
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("parameter '%s' must be a string", paramName)
		}
	case "integer":
		if _, ok := value.(float64); !ok { // JSON numbers are float64
			return fmt.Errorf("parameter '%s' must be an integer", paramName)
		}
	case "number":
		if _, ok := value.(float64); !ok {
			return fmt.Errorf("parameter '%s' must be a number", paramName)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("parameter '%s' must be a boolean", paramName)
		}
	}
	return nil
}
