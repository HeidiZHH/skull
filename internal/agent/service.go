package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"

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
	MCPServer   string // MCP server endpoint for tool discovery
}

// ToolDefinition represents a tool that the agent can call
type ToolDefinition struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  *jsonschema.Schema `json:"parameters"`
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

	// Tools must come from MCP server only
	if config.MCPServer != "" {
		if err := agent.fetchToolsFromMCP(); err != nil {
			agent.logger.Warn().Err(err).Msg("Failed to fetch tools from MCP server; continuing with no tools")
			agent.tools = []ToolDefinition{}
		}
	} else {
		agent.logger.Warn().Msg("MCP_SERVER is not configured; agent will operate with no tools")
		agent.tools = []ToolDefinition{}
	}

	return agent
}

// Tools returns the currently known tool definitions.
func (a *Agent) Tools() []ToolDefinition { return a.tools }

// fetchToolsFromMCP fetches tool definitions from the MCP server endpoint
func (a *Agent) fetchToolsFromMCP() error {
	// Use MCP client to list tools
	client := mcp.NewClient(&mcp.Implementation{Name: "skull-agent-client"}, &mcp.ClientOptions{})
	transport := &mcp.SSEClientTransport{Endpoint: a.config.MCPServer}
	session, err := client.Connect(context.Background(), transport, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	defer session.Close()
	result, err := session.ListTools(context.Background(), &mcp.ListToolsParams{})
	if err != nil {
		return fmt.Errorf("failed to list tools from MCP server: %w", err)
	}
	var tools []ToolDefinition
	for _, t := range result.Tools {
		tools = append(tools, ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}
	a.tools = tools
	a.logger.Info().Int("tool_count", len(a.tools)).Msg("Fetched tools from MCP server")
	return nil
}

// registerDefaultTools registers the web scraping and summarization tools
// default tool registration removed to enforce MCP-only tools

// ProcessInput analyzes user input and determines what tools to call
func (a *Agent) ProcessInput(ctx context.Context, userInput string) (*Response, error) {
	a.logger.Info().Str("input", userInput).Msg("Processing user input")

	// Create system prompt that teaches the agent about tools
	systemPrompt := a.buildSystemPrompt()

	// Create user message that asks for tool analysis (no scenario hints; rely on tool list and schemas)
	userPrompt := fmt.Sprintf(`Analyze this user request and determine if any tools should be called:

User Request: "%s"

Instructions:
1. Decide if any of the available tools are needed based on their names, descriptions, and JSON schemas
2. If tools should be used, specify which ones and with what parameters
3. Provide reasoning for your decisions
4. Respond in JSON format as specified in the system prompt`, userInput)

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
	var guidelines []string
	// We intentionally don't expose a standalone summarize tool; summarize text directly only when part of combined flow
	guidelines = append(guidelines,
		"- If the user asks general questions that don't require web content, set \"should_call\" to false",
		"- Be conservative: only call tools when clearly needed",
		"- Provide clear reasoning for your decisions",
		"- Extract parameters accurately from user input",
		"- Set confidence based on how clear the user's intent is",
	)

	return fmt.Sprintf(`You are an intelligent agent that helps users with tasks. You have access to the following tools:

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
%s`, string(toolsJSON), strings.Join(guidelines, "\n"))
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
	params := toolDef.Parameters.Properties
	required := toolDef.Parameters.Required
	for _, reqParam := range required {
		if _, exists := toolCall.Arguments[reqParam]; !exists {
			return fmt.Errorf("missing required parameter '%s' for tool '%s'", reqParam, toolCall.Name)
		}
	}

	// Validate parameter types (basic validation)
	for paramName, paramValue := range toolCall.Arguments {
		if paramDef, exists := params[paramName]; exists {
			expectedType := paramDef.Type
			if err := a.validateParameterType(paramName, paramValue, expectedType); err != nil {
				return err
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
