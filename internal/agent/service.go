package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/rs/zerolog"
	"github.com/sashabaranov/go-openai"
)

// Agent handles natural language interpretation and tool calling
type Agent struct {
	client    *openai.Client
	logger    zerolog.Logger
	mcpClient *mcp.ClientSession
	tools     []*mcp.Tool
	config    Config
}

// Config represents agent configuration
type Config struct {
	Provider    string // "openai", "custom", etc.
	APIKey      string
	BaseURL     string // For custom OpenAI-compatible endpoints
	Model       string
	MaxTokens   int
	Temperature float64
}

// QueryResponse represents the agent's response to a user query
type QueryResponse struct {
	Response    string                `json:"response"`
	ToolCalls   []*mcp.CallToolParams `json:"tool_calls,omitempty"`
	ToolResults []*mcp.CallToolResult `json:"tool_results,omitempty"`
}

func NewAgent(logger zerolog.Logger, mcpClient *mcp.ClientSession, config Config) *Agent {
	// Initialize OpenAI client with custom configuration
	clientConfig := openai.DefaultConfig(config.APIKey)
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
	}
	llmClient := openai.NewClientWithConfig(clientConfig)
	return &Agent{
		client:    llmClient,
		logger:    logger.With().Str("component", "agent").Logger(),
		mcpClient: mcpClient,
		tools:     []*mcp.Tool{},
		config:    config,
	}
}

func (a *Agent) RefreshTools(ctx context.Context) error {
	// Fetch tools from MCP client session
	tools, err := a.mcpClient.ListTools(ctx, nil)
	if err != nil {
		return err
	}

	a.tools = tools.Tools
	a.logger.Info().Int("tools_count", len(a.tools)).Msg("Refreshed tools from MCP client")
	return nil
}

// ProcessQuery analyzes a user query and determines which tools to use
func (a *Agent) ProcessQuery(ctx context.Context, query string) (*QueryResponse, error) {
	a.logger.Info().Str("query", query).Msg("Processing user query")

	// First, analyze the query to determine which tools are needed
	toolCalls, err := a.analyzeQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query: %w", err)
	}

	response := &QueryResponse{
		ToolCalls: toolCalls,
	}

	// If tools are needed, execute them
	if len(toolCalls) > 0 {
		a.logger.Info().Int("tool_count", len(toolCalls)).Msg("Executing tools")

		toolResults, err := a.executeTools(ctx, toolCalls)
		if err != nil {
			return nil, fmt.Errorf("failed to execute tools: %w", err)
		}
		response.ToolResults = toolResults

		// Generate final response based on tool results
		finalResponse, err := a.generateFinalResponse(ctx, query, toolResults)
		if err != nil {
			return nil, fmt.Errorf("failed to generate final response: %w", err)
		}
		response.Response = finalResponse
	} else {
		// No tools needed, generate direct response
		directResponse, err := a.generateDirectResponse(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("failed to generate direct response: %w", err)
		}
		response.Response = directResponse
	}

	return response, nil
}

// analyzeQuery uses the LLM to determine which tools are needed for the query
func (a *Agent) analyzeQuery(ctx context.Context, query string) ([]*mcp.CallToolParams, error) {
	a.logger.Info().Str("query", query).Msg("Starting query analysis")

	// Build tools description for the LLM
	toolsDesc := a.buildToolsDescription()
	a.logger.Debug().Str("tools_desc", toolsDesc).Msg("Built tools description")

	systemPrompt := fmt.Sprintf(`You are an AI assistant that analyzes user queries and determines which tools to use.

Available tools:
%s

Rules:
1. Analyze the user query and determine if any tools are needed
2. If tools are needed, respond with a JSON array of tool calls
3. If no tools are needed, respond with an empty array []
4. Each tool call should have the format: {"name": "tool_name", "arguments": {...}}
5. Only use tools that are actually available
6. Be precise with tool arguments based on the tool descriptions

Respond with only the JSON array of tool calls, no additional text.`, toolsDesc)

	userPrompt := fmt.Sprintf("User query: %s", query)

	a.logger.Debug().
		Str("model", a.config.Model).
		Str("base_url", a.config.BaseURL).
		Int("max_tokens", a.config.MaxTokens).
		Msg("Making LLM request")

	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
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
		Temperature: 0.1, // Low temperature for precise tool selection
	})

	if err != nil {
		a.logger.Error().Err(err).Msg("LLM request failed")
		return nil, fmt.Errorf("LLM request failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		a.logger.Warn().Msg("No response choices from LLM")
		return nil, fmt.Errorf("no response from LLM")
	}

	content := strings.TrimSpace(resp.Choices[0].Message.Content)
	a.logger.Info().Str("llm_response", content).Msg("LLM tool analysis response")

	// Clean up markdown code blocks if present
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	// Parse the JSON response
	var toolCalls []*mcp.CallToolParams
	if content == "[]" || content == "" {
		return toolCalls, nil // No tools needed
	}

	// Parse JSON into a slice of raw tool calls first
	var rawToolCalls []map[string]interface{}
	if err := json.Unmarshal([]byte(content), &rawToolCalls); err != nil {
		return nil, fmt.Errorf("failed to parse tool calls JSON: %w", err)
	}

	// Convert to proper MCP CallToolParams
	for _, rawCall := range rawToolCalls {
		name, ok := rawCall["name"].(string)
		if !ok {
			a.logger.Warn().Interface("tool_call", rawCall).Msg("Invalid tool call: missing name")
			continue
		}

		arguments, ok := rawCall["arguments"]
		if !ok {
			arguments = map[string]interface{}{} // Default to empty arguments
		}

		toolCall := &mcp.CallToolParams{
			Name:      name,
			Arguments: arguments,
		}
		toolCalls = append(toolCalls, toolCall)
	}

	return toolCalls, nil
}

// executeTools executes the provided tool calls using the MCP client
func (a *Agent) executeTools(ctx context.Context, toolCalls []*mcp.CallToolParams) ([]*mcp.CallToolResult, error) {
	var results []*mcp.CallToolResult

	for _, toolCall := range toolCalls {
		a.logger.Info().
			Str("tool_name", toolCall.Name).
			Interface("arguments", toolCall.Arguments).
			Msg("Executing tool")

		result, err := a.mcpClient.CallTool(ctx, toolCall)
		if err != nil {
			a.logger.Error().
				Err(err).
				Str("tool_name", toolCall.Name).
				Msg("Tool execution failed")
			// Continue with other tools even if one fails
			continue
		}

		results = append(results, result)
		a.logger.Info().
			Str("tool_name", toolCall.Name).
			Bool("is_error", result.IsError).
			Msg("Tool execution completed")

		a.logger.Info().
			Str("tool_name", toolCall.Name).
			Str("structured_content", fmt.Sprintf("%+v", result.StructuredContent)).
			Msg("Tool result")

	}

	return results, nil
}

// generateFinalResponse creates a final response based on tool results
func (a *Agent) generateFinalResponse(ctx context.Context, originalQuery string, toolResults []*mcp.CallToolResult) (string, error) {
	// Build context from tool results
	resultsContext := a.buildResultsContext(toolResults)

	systemPrompt := `You are a helpful AI assistant. Based on the tool execution results, provide a clear and helpful response to the user's query. 

Key guidelines:
- Be concise but comprehensive
- If tools executed successfully, incorporate their results into your response
- If tools failed, acknowledge the issue and provide what information you can
- Format your response in a user-friendly way`

	userPrompt := fmt.Sprintf(`Original user query: %s

Tool execution results:
%s

Please provide a helpful response based on the above information.`, originalQuery, resultsContext)

	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
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
		Temperature: float32(a.config.Temperature),
	})

	if err != nil {
		return "", fmt.Errorf("LLM request failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response from LLM")
	}

	return resp.Choices[0].Message.Content, nil
}

// generateDirectResponse creates a response when no tools are needed
func (a *Agent) generateDirectResponse(ctx context.Context, query string) (string, error) {
	systemPrompt := `You are a helpful AI assistant. The user has asked a question that doesn't require any specific tools. Provide a helpful and informative response based on your knowledge.`

	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: a.config.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: query,
			},
		},
		MaxTokens:   a.config.MaxTokens,
		Temperature: float32(a.config.Temperature),
	})

	if err != nil {
		return "", fmt.Errorf("LLM request failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response from LLM")
	}

	return resp.Choices[0].Message.Content, nil
}

// buildToolsDescription creates a description of available tools for the LLM
func (a *Agent) buildToolsDescription() string {
	if len(a.tools) == 0 {
		return "No tools available."
	}

	var descriptions []string
	for _, tool := range a.tools {
		desc := fmt.Sprintf("- %s: %s", tool.Name, tool.Description)

		// Include input schema if available
		if tool.InputSchema != nil {
			// Convert the schema to JSON for the LLM to understand
			schemaBytes, err := json.MarshalIndent(tool.InputSchema, "  ", "  ")
			if err == nil {
				desc += fmt.Sprintf("\n  Input Schema: %s", string(schemaBytes))
			} else {
				// Fallback: try to extract basic info from schema
				desc += a.extractSchemaInfo(tool.InputSchema)
			}
		}

		descriptions = append(descriptions, desc)
	}

	return strings.Join(descriptions, "\n")
}

// extractSchemaInfo extracts basic parameter information from the schema
func (a *Agent) extractSchemaInfo(schema *jsonschema.Schema) string {
	if schema == nil {
		return ""
	}

	var info strings.Builder
	info.WriteString("\n  Parameters:")

	if schema.Properties != nil {
		for propName, propSchema := range schema.Properties {
			info.WriteString(fmt.Sprintf("\n    - %s", propName))

			if propSchema.Type != "" {
				info.WriteString(fmt.Sprintf(" (%s)", propSchema.Type))
			}

			if propSchema.Description != "" {
				info.WriteString(fmt.Sprintf(": %s", propSchema.Description))
			}

			// Check if it's required
			if schema.Required != nil {
				for _, reqField := range schema.Required {
					if reqField == propName {
						info.WriteString(" [required]")
						break
					}
				}
			}
		}
	}

	return info.String()
}

// buildResultsContext creates a formatted string from tool results
func (a *Agent) buildResultsContext(results []*mcp.CallToolResult) string {
	if len(results) == 0 {
		return "No tool results available."
	}

	var contexts []string
	for i, result := range results {
		var resultText string
		if result.IsError {
			resultText = "Tool execution failed"
		} else {
			// Extract text content from the result
			if len(result.Content) > 0 {
				for _, content := range result.Content {
					if textContent, ok := content.(*mcp.TextContent); ok {
						resultText = textContent.Text
						break
					}
				}
			}
			if resultText == "" {
				resultText = "Tool executed successfully but no text content available"
			}
		}

		contexts = append(contexts, fmt.Sprintf("Tool %d result:\n%s", i+1, resultText))
	}

	return strings.Join(contexts, "\n\n")
}

// GetAvailableTools returns the list of available tools
func (a *Agent) GetAvailableTools() []*mcp.Tool {
	return a.tools
}
