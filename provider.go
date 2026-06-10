package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Role identifies whether a message comes from the user or the assistant.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// PartKind distinguishes the different content types a message part can carry.
type PartKind int

const (
	PartKindText       PartKind = iota
	PartKindToolUse             // assistant → tool invocation
	PartKindToolResult          // user → result of a tool invocation
)

// Part is one piece of content within a Message.
type Part struct {
	Kind PartKind

	// PartKindText
	Text string

	// PartKindToolUse
	ToolUseID   string
	ToolUseName string
	ToolInput   json.RawMessage

	// PartKindToolResult
	ToolResultID    string
	ToolResultValue string
	ToolResultError bool
}

// Message is an entry in the conversation history.
type Message struct {
	Role  Role
	Parts []Part
}

// ToolDef describes a tool the LLM may call.
type ToolDef struct {
	Name        string
	Description string
	InputSchema json.RawMessage // JSON Schema object
}

// Response is the raw LLM output.
type Response struct {
	StopReason string
	Parts      []Part
}

// Provider is the interface for calling an LLM.
type Provider interface {
	Complete(ctx context.Context, system string, messages []Message, tools []ToolDef) (*Response, error)
}

// AnthropicProvider calls the Anthropic Messages API.
type AnthropicProvider struct {
	client *anthropic.Client
	model  anthropic.Model
}

// NewAnthropicProvider creates a provider backed by the Anthropic API.
// If baseURL is non-empty it overrides the default API endpoint, which is
// useful for routing through a proxy such as LiteLLM.
func NewAnthropicProvider(apiKey, model, baseURL string) *AnthropicProvider {
	opts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	client := anthropic.NewClient(opts...)
	return &AnthropicProvider{client: &client, model: anthropic.Model(model)}
}

// Complete sends messages to the Anthropic API and returns the response.
func (p *AnthropicProvider) Complete(ctx context.Context, system string, messages []Message, tools []ToolDef) (*Response, error) {
	params := anthropic.MessageNewParams{
		Model:     p.model,
		MaxTokens: 4096,
	}

	if system != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: system},
		}
	}

	// Convert tools.
	if len(tools) > 0 {
		at := make([]anthropic.ToolUnionParam, 0, len(tools))
		for _, t := range tools {
			// ToolInputSchemaParam expects the inner properties/required fields,
			// not the whole JSON Schema object.
			var s struct {
				Properties interface{} `json:"properties"`
				Required   []string    `json:"required"`
			}
			if err := json.Unmarshal(t.InputSchema, &s); err != nil {
				return nil, fmt.Errorf("tool %q: invalid input schema: %w", t.Name, err)
			}
			tool := anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: s.Properties,
					Required:   s.Required,
				},
			}
			at = append(at, anthropic.ToolUnionParam{OfTool: &tool})
		}
		params.Tools = at
	}

	// Convert messages.
	params.Messages = convertMessages(messages)

	resp, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic API call: %w", err)
	}

	result := &Response{StopReason: string(resp.StopReason)}
	for _, block := range resp.Content {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			result.Parts = append(result.Parts, Part{Kind: PartKindText, Text: v.Text})
		case anthropic.ToolUseBlock:
			input, _ := json.Marshal(v.Input)
			result.Parts = append(result.Parts, Part{
				Kind:        PartKindToolUse,
				ToolUseID:   v.ID,
				ToolUseName: v.Name,
				ToolInput:   input,
			})
		}
	}
	return result, nil
}

// convertMessages converts internal Message slice to Anthropic SDK message params.
func convertMessages(messages []Message) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, 0, len(messages))
	for _, m := range messages {
		var blocks []anthropic.ContentBlockParamUnion
		for _, p := range m.Parts {
			switch p.Kind {
			case PartKindText:
				blocks = append(blocks, anthropic.NewTextBlock(p.Text))
			case PartKindToolUse:
				var input interface{}
				_ = json.Unmarshal(p.ToolInput, &input)
				blocks = append(blocks, anthropic.NewToolUseBlock(p.ToolUseID, input, p.ToolUseName))
			case PartKindToolResult:
				blocks = append(blocks, anthropic.NewToolResultBlock(p.ToolResultID, p.ToolResultValue, p.ToolResultError))
			}
		}
		switch m.Role {
		case RoleUser:
			out = append(out, anthropic.NewUserMessage(blocks...))
		case RoleAssistant:
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		}
	}
	return out
}
