package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/hybridgroup/yzma/pkg/template"
)

// LocalProvider implements Provider using llama.cpp via the yzma library.
// Model weights are loaded once at construction; a fresh llama context is
// created for every Complete call to avoid KV-cache state leaking between turns.
type LocalProvider struct {
	model llama.Model
	cfg   LocalModelConfig
}

// NewLocalProvider loads the llama.cpp shared libraries and the GGUF model file.
// Call Close() when the provider is no longer needed.
func NewLocalProvider(cfg LocalModelConfig) (*LocalProvider, error) {
	if err := llama.Load(cfg.LibDir); err != nil {
		return nil, fmt.Errorf("loading llama.cpp libraries from %q: %w", cfg.LibDir, err)
	}

	llama.Init()

	mParams := llama.ModelDefaultParams()
	mParams.NGpuLayers = cfg.GPULayers

	model, err := llama.ModelLoadFromFile(cfg.Path, mParams)
	if err != nil {
		return nil, fmt.Errorf("loading model %q: %w", cfg.Path, err)
	}

	return &LocalProvider{model: model, cfg: cfg}, nil
}

// Close frees the model and unloads the llama.cpp backend.
func (p *LocalProvider) Close() {
	_ = llama.ModelFree(p.model)
	llama.Close()
}

// Complete runs inference against the local model and returns the response.
func (p *LocalProvider) Complete(ctx context.Context, system string, msgs []Message, tools []ToolDef) (*Response, error) {
	// --- Build yzma message slice ---
	var ymsgs []message.Message

	if system != "" {
		ymsgs = append(ymsgs, message.Chat{Role: "system", Content: system})
	}

	// Prepend tool definitions so the model knows which tools are available.
	if len(tools) > 0 {
		for _, t := range tools {
			var params map[string]interface{}
			if err := json.Unmarshal(t.InputSchema, &params); err != nil {
				return nil, fmt.Errorf("tool %q: invalid input schema: %w", t.Name, err)
			}
			ymsgs = append(ymsgs, message.ToolDefinition{
				Type: "function",
				Function: message.ToolFunctionDefinition{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  params,
				},
			})
		}
	}

	for _, m := range msgs {
		roleStr := string(m.Role)
		for _, part := range m.Parts {
			switch part.Kind {
			case PartKindText:
				ymsgs = append(ymsgs, message.Chat{Role: roleStr, Content: part.Text})

			case PartKindToolUse:
				var argMap map[string]string
				// ToolFunction.Arguments is map[string]string, so stringify values.
				var raw map[string]interface{}
				if err := json.Unmarshal(part.ToolInput, &raw); err == nil {
					argMap = make(map[string]string, len(raw))
					for k, v := range raw {
						switch sv := v.(type) {
						case string:
							argMap[k] = sv
						default:
							b, _ := json.Marshal(v)
							argMap[k] = string(b)
						}
					}
				}
				ymsgs = append(ymsgs, message.Tool{
					Role: "assistant",
					ToolCalls: []message.ToolCall{
						{
							Type: "function",
							Function: message.ToolFunction{
								Name:      part.ToolUseName,
								Arguments: argMap,
							},
						},
					},
				})

			case PartKindToolResult:
				// Find the tool name by scanning history for the matching ToolUse ID.
				toolName := part.ToolResultID
				for _, hm := range msgs {
					for _, hp := range hm.Parts {
						if hp.Kind == PartKindToolUse && hp.ToolUseID == part.ToolResultID {
							toolName = hp.ToolUseName
						}
					}
				}
				ymsgs = append(ymsgs, message.ToolResponse{
					Role:    "tool",
					Name:    toolName,
					Content: part.ToolResultValue,
				})
			}
		}
	}

	// --- Apply chat template ---
	tmpl := llama.ModelChatTemplate(p.model, "")
	prompt, err := template.Apply(tmpl, ymsgs, true)
	if err != nil {
		return nil, fmt.Errorf("applying chat template: %w", err)
	}

	// --- Tokenize ---
	vocab := llama.ModelGetVocab(p.model)
	tokens := llama.Tokenize(vocab, prompt, true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenization produced no tokens")
	}

	// --- Create fresh context ---
	cParams := llama.ContextDefaultParams()
	cParams.NCtx = p.cfg.ContextSize
	cParams.NBatch = 512
	if p.cfg.Threads > 0 {
		cParams.NThreads = p.cfg.Threads
		cParams.NThreadsBatch = p.cfg.Threads
	}

	lctx, err := llama.InitFromModel(p.model, cParams)
	if err != nil {
		return nil, fmt.Errorf("creating llama context: %w", err)
	}
	defer func() { _ = llama.Free(lctx) }()

	// --- Set up sampler ---
	samplerParams := llama.DefaultSamplerParams()
	samplerParams.Temp = p.cfg.Temperature
	sampler := llama.NewSampler(p.model, llama.DefaultSamplers, samplerParams)
	defer llama.SamplerFree(sampler)

	// --- Generate tokens ---
	// First batch is the full prompt; after the first decode+sample we switch to
	// one-token batches, matching the pattern in the yzma chat example.
	batch := llama.BatchGetOne(tokens)
	var generated []llama.Token

	for i := 0; i < p.cfg.MaxTokens; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if _, decErr := llama.Decode(lctx, batch); decErr != nil {
			return nil, fmt.Errorf("decode: %w", decErr)
		}

		token := llama.SamplerSample(sampler, lctx, -1)
		llama.SamplerAccept(sampler, token)

		if llama.VocabIsEOG(vocab, token) {
			break
		}

		generated = append(generated, token)
		batch = llama.BatchGetOne([]llama.Token{token})
	}

	// --- Detokenize ---
	output := llama.Detokenize(vocab, generated, false, true)

	// --- Parse tool calls ---
	calls := message.ParseToolCalls(output)

	result := &Response{}
	if len(calls) > 0 {
		result.StopReason = "tool_use"
		for i, call := range calls {
			inputJSON, _ := json.Marshal(call.Function.Arguments)
			result.Parts = append(result.Parts, Part{
				Kind:        PartKindToolUse,
				ToolUseID:   fmt.Sprintf("call_%d", i),
				ToolUseName: call.Function.Name,
				ToolInput:   inputJSON,
			})
		}
		// Include any trailing plain text alongside the tool calls.
		if plain := message.TextAfterToolCalls(output); plain != "" {
			result.Parts = append(result.Parts, Part{Kind: PartKindText, Text: plain})
		}
	} else {
		result.StopReason = "end_turn"
		result.Parts = []Part{{Kind: PartKindText, Text: output}}
	}

	return result, nil
}
