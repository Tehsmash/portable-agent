package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	maxReadLines = 2000
	maxReadBytes = 50 * 1024 // 50 KB
)

// Tool is the interface every tool must satisfy.
type Tool interface {
	Definition() ToolDef
	Execute(ctx context.Context, input json.RawMessage) (string, error)
}

// Registry holds the set of tools available to the agent loop.
type Registry struct {
	tools map[string]Tool
}

// NewRegistry creates an empty tool registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds a tool to the registry.
func (r *Registry) Register(t Tool) {
	r.tools[t.Definition().Name] = t
}

// Get looks up a tool by name.
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Definitions returns all tool definitions for passing to the LLM.
func (r *Registry) Definitions() []ToolDef {
	defs := make([]ToolDef, 0, len(r.tools))
	for _, t := range r.tools {
		defs = append(defs, t.Definition())
	}
	return defs
}

// resolvePath resolves a path relative to the current working directory.
func resolvePath(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	cwd, err := os.Getwd()
	if err != nil {
		return p
	}
	return filepath.Join(cwd, p)
}

// ── ReadTool ──────────────────────────────────────────────────────────────────

type readInput struct {
	Path   string `json:"path"`
	Offset *int   `json:"offset"`
	Limit  *int   `json:"limit"`
}

// ReadTool reads a file's contents with optional line offset and limit.
type ReadTool struct{}

var readSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to the file to read (relative or absolute)"
    },
    "offset": {
      "type": "number",
      "description": "Line number to start reading from (1-indexed)"
    },
    "limit": {
      "type": "number",
      "description": "Maximum number of lines to read"
    }
  },
  "required": ["path"]
}`)

func (ReadTool) Definition() ToolDef {
	return ToolDef{
		Name:        "read",
		Description: "Read a file's contents. Supports optional line offset and limit for large files. Output is capped at 2000 lines / 50 KB; use offset to page through longer files.",
		InputSchema: readSchema,
	}
}

func (ReadTool) Execute(_ context.Context, input json.RawMessage) (string, error) {
	var ri readInput
	if err := json.Unmarshal(input, &ri); err != nil {
		return "", fmt.Errorf("read: invalid input: %w", err)
	}
	if ri.Path == "" {
		return "", fmt.Errorf("read: path is required")
	}

	data, err := os.ReadFile(resolvePath(ri.Path))
	if err != nil {
		return "", fmt.Errorf("read: %w", err)
	}

	lines := strings.Split(string(data), "\n")
	totalLines := len(lines)

	// Apply 1-indexed offset.
	start := 0
	if ri.Offset != nil {
		start = *ri.Offset - 1
		if start < 0 {
			start = 0
		}
		if start >= totalLines {
			return "", fmt.Errorf("read: offset %d is beyond end of file (%d lines)", *ri.Offset, totalLines)
		}
	}
	lines = lines[start:]

	// Apply caller-requested line limit before the hard cap.
	if ri.Limit != nil && *ri.Limit < len(lines) {
		lines = lines[:*ri.Limit]
	}

	// Hard cap: head truncation (first N lines / N bytes).
	kept, keptCount := truncateHead(lines, maxReadLines, maxReadBytes)
	result := strings.Join(kept, "\n")

	if keptCount < len(lines) {
		absEnd := start + keptCount
		result += fmt.Sprintf("\n[Showing lines %d–%d of %d. Use offset=%d to continue.]",
			start+1, absEnd, totalLines, absEnd+1)
	}

	return result, nil
}

// truncateHead returns the first lines up to maxL lines or maxB bytes,
// and the count of lines kept.
func truncateHead(lines []string, maxL, maxB int) ([]string, int) {
	byteCount := 0
	for i, line := range lines {
		if i >= maxL {
			return lines[:i], i
		}
		byteCount += len(line) + 1
		if byteCount > maxB {
			return lines[:i], i
		}
	}
	return lines, len(lines)
}

// ── BashTool ──────────────────────────────────────────────────────────────────

type bashInput struct {
	Command string `json:"command"`
	Timeout *int   `json:"timeout"`
}

// BashTool executes bash commands with optional timeout.
type BashTool struct{}

var bashSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "Bash command to execute"
    },
    "timeout": {
      "type": "number",
      "description": "Timeout in seconds (optional)"
    }
  },
  "required": ["command"]
}`)

func (BashTool) Definition() ToolDef {
	return ToolDef{
		Name:        "bash",
		Description: "Execute a bash command and return its combined stdout/stderr output. The last 2000 lines / 50 KB of output are returned when truncated.",
		InputSchema: bashSchema,
	}
}

func (BashTool) Execute(ctx context.Context, input json.RawMessage) (string, error) {
	var bi bashInput
	if err := json.Unmarshal(input, &bi); err != nil {
		return "", fmt.Errorf("bash: invalid input: %w", err)
	}
	if bi.Command == "" {
		return "", fmt.Errorf("bash: command is required")
	}

	if bi.Timeout != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(*bi.Timeout)*time.Second)
		defer cancel()
	}

	cmd := exec.CommandContext(ctx, "bash", "-c", bi.Command)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	runErr := cmd.Run()
	output := buf.String()

	// Tail truncation: show the end of output (most relevant for errors/results).
	lines := strings.Split(output, "\n")
	kept, startIdx := truncateTail(lines, maxReadLines, maxReadBytes)
	result := strings.Join(kept, "\n")
	if startIdx > 0 {
		result = fmt.Sprintf("[Output truncated, showing lines %d–%d of %d]\n", startIdx+1, len(lines), len(lines)) + result
	}

	if runErr != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return result, fmt.Errorf("bash: command timed out after %d seconds", *bi.Timeout)
		}
		exitCode := -1
		if cmd.ProcessState != nil {
			exitCode = cmd.ProcessState.ExitCode()
		}
		// Return output alongside exit code so the LLM can see what went wrong.
		return fmt.Sprintf("%s\nCommand exited with code %d", result, exitCode), nil
	}
	return result, nil
}

// truncateTail returns the last lines up to maxL lines or maxB bytes,
// and the start index in the original slice.
func truncateTail(lines []string, maxL, maxB int) ([]string, int) {
	start := len(lines)
	byteCount := 0
	for start > 0 {
		candidate := start - 1
		if len(lines)-candidate > maxL {
			break
		}
		byteCount += len(lines[candidate]) + 1
		if byteCount > maxB {
			break
		}
		start = candidate
	}
	return lines[start:], start
}

// ── EditTool ──────────────────────────────────────────────────────────────────

type editItem struct {
	OldText string `json:"old_text"`
	NewText string `json:"new_text"`
}

type editInput struct {
	Path  string     `json:"path"`
	Edits []editItem `json:"edits"`
}

type editMatch struct {
	pos  int
	item editItem
}

// EditTool applies exact-string replacements to a file.
type EditTool struct{}

var editSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to the file to edit (relative or absolute)"
    },
    "edits": {
      "type": "array",
      "description": "One or more targeted replacements. Each old_text must appear exactly once in the file and must not overlap with other edits. Edits are matched against the original content, not applied incrementally.",
      "items": {
        "type": "object",
        "properties": {
          "old_text": {
            "type": "string",
            "description": "Exact text to replace. Must appear exactly once in the file."
          },
          "new_text": {
            "type": "string",
            "description": "Replacement text."
          }
        },
        "required": ["old_text", "new_text"]
      }
    }
  },
  "required": ["path", "edits"]
}`)

func (EditTool) Definition() ToolDef {
	return ToolDef{
		Name:        "edit",
		Description: "Edit a file by replacing exact text strings. Each old_text must appear exactly once in the file. Multiple non-overlapping edits can be applied in a single call.",
		InputSchema: editSchema,
	}
}

func (EditTool) Execute(_ context.Context, input json.RawMessage) (string, error) {
	var ei editInput
	if err := json.Unmarshal(input, &ei); err != nil {
		return "", fmt.Errorf("edit: invalid input: %w", err)
	}
	if ei.Path == "" {
		return "", fmt.Errorf("edit: path is required")
	}
	if len(ei.Edits) == 0 {
		return "", fmt.Errorf("edit: edits array must be non-empty")
	}

	data, err := os.ReadFile(resolvePath(ei.Path))
	if err != nil {
		return "", fmt.Errorf("edit: read file: %w", err)
	}
	content := string(data)

	// Find each old_text; validate it appears exactly once.
	matches := make([]editMatch, 0, len(ei.Edits))
	for _, e := range ei.Edits {
		if e.OldText == "" {
			return "", fmt.Errorf("edit: old_text must not be empty")
		}
		count := strings.Count(content, e.OldText)
		switch count {
		case 0:
			return "", fmt.Errorf("edit: old_text not found in file:\n%s", e.OldText)
		case 1:
			matches = append(matches, editMatch{pos: strings.Index(content, e.OldText), item: e})
		default:
			return "", fmt.Errorf("edit: old_text appears %d times (must be unique):\n%s", count, e.OldText)
		}
	}

	// Sort matches by position (insertion sort; N is typically small).
	for i := 1; i < len(matches); i++ {
		for j := i; j > 0 && matches[j].pos < matches[j-1].pos; j-- {
			matches[j], matches[j-1] = matches[j-1], matches[j]
		}
	}

	// Reject overlapping edits.
	for i := 1; i < len(matches); i++ {
		prev, curr := matches[i-1], matches[i]
		if prev.pos+len(prev.item.OldText) > curr.pos {
			return "", fmt.Errorf("edit: edits overlap: %q and %q", prev.item.OldText, curr.item.OldText)
		}
	}

	// Apply in reverse order so earlier positions remain stable.
	for i := len(matches) - 1; i >= 0; i-- {
		m := matches[i]
		content = content[:m.pos] + m.item.NewText + content[m.pos+len(m.item.OldText):]
	}

	if err := os.WriteFile(resolvePath(ei.Path), []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("edit: write file: %w", err)
	}

	return fmt.Sprintf("Applied %d edit(s) to %s", len(ei.Edits), ei.Path), nil
}

// ── WriteTool ─────────────────────────────────────────────────────────────────

type writeInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// WriteTool writes content to a file, creating parent directories as needed.
type WriteTool struct{}

var writeSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to the file to write (relative or absolute)"
    },
    "content": {
      "type": "string",
      "description": "Content to write to the file"
    }
  },
  "required": ["path", "content"]
}`)

func (WriteTool) Definition() ToolDef {
	return ToolDef{
		Name:        "write",
		Description: "Write content to a file. Creates parent directories automatically. Overwrites any existing content.",
		InputSchema: writeSchema,
	}
}

func (WriteTool) Execute(_ context.Context, input json.RawMessage) (string, error) {
	var wi writeInput
	if err := json.Unmarshal(input, &wi); err != nil {
		return "", fmt.Errorf("write: invalid input: %w", err)
	}
	if wi.Path == "" {
		return "", fmt.Errorf("write: path is required")
	}

	path := resolvePath(wi.Path)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", fmt.Errorf("write: create directories: %w", err)
	}
	if err := os.WriteFile(path, []byte(wi.Content), 0o644); err != nil {
		return "", fmt.Errorf("write: %w", err)
	}

	return fmt.Sprintf("Wrote %d bytes to %s", len(wi.Content), wi.Path), nil
}
