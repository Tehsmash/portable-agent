package main

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config holds the full agent configuration loaded from YAML.
type Config struct {
	Name         string        `yaml:"name"`
	Description  string        `yaml:"description"`
	Version      string        `yaml:"version"`
	SystemPrompt string        `yaml:"system_prompt"`
	Model           string        `yaml:"model"`
	APIKey          string        `yaml:"api_key"`
	ProviderBaseURL string        `yaml:"provider_base_url"` // optional; overrides the default Anthropic API endpoint
	Skills       []SkillConfig    `yaml:"skills"`
	SkillsDirs   []string         `yaml:"skills_dirs"`
	Server       ServerConfig     `yaml:"server"`
	Tools        ToolsConfig      `yaml:"tools"`
	LocalModel   LocalModelConfig `yaml:"local_model"`
}

// LocalModelConfig configures a local GGUF model run via llama.cpp/yzma.
type LocalModelConfig struct {
	Path        string  `yaml:"path"`         // path to GGUF file (required)
	LibDir      string  `yaml:"lib_dir"`      // dir with llama.cpp shared libs; falls back to $YZMA_LIB
	ContextSize uint32  `yaml:"context_size"` // KV context window (default 4096)
	GPULayers   int32   `yaml:"gpu_layers"`   // layers offloaded to GPU (default 0 = CPU-only)
	Threads     int32   `yaml:"threads"`      // CPU threads (default 0 = auto)
	MaxTokens   int     `yaml:"max_tokens"`   // max tokens to generate per call (default 1024)
	Temperature float32 `yaml:"temperature"`  // sampling temperature (default 0.7)
}

// SkillConfig describes a single agent skill exposed via the A2A agent card.
type SkillConfig struct {
	ID          string   `yaml:"id"`
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	Tags        []string `yaml:"tags"`
	Examples    []string `yaml:"examples"`
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	Port    int    `yaml:"port"`
	Host    string `yaml:"host"`
	BaseURL string `yaml:"base_url"`
}

// ToolsConfig toggles built-in tools.
type ToolsConfig struct {
	Bash  bool `yaml:"bash"`
	Read  bool `yaml:"read"`
	Edit  bool `yaml:"edit"`
	Write bool `yaml:"write"`
}

// loadConfig reads the YAML file at path, expands ${ENV_VAR} references, and
// fills in sensible defaults.
func loadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	// Expand environment variables before parsing.
	expanded := os.ExpandEnv(string(raw))

	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	// Defaults.
	if cfg.Model == "" {
		cfg.Model = "claude-opus-4-6"
	}
	if cfg.Server.Port == 0 {
		cfg.Server.Port = 8080
	}
	if cfg.Server.Host == "" {
		cfg.Server.Host = "0.0.0.0"
	}
	if cfg.Server.BaseURL == "" {
		cfg.Server.BaseURL = fmt.Sprintf("http://localhost:%d", cfg.Server.Port)
	}
	if cfg.Version == "" {
		cfg.Version = "0.1.0"
	}

	// API key: config file takes precedence, then environment variable.
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("ANTHROPIC_API_KEY")
	}

	// LocalModel defaults.
	if cfg.LocalModel.ContextSize == 0 {
		cfg.LocalModel.ContextSize = 4096
	}
	if cfg.LocalModel.MaxTokens == 0 {
		cfg.LocalModel.MaxTokens = 1024
	}
	if cfg.LocalModel.Temperature == 0 {
		cfg.LocalModel.Temperature = 0.7
	}
	if cfg.LocalModel.LibDir == "" {
		cfg.LocalModel.LibDir = os.Getenv("YZMA_LIB")
	}

	// AgentSkills discovery paths: project-level dirs first so they take precedence.
	if len(cfg.SkillsDirs) == 0 {
		cfg.SkillsDirs = []string{
			".agents/skills",   // project-level (cross-client)
			".claude/skills",   // project-level (Claude Code compat)
			"~/.agents/skills", // user-level (cross-client)
			"~/.claude/skills", // user-level (Claude Code compat)
		}
	}

	return &cfg, nil
}
