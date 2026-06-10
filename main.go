package main

import (
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
)

func main() {
	configPath := flag.String("config", "agent.yaml", "path to agent config file")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	cfg, err := loadConfig(*configPath)
	if err != nil {
		logger.Error("failed to load config", "path", *configPath, "err", err)
		os.Exit(1)
	}

	if cfg.APIKey == "" {
		logger.Error("no API key: set ANTHROPIC_API_KEY or api_key in config")
		os.Exit(1)
	}

	// Discover AgentSkills and append catalog to system prompt.
	skills := DiscoverSkills(cfg.SkillsDirs)
	if len(skills) > 0 {
		logger.Info("skills discovered", "count", len(skills))
		catalog := BuildSkillsCatalog(skills)
		cfg.SystemPrompt = cfg.SystemPrompt + "\n\n" + catalog + `

When a task matches one of the available skills, use the read tool to load the full instructions from the skill's <location> before proceeding.`
	} else {
		logger.Info("no skills found")
	}

	// Build tool registry.
	registry := NewRegistry()
	if cfg.Tools.Read {
		registry.Register(ReadTool{})
	}
	if cfg.Tools.Edit {
		registry.Register(EditTool{})
	}
	if cfg.Tools.Write {
		registry.Register(WriteTool{})
	}
	if cfg.Tools.Bash {
		registry.Register(BashTool{})
	}

	// Log registered tools so misconfiguration is immediately visible.
	defs := registry.Definitions()
	if len(defs) == 0 {
		logger.Warn("no tools registered; agent will have no tool access")
	} else {
		names := make([]string, len(defs))
		for i, d := range defs {
			names[i] = d.Name
		}
		logger.Info("tools registered", "tools", names)
	}

	// Build provider.
	provider := NewAnthropicProvider(cfg.APIKey, cfg.Model, cfg.ProviderBaseURL)

	// Build A2A agent card.
	agentCard := buildAgentCard(cfg)

	// Wire up executor and A2A request handler.
	executor := NewExecutor(cfg, provider, registry, logger)
	requestHandler := a2asrv.NewHandler(executor, a2asrv.WithLogger(logger))

	// Set up HTTP mux.
	mux := http.NewServeMux()
	mux.Handle("/", a2asrv.NewRESTHandler(requestHandler))
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Error("failed to listen", "addr", addr, "err", err)
		os.Exit(1)
	}

	logger.Info("portable-agent started",
		"name", cfg.Name,
		"model", cfg.Model,
		"addr", ln.Addr().String(),
		"base_url", cfg.Server.BaseURL,
	)

	if err := http.Serve(ln, mux); err != nil {
		logger.Error("server error", "err", err)
		os.Exit(1)
	}
}

// buildAgentCard constructs the A2A agent card from config.
func buildAgentCard(cfg *Config) *a2a.AgentCard {
	skills := make([]a2a.AgentSkill, 0, len(cfg.Skills))
	for _, s := range cfg.Skills {
		skills = append(skills, a2a.AgentSkill{
			ID:          s.ID,
			Name:        s.Name,
			Description: s.Description,
			Tags:        s.Tags,
			Examples:    s.Examples,
		})
	}

	return &a2a.AgentCard{
		Name:        cfg.Name,
		Description: cfg.Description,
		Version:     cfg.Version,
		Skills:      skills,
		Capabilities: a2a.AgentCapabilities{
			Streaming: true,
		},
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(cfg.Server.BaseURL, a2a.TransportProtocolHTTPJSON),
		},
	}
}
