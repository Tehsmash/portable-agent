package main

import (
	"context"
	"iter"
	"log/slog"
	"sync"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
)

// Executor is a thin A2A AgentExecutor. Its only job is to maintain the session
// registry and route incoming Execute calls to the appropriate Session.
//
// All agent logic lives in Session. The Executor never touches conversation
// history or calls the LLM directly.
type Executor struct {
	config   *Config
	provider Provider
	registry *Registry
	logger   *slog.Logger

	mu       sync.Mutex
	sessions map[string]*Session // keyed by A2A contextID
}

// NewExecutor creates an Executor ready to serve A2A requests.
func NewExecutor(cfg *Config, provider Provider, registry *Registry, logger *slog.Logger) *Executor {
	return &Executor{
		config:   cfg,
		provider: provider,
		registry: registry,
		logger:   logger,
		sessions: make(map[string]*Session),
	}
}

// Execute routes the incoming A2A message to the appropriate session and returns
// an iterator that streams events back to the A2A framework as the agent works.
// New messages for the same context that arrive while the agent is processing
// are queued in the session's inbox and handled via the mid-loop interrupt check.
func (e *Executor) Execute(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	session := e.getOrCreate(execCtx.ContextID)
	resultCh := session.Enqueue(ctx, execCtx)

	return func(yield func(a2a.Event, error) bool) {
		for result := range resultCh {
			if result.err != nil {
				yield(nil, result.err)
				return
			}
			if !yield(result.event, nil) {
				// Consumer stopped (client disconnected). The session goroutine
				// will notice ctx.Done() on its next iteration.
				return
			}
		}
	}
}

// Cancel shuts down the session for the given context, emitting a Canceled event.
func (e *Executor) Cancel(ctx context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		e.mu.Lock()
		s, ok := e.sessions[execCtx.ContextID]
		if ok {
			delete(e.sessions, execCtx.ContextID)
		}
		e.mu.Unlock()

		if ok {
			s.Stop()
		}

		yield(a2a.NewStatusUpdateEvent(execCtx, a2a.TaskStateCanceled, nil), nil)
	}
}

// getOrCreate returns the existing session for contextID or creates a new one.
func (e *Executor) getOrCreate(contextID string) *Session {
	e.mu.Lock()
	defer e.mu.Unlock()

	if s, ok := e.sessions[contextID]; ok {
		return s
	}
	s := NewSession(contextID, e.config, e.provider, e.registry, e.logger)
	e.sessions[contextID] = s
	return s
}
