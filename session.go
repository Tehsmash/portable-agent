package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
)

const (
	inboxSize     = 100
	resultBufSize = 32
	maxIterations = 10
)

// sessionResult is one event or error delivered to an Execute caller.
type sessionResult struct {
	event a2a.Event
	err   error
}

// pendingTask is a single queued A2A message waiting to be processed by the session.
type pendingTask struct {
	ctx      context.Context // cancelled if the A2A client disconnects
	execCtx  *a2asrv.ExecutorContext
	resultCh chan sessionResult // buffered; closed when task is done
}

// Session is a long-lived per-contextID agent: one goroutine, one inbox, one history.
// New messages arriving while the agent is processing queue up in inbox.
// Between LLM iterations, injectPending drains the inbox, allowing the agent to
// pivot to a newer instruction without discarding the work done so far.
type Session struct {
	contextID string
	config    *Config
	provider  Provider
	registry  *Registry
	logger    *slog.Logger

	inbox   chan *pendingTask // buffered; producers never block unless full
	history []Message        // exclusively owned by the run goroutine; no lock needed
	done    chan struct{}     // closed to stop the run goroutine
}

// NewSession creates a Session and starts its background goroutine.
func NewSession(contextID string, cfg *Config, p Provider, r *Registry, logger *slog.Logger) *Session {
	s := &Session{
		contextID: contextID,
		config:    cfg,
		provider:  p,
		registry:  r,
		logger:    logger,
		inbox:     make(chan *pendingTask, inboxSize),
		done:      make(chan struct{}),
	}
	go s.run()
	return s
}

// Enqueue queues a pending task and returns the channel the caller should read events from.
// If the inbox is full the task is rejected immediately with an error.
func (s *Session) Enqueue(ctx context.Context, execCtx *a2asrv.ExecutorContext) <-chan sessionResult {
	task := &pendingTask{
		ctx:      ctx,
		execCtx:  execCtx,
		resultCh: make(chan sessionResult, resultBufSize),
	}
	select {
	case s.inbox <- task:
	default:
		task.resultCh <- sessionResult{err: errors.New("session inbox full")}
		close(task.resultCh)
	}
	return task.resultCh
}

// Stop signals the run goroutine to exit.
func (s *Session) Stop() {
	select {
	case <-s.done:
	default:
		close(s.done)
	}
}

// run is the session's background goroutine. It processes one task at a time;
// additional tasks accumulate in inbox until the current one finishes (or is superseded).
// A case for scheduled/proactive work can be added here without touching any other code.
func (s *Session) run() {
	for {
		select {
		case task := <-s.inbox:
			s.processTask(task)
		case <-s.done:
			return
		}
	}
}

// processTask drives one pending task through the full agent loop.
// The task pointer may be redirected mid-loop by injectPending if a newer message arrives.
func (s *Session) processTask(task *pendingTask) {
	// current is a pointer we pass by address so injectPending can redirect it.
	current := task

	defer func() {
		// Always close the final active task's channel when we're done.
		close(current.resultCh)
	}()

	// Emit the submitted task event.
	if !emit(current, a2a.NewSubmittedTask(current.execCtx, current.execCtx.Message)) {
		return
	}

	// Add the incoming user message to conversation history.
	userText := extractText(current.execCtx.Message)
	s.history = append(s.history, Message{Role: RoleUser, Parts: []Part{{Kind: PartKindText, Text: userText}}})

	// Signal we are now working.
	if !emit(current, a2a.NewStatusUpdateEvent(current.execCtx, a2a.TaskStateWorking, nil)) {
		return
	}

	// Run the agentic loop. current may be updated if the user changes their mind.
	reply, err := s.runLoop(&current)
	if err != nil {
		emit(current, a2a.NewStatusUpdateEvent(
			current.execCtx,
			a2a.TaskStateFailed,
			a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(err.Error())),
		))
		return
	}

	emit(current, a2a.NewStatusUpdateEvent(
		current.execCtx,
		a2a.TaskStateCompleted,
		a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart(reply)),
	))
}

// runLoop runs the LLM → tool-use → LLM cycle.
// Before each LLM call it checks for newer queued messages and redirects current if found.
// The pointer-to-pointer lets us replace the active task without the caller needing to know.
func (s *Session) runLoop(current **pendingTask) (string, error) {
	tools := s.registry.Definitions()

	for i := 0; i < maxIterations; i++ {
		// ── Interrupt check ─────────────────────────────────────────────────────
		// Non-blocking drain: if the user sent a new message while we were
		// executing tools, inject it into the history and redirect current.
		s.injectPending(current)
		// ────────────────────────────────────────────────────────────────────────

		// Respect client disconnection.
		if err := (*current).ctx.Err(); err != nil {
			return "", err
		}

		resp, err := s.provider.Complete((*current).ctx, s.config.SystemPrompt, s.history, tools)
		if err != nil {
			return "", fmt.Errorf("LLM call failed: %w", err)
		}

		// Build the assistant turn from the response.
		var assistantParts []Part
		for _, p := range resp.Parts {
			assistantParts = append(assistantParts, p)
		}
		s.history = append(s.history, Message{Role: RoleAssistant, Parts: assistantParts})

		// No tool calls → we have a final answer.
		if resp.StopReason != "tool_use" {
			// Collect all text from the response.
			var sb strings.Builder
			for _, p := range resp.Parts {
				if p.Kind == PartKindText {
					sb.WriteString(p.Text)
				}
			}
			return sb.String(), nil
		}

		// Execute tool calls and collect results.
		var resultParts []Part
		for _, p := range resp.Parts {
			if p.Kind != PartKindToolUse {
				continue
			}
			tool, ok := s.registry.Get(p.ToolUseName)
			if !ok {
				resultParts = append(resultParts, Part{
					Kind:            PartKindToolResult,
					ToolResultID:    p.ToolUseID,
					ToolResultValue: fmt.Sprintf("unknown tool: %s", p.ToolUseName),
					ToolResultError: true,
				})
				continue
			}
			output, execErr := tool.Execute((*current).ctx, p.ToolInput)
			rp := Part{
				Kind:            PartKindToolResult,
				ToolResultID:    p.ToolUseID,
				ToolResultValue: output,
				ToolResultError: execErr != nil,
			}
			if execErr != nil && output == "" {
				rp.ToolResultValue = execErr.Error()
			}
			resultParts = append(resultParts, rp)
		}

		s.history = append(s.history, Message{Role: RoleUser, Parts: resultParts})
	}

	return "", errors.New("max iterations reached")
}

// injectPending does a non-blocking drain of the inbox.
// For each queued message it:
//  1. Cancels the current task with TaskStateCanceled (closes its resultCh).
//  2. Appends the new message to the shared history.
//  3. Emits the task lifecycle events on the new task's resultCh.
//  4. Redirects *current to the new task.
//
// After return, *current points to the most recently queued task.
// The LLM will see the full accumulated history on the next call.
func (s *Session) injectPending(current **pendingTask) {
	for {
		select {
		case next := <-s.inbox:
			s.logger.Info("mid-loop interrupt: newer message arrived, superseding current task",
				"old_task", (*current).execCtx.TaskID,
				"new_task", next.execCtx.TaskID,
			)

			// Close the superseded task with a cancellation terminal event.
			(*current).resultCh <- sessionResult{
				event: a2a.NewStatusUpdateEvent((*current).execCtx, a2a.TaskStateCanceled, nil),
			}
			close((*current).resultCh)

			// Inject the new user message into history.
			userText := extractText(next.execCtx.Message)
			s.history = append(s.history, Message{
				Role:  RoleUser,
				Parts: []Part{{Kind: PartKindText, Text: userText}},
			})

			// Emit lifecycle events for the new task.
			next.resultCh <- sessionResult{event: a2a.NewSubmittedTask(next.execCtx, next.execCtx.Message)}
			next.resultCh <- sessionResult{event: a2a.NewStatusUpdateEvent(next.execCtx, a2a.TaskStateWorking, nil)}

			// Redirect: all future events go to the new task's channel.
			*current = next

		default:
			return
		}
	}
}

// emit sends an event on the task's result channel.
// Returns false if the channel is full and the event was dropped (shouldn't happen with buffered ch).
func emit(task *pendingTask, event a2a.Event) bool {
	select {
	case task.resultCh <- sessionResult{event: event}:
		return true
	case <-task.ctx.Done():
		return false
	}
}

// extractText concatenates all text parts from an A2A message.
func extractText(msg *a2a.Message) string {
	if msg == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range msg.Parts {
		if t := p.Text(); t != "" {
			sb.WriteString(t)
		}
	}
	return sb.String()
}
