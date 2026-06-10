package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// SkillEntry holds the parsed metadata and path for a single AgentSkill.
type SkillEntry struct {
	Name        string
	Description string
	Path        string // absolute path to SKILL.md
}

// DiscoverSkills scans each directory in dirs for AgentSkills-compatible skill
// directories (subdirs containing a SKILL.md file). Project-level dirs should
// be listed before user-level dirs; the first occurrence of a skill name wins.
func DiscoverSkills(dirs []string) []SkillEntry {
	seen := make(map[string]struct{})
	var skills []SkillEntry

	for _, dir := range dirs {
		expanded := expandTilde(dir)
		abs, err := filepath.Abs(expanded)
		if err != nil {
			continue
		}
		entries := scanDir(abs, 0)
		for _, e := range entries {
			if _, exists := seen[e.Name]; !exists {
				seen[e.Name] = struct{}{}
				skills = append(skills, e)
			}
		}
	}

	return skills
}

// scanDir recursively walks dir up to maxDepth looking for subdirectories that
// contain a SKILL.md file.
const maxSkillDepth = 5

func scanDir(dir string, depth int) []SkillEntry {
	if depth > maxSkillDepth {
		return nil
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}

	var skills []SkillEntry

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if name == ".git" || name == "node_modules" {
			continue
		}

		subdir := filepath.Join(dir, name)
		skillFile := filepath.Join(subdir, "SKILL.md")

		if _, err := os.Stat(skillFile); err == nil {
			skill, err := parseSkillFile(skillFile)
			if err != nil {
				// Warn but skip.
				fmt.Fprintf(os.Stderr, "skills: skipping %s: %v\n", skillFile, err)
				continue
			}
			if skill.Name != name {
				fmt.Fprintf(os.Stderr, "skills: warning: skill name %q does not match directory %q\n", skill.Name, name)
			}
			skills = append(skills, *skill)
		} else {
			// Recurse into subdirectory.
			skills = append(skills, scanDir(subdir, depth+1)...)
		}
	}

	return skills
}

// skillFrontmatter is used only for YAML unmarshalling.
type skillFrontmatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// parseSkillFile reads a SKILL.md file and extracts the name and description
// from the YAML frontmatter block.
func parseSkillFile(path string) (*SkillEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}

	content := string(data)

	// Find opening ---
	const delim = "---"
	start := strings.Index(content, delim)
	if start == -1 {
		return nil, fmt.Errorf("%s: no YAML frontmatter found", path)
	}
	rest := content[start+len(delim):]

	// Find closing ---
	end := strings.Index(rest, delim)
	if end == -1 {
		return nil, fmt.Errorf("%s: unclosed YAML frontmatter", path)
	}
	yamlBlock := rest[:end]

	var fm skillFrontmatter
	if err := yaml.Unmarshal([]byte(yamlBlock), &fm); err != nil {
		return nil, fmt.Errorf("%s: parsing frontmatter: %w", path, err)
	}

	if fm.Name == "" {
		return nil, fmt.Errorf("%s: missing required field 'name'", path)
	}
	if fm.Description == "" {
		return nil, fmt.Errorf("%s: missing required field 'description'", path)
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}

	return &SkillEntry{
		Name:        fm.Name,
		Description: fm.Description,
		Path:        abs,
	}, nil
}

// BuildSkillsCatalog returns an XML block listing all available skills, or an
// empty string when the slice is empty.
func BuildSkillsCatalog(skills []SkillEntry) string {
	if len(skills) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("<available_skills>\n")
	for _, s := range skills {
		sb.WriteString("  <skill>\n")
		fmt.Fprintf(&sb, "    <name>%s</name>\n", s.Name)
		fmt.Fprintf(&sb, "    <description>%s</description>\n", s.Description)
		fmt.Fprintf(&sb, "    <location>%s</location>\n", s.Path)
		sb.WriteString("  </skill>\n")
	}
	sb.WriteString("</available_skills>")
	return sb.String()
}

// expandTilde replaces a leading "~" with the user's home directory.
func expandTilde(path string) string {
	if path == "~" {
		if home, err := os.UserHomeDir(); err == nil {
			return home
		}
	}
	if strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}
