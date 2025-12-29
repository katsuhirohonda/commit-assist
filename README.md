# commit-assist

AI-powered git commit message generator using LLM APIs.

## Features

- Analyzes staged git changes (`git diff --staged`)
- Prompts LLM to generate [Conventional Commits](https://www.conventionalcommits.org/) format messages
- Supports multiple LLM providers (Claude, OpenAI)

## Installation

```bash
cargo install commit-assist
```

Or build from source:

```bash
git clone https://github.com/katsuhirohonda/commit-assist.git
cd commit-assist
cargo build --release
```

## Usage

### Basic Usage

```bash
# Stage your changes
git add .

# Generate commit message using Claude (default)
commit-assist
```

### Options

```bash
# Use OpenAI instead of Claude
commit-assist --provider openai

# Specify a model
commit-assist --model claude-3-5-sonnet-20241022

# Short flags
commit-assist -p openai -m gpt-4-turbo
```

### Environment Variables

Set the API key for your chosen provider:

```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"
```

## Commit Message Format

Generated messages follow Conventional Commits format:

```
<type>(<scope>): <description>

[optional body]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Development

```bash
# Run tests
cargo test

# Generate documentation
cargo doc --open

# Build release
cargo build --release
```

## License

MIT License - see [LICENSE](LICENSE) for details.
