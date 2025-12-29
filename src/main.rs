//! # commit-assist
//!
//! AI-powered git commit message generator.
//!
//! This CLI tool analyzes staged git changes and generates
//! commit messages in Conventional Commits format using LLM APIs.
//!
//! ## Supported Providers
//!
//! - **Claude** (Anthropic) - Default provider
//! - **OpenAI** (GPT models)
//!
//! ## Usage
//!
//! ```bash
//! # Generate commit message using Claude (default)
//! commit-assist
//!
//! # Use OpenAI instead
//! commit-assist --provider openai
//!
//! # Specify a model
//! commit-assist --model claude-3-5-sonnet-20241022
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::process::Command;

/// CLI arguments for commit-assist.
#[derive(Parser, Debug)]
#[command(name = "commit-assist")]
#[command(author, version, about = "AI-powered git commit message generator")]
#[command(long_about = "Analyzes staged git changes and generates commit messages in Conventional Commits format using AI.")]
pub struct Cli {
    /// LLM provider to use for message generation.
    ///
    /// Supported values: "claude", "openai"
    #[arg(short, long, default_value = "claude")]
    pub provider: String,

    /// Specific model to use.
    ///
    /// If not specified, uses the default model for each provider:
    /// - Claude: claude-sonnet-4-20250514
    /// - OpenAI: gpt-4o
    #[arg(short, long)]
    pub model: Option<String>,
}

/// Request body for Claude API.
#[derive(Serialize, Debug)]
pub struct ClaudeRequest {
    /// Model identifier (e.g., "claude-sonnet-4-20250514")
    pub model: String,
    /// Maximum tokens in the response
    pub max_tokens: u32,
    /// Conversation messages
    pub messages: Vec<Message>,
}

/// A message in the conversation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Message {
    /// Role of the message sender ("user", "assistant", or "system")
    pub role: String,
    /// Content of the message
    pub content: String,
}

/// Response from Claude API.
#[derive(Deserialize, Debug)]
pub struct ClaudeResponse {
    /// Content blocks in the response
    pub content: Vec<ContentBlock>,
}

/// A content block in Claude's response.
#[derive(Deserialize, Debug)]
pub struct ContentBlock {
    /// Text content
    pub text: String,
}

/// Request body for OpenAI API.
#[derive(Serialize, Debug)]
pub struct OpenAIRequest {
    /// Model identifier (e.g., "gpt-4o")
    pub model: String,
    /// Conversation messages
    pub messages: Vec<Message>,
}

/// Response from OpenAI API.
#[derive(Deserialize, Debug)]
pub struct OpenAIResponse {
    /// List of completion choices
    pub choices: Vec<Choice>,
}

/// A completion choice from OpenAI.
#[derive(Deserialize, Debug)]
pub struct Choice {
    /// The generated message
    pub message: Message,
}

/// Retrieves the staged diff from git.
///
/// Executes `git diff --staged` and returns the output.
///
/// # Errors
///
/// Returns an error if:
/// - `git diff` command fails to execute
/// - The output contains invalid UTF-8
/// - No changes are staged
///
/// # Examples
///
/// ```no_run
/// let diff = commit_assist::get_staged_diff()?;
/// println!("Staged changes:\n{}", diff);
/// ```
pub fn get_staged_diff() -> Result<String> {
    let output = Command::new("git")
        .args(["diff", "--staged"])
        .output()
        .context("Failed to execute git diff")?;

    if !output.status.success() {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let diff = String::from_utf8(output.stdout).context("Invalid UTF-8 in git diff output")?;

    if diff.trim().is_empty() {
        bail!("No staged changes found. Use 'git add' to stage changes first.");
    }

    Ok(diff)
}

/// Builds the prompt for the LLM to generate a commit message.
///
/// The prompt instructs the LLM to follow Conventional Commits format
/// and includes the diff to analyze.
///
/// # Arguments
///
/// * `diff` - The git diff output to include in the prompt
///
/// # Returns
///
/// A formatted prompt string ready to send to the LLM.
///
/// # Examples
///
/// ```
/// let diff = "+fn hello() {}";
/// let prompt = commit_assist::build_prompt(diff);
/// assert!(prompt.contains("Conventional Commits"));
/// assert!(prompt.contains(diff));
/// ```
pub fn build_prompt(diff: &str) -> String {
    format!(
        r#"Generate a git commit message for the following diff.

Follow the Conventional Commits format:
<type>(<scope>): <description>

[optional body]

Types: feat, fix, docs, style, refactor, test, chore

Rules:
- First line should be max 72 characters
- Use imperative mood ("add" not "added")
- Be concise but descriptive
- Only output the commit message, nothing else

Diff:
```
{}
```"#,
        diff
    )
}

/// Generates a commit message using Claude API.
///
/// Sends the diff to Claude and returns the generated commit message.
///
/// # Arguments
///
/// * `diff` - The git diff to analyze
/// * `model` - The Claude model to use (e.g., "claude-sonnet-4-20250514")
///
/// # Errors
///
/// Returns an error if:
/// - `ANTHROPIC_API_KEY` environment variable is not set
/// - The API request fails
/// - The response cannot be parsed
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> anyhow::Result<()> {
/// let diff = "+fn new_feature() {}";
/// let message = commit_assist::generate_with_claude(diff, "claude-sonnet-4-20250514").await?;
/// println!("Generated message: {}", message);
/// # Ok(())
/// # }
/// ```
pub async fn generate_with_claude(diff: &str, model: &str) -> Result<String> {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY environment variable not set")?;

    let client = reqwest::Client::new();
    let request = ClaudeRequest {
        model: model.to_string(),
        max_tokens: 1024,
        messages: vec![Message {
            role: "user".to_string(),
            content: build_prompt(diff),
        }],
    };

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request)
        .send()
        .await
        .context("Failed to send request to Claude API")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("Claude API error ({}): {}", status, body);
    }

    let claude_response: ClaudeResponse = response
        .json()
        .await
        .context("Failed to parse Claude API response")?;

    claude_response
        .content
        .first()
        .map(|c| c.text.clone())
        .context("No content in Claude API response")
}

/// Generates a commit message using OpenAI API.
///
/// Sends the diff to OpenAI and returns the generated commit message.
///
/// # Arguments
///
/// * `diff` - The git diff to analyze
/// * `model` - The OpenAI model to use (e.g., "gpt-4o")
///
/// # Errors
///
/// Returns an error if:
/// - `OPENAI_API_KEY` environment variable is not set
/// - The API request fails
/// - The response cannot be parsed
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> anyhow::Result<()> {
/// let diff = "+fn new_feature() {}";
/// let message = commit_assist::generate_with_openai(diff, "gpt-4o").await?;
/// println!("Generated message: {}", message);
/// # Ok(())
/// # }
/// ```
pub async fn generate_with_openai(diff: &str, model: &str) -> Result<String> {
    let api_key =
        std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY environment variable not set")?;

    let client = reqwest::Client::new();
    let request = OpenAIRequest {
        model: model.to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant that generates git commit messages."
                    .to_string(),
            },
            Message {
                role: "user".to_string(),
                content: build_prompt(diff),
            },
        ],
    };

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .context("Failed to send request to OpenAI API")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("OpenAI API error ({}): {}", status, body);
    }

    let openai_response: OpenAIResponse = response
        .json()
        .await
        .context("Failed to parse OpenAI API response")?;

    openai_response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .context("No content in OpenAI API response")
}

/// Runs the commit-assist CLI.
///
/// This is the main entry point that:
/// 1. Parses CLI arguments
/// 2. Gets staged diff from git
/// 3. Generates commit message using the specified provider
/// 4. Prints the generated message
pub async fn run(cli: Cli) -> Result<()> {
    let diff = get_staged_diff()?;

    let message = match cli.provider.as_str() {
        "claude" => {
            let model = cli
                .model
                .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());
            generate_with_claude(&diff, &model).await?
        }
        "openai" => {
            let model = cli.model.unwrap_or_else(|| "gpt-4o".to_string());
            generate_with_openai(&diff, &model).await?
        }
        _ => bail!(
            "Unknown provider: {}. Use 'claude' or 'openai'.",
            cli.provider
        ),
    };

    println!("{}", message);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_contains_diff() {
        let diff = "+fn hello() { println!(\"hello\"); }";
        let prompt = build_prompt(diff);

        assert!(prompt.contains(diff));
    }

    #[test]
    fn test_build_prompt_contains_conventional_commits_instructions() {
        let prompt = build_prompt("some diff");

        assert!(prompt.contains("Conventional Commits"));
        assert!(prompt.contains("feat"));
        assert!(prompt.contains("fix"));
        assert!(prompt.contains("docs"));
        assert!(prompt.contains("refactor"));
    }

    #[test]
    fn test_build_prompt_contains_rules() {
        let prompt = build_prompt("some diff");

        assert!(prompt.contains("72 characters"));
        assert!(prompt.contains("imperative mood"));
    }

    #[test]
    fn test_cli_default_provider() {
        let cli = Cli::parse_from(["commit-assist"]);

        assert_eq!(cli.provider, "claude");
        assert!(cli.model.is_none());
    }

    #[test]
    fn test_cli_openai_provider() {
        let cli = Cli::parse_from(["commit-assist", "--provider", "openai"]);

        assert_eq!(cli.provider, "openai");
    }

    #[test]
    fn test_cli_custom_model() {
        let cli = Cli::parse_from(["commit-assist", "--model", "gpt-4-turbo"]);

        assert_eq!(cli.model, Some("gpt-4-turbo".to_string()));
    }

    #[test]
    fn test_cli_short_flags() {
        let cli = Cli::parse_from(["commit-assist", "-p", "openai", "-m", "gpt-4"]);

        assert_eq!(cli.provider, "openai");
        assert_eq!(cli.model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_message_struct() {
        let msg = Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };

        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_message_clone() {
        let msg = Message {
            role: "assistant".to_string(),
            content: "Hi there".to_string(),
        };
        let cloned = msg.clone();

        assert_eq!(msg, cloned);
    }

    #[test]
    fn test_claude_request_serialization() {
        let request = ClaudeRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: "test".to_string(),
            }],
        };

        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("claude-sonnet-4-20250514"));
        assert!(json.contains("1024"));
        assert!(json.contains("user"));
    }

    #[test]
    fn test_openai_request_serialization() {
        let request = OpenAIRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: "You are helpful".to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("gpt-4o"));
        assert!(json.contains("system"));
        assert!(json.contains("user"));
    }

    #[test]
    fn test_claude_response_deserialization() {
        let json = r#"{"content": [{"text": "feat: add new feature"}]}"#;
        let response: ClaudeResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.content.len(), 1);
        assert_eq!(response.content[0].text, "feat: add new feature");
    }

    #[test]
    fn test_openai_response_deserialization() {
        let json = r#"{"choices": [{"message": {"role": "assistant", "content": "fix: resolve bug"}}]}"#;
        let response: OpenAIResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, "fix: resolve bug");
    }
}
