//! Heuristic query analysis for smart routing
//!
//! Classifies queries by task type and complexity using token counting
//! and keyword/pattern matching. No ML pipeline — pure heuristics.

use tiktoken_rs::o200k_base;

/// Broad task classification for a query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Code generation, debugging, implementation
    Code,
    /// Mathematical reasoning, calculations, proofs
    Math,
    /// Creative writing, storytelling
    Creative,
    /// Simple factual questions
    SimpleQa,
    /// General queries that don't fit other categories
    General,
}

/// Complexity level for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Complexity {
    Low,
    Medium,
    High,
}

/// Profile of a query for routing decisions
#[derive(Debug, Clone)]
pub struct QueryProfile {
    /// Estimated input token count
    pub estimated_input_tokens: usize,
    /// Classified task type
    pub task_type: TaskType,
    /// Overall complexity assessment
    pub complexity: Complexity,
    /// Whether the request uses tool calling
    pub requires_tool_use: bool,
}

/// Analyze a query for routing decisions
///
/// Examines the last user message for task classification and estimates
/// token count across all messages.
pub fn analyze_query(messages: &[serde_json::Value], has_tools: bool) -> QueryProfile {
    let full_text = extract_text(messages);
    let last_user = extract_last_user_message(messages);
    let estimated_input_tokens = estimate_tokens(&full_text);
    let task_type = classify_task(&last_user);
    let complexity = assess_complexity(estimated_input_tokens, task_type, has_tools);

    QueryProfile {
        estimated_input_tokens,
        task_type,
        complexity,
        requires_tool_use: has_tools,
    }
}

/// Extract all text content from messages for token counting
fn extract_text(messages: &[serde_json::Value]) -> String {
    let mut text = String::new();
    for msg in messages {
        if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
            text.push_str(content);
            text.push('\n');
        }
    }
    text
}

/// Extract the last user message content
fn extract_last_user_message(messages: &[serde_json::Value]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .unwrap_or("")
        .to_owned()
}

/// Estimate token count using tiktoken
fn estimate_tokens(text: &str) -> usize {
    o200k_base().map_or_else(|_| text.len() / 4, |bpe| bpe.encode_with_special_tokens(text).len())
}

/// Classify the task type from the last user message
fn classify_task(text: &str) -> TaskType {
    let lower = text.to_lowercase();

    if is_code_task(&lower) {
        TaskType::Code
    } else if is_math_task(&lower) {
        TaskType::Math
    } else if is_creative_task(&lower, text) {
        TaskType::Creative
    } else if is_simple_qa(&lower, text) {
        TaskType::SimpleQa
    } else {
        TaskType::General
    }
}

fn is_code_task(lower: &str) -> bool {
    // Code fences
    if lower.contains("```") {
        return true;
    }

    // Programming keywords
    let code_keywords = [
        "implement",
        "debug",
        "function",
        "refactor",
        "compile",
        "runtime error",
        "syntax error",
        "code review",
        "write a program",
        "write code",
        "fix this code",
        "bug in",
        "stack trace",
        "unit test",
    ];
    if code_keywords.iter().any(|k| lower.contains(k)) {
        return true;
    }

    // File extensions
    let extensions = [".rs", ".ts", ".py", ".js", ".go", ".java", ".cpp", ".tsx", ".jsx"];
    extensions.iter().any(|ext| lower.contains(ext))
}

fn is_math_task(lower: &str) -> bool {
    let math_keywords = [
        "calculate",
        "solve",
        "prove",
        "equation",
        "integral",
        "derivative",
        "theorem",
        "∫",
        "∑",
        "∏",
        "matrix",
        "eigenvalue",
        "probability",
    ];
    math_keywords.iter().any(|k| lower.contains(k))
}

fn is_creative_task(lower: &str, _original: &str) -> bool {
    let creative_keywords = [
        "write a story",
        "write a poem",
        "creative writing",
        "compose",
        "fictional",
        "narrative",
        "write me a",
        "tell me a story",
    ];
    creative_keywords.iter().any(|k| lower.contains(k))
}

fn is_simple_qa(lower: &str, original: &str) -> bool {
    // Short single-turn questions
    if original.len() > 200 {
        return false;
    }

    let qa_prefixes = [
        "what is",
        "what are",
        "who is",
        "who was",
        "when did",
        "where is",
        "how many",
        "how much",
        "define ",
        "what does",
    ];
    qa_prefixes.iter().any(|p| lower.starts_with(p))
}

/// Assess overall complexity from multiple signals
fn assess_complexity(tokens: usize, task_type: TaskType, has_tools: bool) -> Complexity {
    // Tool use always bumps complexity
    if has_tools {
        return Complexity::High;
    }

    match task_type {
        TaskType::SimpleQa => Complexity::Low,
        TaskType::Code | TaskType::Math => {
            if tokens > 2000 {
                Complexity::High
            } else {
                Complexity::Medium
            }
        }
        TaskType::Creative => {
            if tokens > 1000 {
                Complexity::High
            } else {
                Complexity::Medium
            }
        }
        TaskType::General => {
            if tokens > 3000 {
                Complexity::High
            } else if tokens > 500 {
                Complexity::Medium
            } else {
                Complexity::Low
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(content: &str) -> serde_json::Value {
        serde_json::json!({"role": "user", "content": content})
    }

    #[test]
    fn code_detection() {
        assert_eq!(classify_task("implement a function to sort"), TaskType::Code);
        assert_eq!(classify_task("fix this code ```rust\nfn main()```"), TaskType::Code);
        assert_eq!(classify_task("debug the error in main.rs"), TaskType::Code);
    }

    #[test]
    fn math_detection() {
        assert_eq!(classify_task("solve this equation: 2x + 3 = 7"), TaskType::Math);
        assert_eq!(classify_task("calculate the integral"), TaskType::Math);
    }

    #[test]
    fn creative_detection() {
        assert_eq!(classify_task("write a story about a dragon"), TaskType::Creative);
        assert_eq!(classify_task("write a poem about the sea"), TaskType::Creative);
    }

    #[test]
    fn simple_qa_detection() {
        assert_eq!(classify_task("what is rust?"), TaskType::SimpleQa);
        assert_eq!(classify_task("who is Ada Lovelace?"), TaskType::SimpleQa);
    }

    #[test]
    fn general_detection() {
        assert_eq!(classify_task("explain how DNS works in detail"), TaskType::General);
    }

    #[test]
    fn complexity_tool_use_is_high() {
        let profile = analyze_query(&[user_msg("hello")], true);
        assert_eq!(profile.complexity, Complexity::High);
    }

    #[test]
    fn complexity_simple_qa_is_low() {
        let profile = analyze_query(&[user_msg("what is rust?")], false);
        assert_eq!(profile.complexity, Complexity::Low);
    }
}
