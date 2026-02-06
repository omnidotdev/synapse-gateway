use std::sync::OnceLock;

use regex::Regex;

/// Expand `{{ env.VAR }}` placeholders in a raw TOML string
///
/// This replaces `DynamicString<T>` by operating on the raw config text
/// before deserialization, so config structs use plain String/SecretString.
pub fn expand_env(input: &str) -> Result<String, String> {
    fn re() -> &'static Regex {
        static RE: OnceLock<Regex> = OnceLock::new();
        RE.get_or_init(|| Regex::new(r"\{\{\s*([[[:alnum:]]_.]+)\s*\}\}").expect("must be valid regex"))
    }

    let mut result = String::with_capacity(input.len());
    let mut last_end = 0;

    for captures in re().captures_iter(input) {
        let overall = captures.get(0).unwrap();
        let key = captures.get(1).unwrap().as_str();

        // Append text between matches
        result.push_str(&input[last_end..overall.start()]);

        let mut parts = key.split('.');
        match (parts.next(), parts.next(), parts.next()) {
            (Some("env"), Some(var_name), None) => match std::env::var(var_name) {
                Ok(value) => result.push_str(&value),
                Err(_) => {
                    return Err(format!("environment variable not found: `{var_name}`"));
                }
            },
            _ => {
                return Err(format!("only variables scoped with 'env.' are supported: `{key}`"));
            }
        }

        last_end = overall.end();
    }

    result.push_str(&input[last_end..]);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_placeholders() {
        let input = "key = \"value\"";
        assert_eq!(expand_env(input).unwrap(), input);
    }

    #[test]
    fn single_env_var() {
        temp_env::with_var("TEST_VAR", Some("hello"), || {
            let result = expand_env("key = \"{{ env.TEST_VAR }}\"").unwrap();
            assert_eq!(result, "key = \"hello\"");
        });
    }

    #[test]
    fn multiple_env_vars() {
        let vars = [("FOO", Some("foo")), ("BAR", Some("bar"))];
        temp_env::with_vars(vars, || {
            let result = expand_env("a = \"{{ env.FOO }}\"\nb = \"{{ env.BAR }}\"").unwrap();
            assert_eq!(result, "a = \"foo\"\nb = \"bar\"");
        });
    }

    #[test]
    fn missing_env_var() {
        temp_env::with_var_unset("MISSING_VAR", || {
            let err = expand_env("key = \"{{ env.MISSING_VAR }}\"").unwrap_err();
            assert!(err.contains("MISSING_VAR"));
        });
    }

    #[test]
    fn unsupported_scope() {
        let err = expand_env("key = \"{{ foo.BAR }}\"").unwrap_err();
        assert!(err.contains("only variables scoped with 'env.'"));
    }
}
