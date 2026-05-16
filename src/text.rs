//! Text → token ids for Supertonic 3.
//!
//! The model uses raw Unicode codepoints (no phonemes, no G2P). A single
//! JSON file `unicode_indexer.json` is a flat `Vec<i64>` indexed by
//! codepoint; out-of-range codepoints map to -1. Text is wrapped with a
//! `<lang>...</lang>` tag pair so the model knows which language to
//! synthesize.
//!
//! Pre-processing mirrors the official reference (NFKD normalization,
//! emoji stripping, dash/quote folding, punctuation tidy-up) — see
//! https://github.com/supertone-inc/supertonic/blob/main/rust/src/helper.rs

use anyhow::{bail, Context, Result};
use ndarray::Array3;
use regex::Regex;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

pub const AVAILABLE_LANGS: &[&str] = &[
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
];

pub fn is_valid_lang(lang: &str) -> bool {
    AVAILABLE_LANGS.contains(&lang)
}

/// Codepoint → token id lookup table, loaded once from
/// `onnx/unicode_indexer.json`.
pub struct UnicodeIndexer {
    table: Vec<i64>,
}

impl UnicodeIndexer {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("opening unicode_indexer at {}", path.display()))?;
        let table: Vec<i64> = serde_json::from_reader(BufReader::new(file))
            .context("parsing unicode_indexer.json")?;
        Ok(Self { table })
    }

    /// Number of entries in the table (codepoint range covered).
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Encode one preprocessed (language-tagged) string into a row of i64 ids.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        text.chars()
            .map(|c| {
                let cp = c as usize;
                if cp < self.table.len() {
                    self.table[cp]
                } else {
                    -1
                }
            })
            .collect()
    }

    /// Encode a batch of (already-preprocessed) strings, right-padding each
    /// row to the longest with `0`. Returns the id matrix as a flat `Vec<i64>`
    /// of shape `(batch, max_len)` along with the per-row real lengths and a
    /// 3-D mask `(batch, 1, max_len)` of ones for valid positions, zeros
    /// elsewhere — that's exactly what the ONNX models expect.
    pub fn encode_batch(&self, texts: &[String]) -> (Vec<Vec<i64>>, Vec<usize>, Array3<f32>) {
        let rows: Vec<Vec<i64>> = texts.iter().map(|t| self.encode(t)).collect();
        let lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        let max_len = lengths.iter().copied().max().unwrap_or(0);
        let mut padded = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut p = row.clone();
            p.resize(max_len, 0);
            padded.push(p);
        }
        let mut mask = Array3::<f32>::zeros((rows.len(), 1, max_len));
        for (b, &len) in lengths.iter().enumerate() {
            for t in 0..len {
                mask[[b, 0, t]] = 1.0;
            }
        }
        (padded, lengths, mask)
    }
}

/// Clean and language-tag a single utterance, returning the string that
/// gets fed to `UnicodeIndexer::encode`.
pub fn preprocess(text: &str, lang: &str) -> Result<String> {
    if !is_valid_lang(lang) {
        bail!("invalid lang {lang:?}; available: {AVAILABLE_LANGS:?}");
    }

    // NFKD: compatibility decomposition so "ﬁ" → "fi", "①" → "1", etc.
    let mut t: String = text.nfkd().collect();

    t = static_re::emoji().replace_all(&t, "").to_string();

    for (from, to) in &[
        ("\u{2013}", "-"), ("\u{2011}", "-"), ("\u{2014}", "-"),
        ("_", " "),
        ("\u{201C}", "\""), ("\u{201D}", "\""),
        ("\u{2018}", "'"), ("\u{2019}", "'"),
        ("\u{00B4}", "'"), ("`", "'"),
        ("[", " "), ("]", " "), ("|", " "), ("/", " "), ("#", " "),
        ("\u{2192}", " "), ("\u{2190}", " "),
    ] {
        t = t.replace(from, to);
    }
    for sym in &["\u{2665}", "\u{2606}", "\u{2661}", "\u{00A9}", "\\"] {
        t = t.replace(sym, "");
    }
    for (from, to) in &[("@", " at "), ("e.g.,", "for example, "), ("i.e.,", "that is, ")] {
        t = t.replace(from, to);
    }

    // Tidy whitespace around punctuation: " ," → ",", etc.
    for (re, repl) in static_re::space_before_punct() {
        t = re.replace_all(&t, *repl).to_string();
    }

    while t.contains("\"\"") { t = t.replace("\"\"", "\""); }
    while t.contains("''") { t = t.replace("''", "'"); }
    while t.contains("``") { t = t.replace("``", "`"); }

    t = static_re::whitespace().replace_all(&t, " ").to_string();
    let mut t = t.trim().to_string();

    if !t.is_empty() && !static_re::ends_with_punct().is_match(&t) {
        t.push('.');
    }

    Ok(format!("<{lang}>{t}</{lang}>"))
}

mod static_re {
    use regex::Regex;
    use std::sync::OnceLock;

    pub fn emoji() -> &'static Regex {
        static R: OnceLock<Regex> = OnceLock::new();
        R.get_or_init(|| Regex::new(r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{1F1E6}-\x{1F1FF}]+").unwrap())
    }

    pub fn whitespace() -> &'static Regex {
        static R: OnceLock<Regex> = OnceLock::new();
        R.get_or_init(|| Regex::new(r"\s+").unwrap())
    }

    pub fn ends_with_punct() -> &'static Regex {
        static R: OnceLock<Regex> = OnceLock::new();
        R.get_or_init(|| Regex::new(r#"[.!?;:,'"\u{201C}\u{201D}\u{2018}\u{2019})\]}…。」』】〉》›»]$"#).unwrap())
    }

    pub fn space_before_punct() -> &'static [(Regex, &'static str)] {
        static R: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
        R.get_or_init(|| {
            [
                (r" ,", ","), (r" \.", "."), (r" !", "!"), (r" \?", "?"),
                (r" ;", ";"), (r" :", ":"), (r" '", "'"),
            ]
            .iter()
            .map(|(p, r)| (Regex::new(p).unwrap(), *r))
            .collect()
        })
        .as_slice()
    }
}

/// Max characters per synth chunk. Korean/Japanese pack more glyphs into
/// the model's context window than Latin text, so we use a lower cap for
/// those — same convention as the reference impl.
pub const DEFAULT_MAX_CHUNK_LEN: usize = 300;
pub const CJK_MAX_CHUNK_LEN: usize = 120;

pub fn chunk_max_for(lang: &str) -> usize {
    match lang {
        "ko" | "ja" => CJK_MAX_CHUNK_LEN,
        _ => DEFAULT_MAX_CHUNK_LEN,
    }
}

const ABBREVIATIONS: &[&str] = &[
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "Ave.", "Rd.", "Blvd.", "Dept.",
    "Inc.", "Ltd.", "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
];

/// Split text into chunks at most `max_len` chars long, preferring
/// paragraph → sentence → comma → word boundaries (in that order).
pub fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![String::new()];
    }
    let para_re = Regex::new(r"\n\s*\n").unwrap();
    let mut chunks = Vec::new();
    for para in para_re.split(text) {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.chars().count() <= max_len {
            chunks.push(para.to_string());
            continue;
        }
        chunks.extend(split_long_paragraph(para, max_len));
    }
    if chunks.is_empty() {
        vec![String::new()]
    } else {
        chunks
    }
}

fn split_long_paragraph(para: &str, max_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for sentence in split_sentences(para) {
        let s = sentence.trim();
        if s.is_empty() {
            continue;
        }
        if s.chars().count() > max_len {
            if !cur.is_empty() {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            out.extend(split_long_sentence(s, max_len));
            continue;
        }
        let projected = cur.chars().count() + 1 + s.chars().count();
        if !cur.is_empty() && projected > max_len {
            out.push(cur.trim().to_string());
            cur.clear();
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(s);
    }
    if !cur.is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn split_long_sentence(sentence: &str, max_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for part in sentence.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part.chars().count() > max_len {
            if !cur.is_empty() {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            out.extend(split_by_words(part, max_len));
            continue;
        }
        let projected = cur.chars().count() + 2 + part.chars().count();
        if !cur.is_empty() && projected > max_len {
            out.push(cur.trim().to_string());
            cur.clear();
        }
        if !cur.is_empty() {
            cur.push_str(", ");
        }
        cur.push_str(part);
    }
    if !cur.is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn split_by_words(text: &str, max_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let projected = if cur.is_empty() {
            word.chars().count()
        } else {
            cur.chars().count() + 1 + word.chars().count()
        };
        if !cur.is_empty() && projected > max_len {
            out.push(cur.clone());
            cur.clear();
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn split_sentences(text: &str) -> Vec<String> {
    let re = Regex::new(r"([.!?])\s+").unwrap();
    let matches: Vec<_> = re.find_iter(text).collect();
    if matches.is_empty() {
        return vec![text.to_string()];
    }
    let mut sentences = Vec::new();
    let mut last = 0;
    for m in matches {
        let before = &text[last..m.start()];
        let candidate = format!("{}{}", before.trim_end(), &text[m.start()..m.start() + 1]);
        let is_abbrev = ABBREVIATIONS.iter().any(|abbr| candidate.ends_with(abbr));
        if !is_abbrev {
            sentences.push(text[last..m.end()].to_string());
            last = m.end();
        }
    }
    if last < text.len() {
        sentences.push(text[last..].to_string());
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_wraps_with_lang() {
        let out = preprocess("Hello world", "en").unwrap();
        assert_eq!(out, "<en>Hello world.</en>");
    }

    #[test]
    fn preprocess_keeps_existing_terminator() {
        let out = preprocess("Hello world!", "en").unwrap();
        assert_eq!(out, "<en>Hello world!</en>");
    }

    #[test]
    fn preprocess_collapses_smart_quotes_and_dashes() {
        let out = preprocess("\u{201C}hello\u{2014}world\u{201D}", "en").unwrap();
        assert!(out.contains("\"hello-world\""), "got: {out}");
    }

    #[test]
    fn invalid_lang_errors() {
        assert!(preprocess("hi", "zz").is_err());
    }

    #[test]
    fn chunk_short_passthrough() {
        let chunks = chunk_text("hello world", 300);
        assert_eq!(chunks, vec!["hello world".to_string()]);
    }

    #[test]
    fn chunk_long_paragraph_splits() {
        let text = "Short one. ".to_string() + &"long ".repeat(100);
        let chunks = chunk_text(&text, 80);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= 80, "chunk too long: {} chars", c.chars().count());
        }
    }
}
