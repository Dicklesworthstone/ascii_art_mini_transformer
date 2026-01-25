//! Character-level tokenizer for ASCII art generation.
//!
//! Matches the Python `AsciiTokenizer` vocabulary:
//! - Special tokens occupy IDs 0..=11
//! - Printable ASCII (0x20..=0x7E) occupies IDs 12..=106
//! - Newlines are represented explicitly via `<NEWLINE>` (ID 7)

/// Special token IDs.
pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const SEP_ID: u32 = 4;
pub const WIDTH_ID: u32 = 5;
pub const HEIGHT_ID: u32 = 6;
pub const NEWLINE_ID: u32 = 7;
pub const STYLE_ART_ID: u32 = 8;
pub const STYLE_BANNER_ID: u32 = 9;
pub const STYLE_SIMPLE_ID: u32 = 10;
pub const STYLE_DETAILED_ID: u32 = 11;

pub const NUM_SPECIAL_TOKENS: u32 = 12;

pub const PRINTABLE_ASCII_START: u32 = NUM_SPECIAL_TOKENS; // 12
pub const PRINTABLE_ASCII_LEN: u32 = 95; // 0x20..=0x7E inclusive
pub const PRINTABLE_ASCII_END: u32 = PRINTABLE_ASCII_START + PRINTABLE_ASCII_LEN - 1; // 106

/// Total vocabulary size: 12 special tokens + 95 printable ASCII = 107.
pub const VOCAB_SIZE: u32 = NUM_SPECIAL_TOKENS + PRINTABLE_ASCII_LEN;

/// Minimal tokenizer for ASCII art.
///
/// This is intentionally stateless: mapping is fixed.
#[derive(Debug, Default, Clone, Copy)]
pub struct AsciiTokenizer;

impl AsciiTokenizer {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Total vocabulary size (107 tokens).
    #[must_use]
    pub fn vocab_size(self) -> u32 {
        VOCAB_SIZE
    }

    // --- Special token accessors ---

    #[must_use]
    pub fn pad_id(self) -> u32 {
        PAD_ID
    }

    #[must_use]
    pub fn bos_id(self) -> u32 {
        BOS_ID
    }

    #[must_use]
    pub fn eos_id(self) -> u32 {
        EOS_ID
    }

    #[must_use]
    pub fn unk_id(self) -> u32 {
        UNK_ID
    }

    #[must_use]
    pub fn sep_id(self) -> u32 {
        SEP_ID
    }

    #[must_use]
    pub fn width_id(self) -> u32 {
        WIDTH_ID
    }

    #[must_use]
    pub fn height_id(self) -> u32 {
        HEIGHT_ID
    }

    #[must_use]
    pub fn newline_id(self) -> u32 {
        NEWLINE_ID
    }

    #[must_use]
    pub fn style_art_id(self) -> u32 {
        STYLE_ART_ID
    }

    #[must_use]
    pub fn style_banner_id(self) -> u32 {
        STYLE_BANNER_ID
    }

    #[must_use]
    pub fn style_simple_id(self) -> u32 {
        STYLE_SIMPLE_ID
    }

    #[must_use]
    pub fn style_detailed_id(self) -> u32 {
        STYLE_DETAILED_ID
    }

    /// Get style token ID by name.
    ///
    /// Returns `None` for unknown style names.
    #[must_use]
    pub fn get_style_token_id(self, style: &str) -> Option<u32> {
        match style.to_lowercase().as_str() {
            "art" => Some(STYLE_ART_ID),
            "banner" => Some(STYLE_BANNER_ID),
            "simple" => Some(STYLE_SIMPLE_ID),
            "detailed" => Some(STYLE_DETAILED_ID),
            _ => None,
        }
    }

    /// Returns true if this token represents output content (printable ASCII or newline).
    #[must_use]
    pub fn is_output_token(self, token_id: u32) -> bool {
        token_id == NEWLINE_ID || (PRINTABLE_ASCII_START..=PRINTABLE_ASCII_END).contains(&token_id)
    }

    /// Encode a single character into a token ID.
    #[must_use]
    pub fn encode_char(self, ch: char) -> u32 {
        if ch == '\n' {
            return NEWLINE_ID;
        }
        let cp = ch as u32;
        if (0x20..=0x7E).contains(&cp) {
            return PRINTABLE_ASCII_START + (cp - 0x20);
        }
        UNK_ID
    }

    /// Encode a string into token IDs.
    #[must_use]
    pub fn encode(self, text: &str) -> Vec<u32> {
        text.chars().map(|ch| self.encode_char(ch)).collect()
    }

    /// Decode a token ID into an output character.
    ///
    /// Returns `None` for non-output (control/style) tokens.
    #[must_use]
    pub fn decode_token(self, token_id: u32) -> Option<char> {
        if token_id == NEWLINE_ID {
            return Some('\n');
        }
        if (PRINTABLE_ASCII_START..=PRINTABLE_ASCII_END).contains(&token_id) {
            let offset = token_id - PRINTABLE_ASCII_START;
            let cp = 0x20 + offset;
            return char::from_u32(cp);
        }
        None
    }

    /// Decode a token sequence into a string, skipping non-output tokens.
    #[must_use]
    pub fn decode(self, token_ids: &[u32]) -> String {
        let mut out = String::new();
        for &tid in token_ids {
            if let Some(ch) = self.decode_token(tid) {
                out.push(ch);
            }
        }
        out
    }

    /// Encode an inference prompt with width, height, style, and description.
    ///
    /// The format matches Python's `encode_inference_prompt`:
    /// `<BOS> <WIDTH> {width_digits} <HEIGHT> {height_digits} <STYLE_X> {description} <SEP>`
    ///
    /// # Arguments
    /// * `width` - Target width in characters
    /// * `height` - Target height in lines
    /// * `style` - Style name: "art", "banner", "simple", or "detailed"
    /// * `description` - Text description of the art to generate
    ///
    /// # Returns
    /// Token IDs for the inference prompt (no EOS at end)
    #[must_use]
    pub fn encode_prompt(
        self,
        width: usize,
        height: usize,
        style: &str,
        description: &str,
    ) -> Vec<u32> {
        let mut tokens = Vec::new();

        // BOS
        tokens.push(BOS_ID);

        // WIDTH + digits
        tokens.push(WIDTH_ID);
        for ch in width.to_string().chars() {
            tokens.push(self.encode_char(ch));
        }

        // HEIGHT + digits
        tokens.push(HEIGHT_ID);
        for ch in height.to_string().chars() {
            tokens.push(self.encode_char(ch));
        }

        // Style token
        let style_id = self.get_style_token_id(style).unwrap_or(STYLE_ART_ID);
        tokens.push(style_id);

        // Description characters
        for ch in description.chars() {
            tokens.push(self.encode_char(ch));
        }

        // SEP marks end of prompt
        tokens.push(SEP_ID);

        tokens
    }

    /// Encode training data with the complete format.
    ///
    /// The format is: `<BOS> <WIDTH> {w} <HEIGHT> {h} <STYLE> {desc} <SEP> {art} <EOS>`
    #[must_use]
    pub fn encode_training(
        self,
        width: usize,
        height: usize,
        style: &str,
        description: &str,
        art: &str,
    ) -> Vec<u32> {
        let mut tokens = self.encode_prompt(width, height, style, description);

        // Encode the art content
        for ch in art.chars() {
            tokens.push(self.encode_char(ch));
        }

        // EOS marks end of sequence
        tokens.push(EOS_ID);

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Vocabulary Tests ====================

    #[test]
    fn test_vocab_size() {
        let tokenizer = AsciiTokenizer::new();
        // 95 printable + 12 special tokens
        assert_eq!(tokenizer.vocab_size(), 107);
    }

    #[test]
    fn test_special_token_ids() {
        let tokenizer = AsciiTokenizer::new();
        assert_eq!(tokenizer.pad_id(), 0);
        assert_eq!(tokenizer.bos_id(), 1);
        assert_eq!(tokenizer.eos_id(), 2);
        assert_eq!(tokenizer.unk_id(), 3);
        assert_eq!(tokenizer.sep_id(), 4);
        assert_eq!(tokenizer.width_id(), 5);
        assert_eq!(tokenizer.height_id(), 6);
        assert_eq!(tokenizer.newline_id(), 7);
        // Style tokens are distinct
        assert_eq!(tokenizer.style_art_id(), 8);
        assert_eq!(tokenizer.style_banner_id(), 9);
        assert_eq!(tokenizer.style_simple_id(), 10);
        assert_eq!(tokenizer.style_detailed_id(), 11);
    }

    #[test]
    fn test_printable_ascii_range() {
        let tokenizer = AsciiTokenizer::new();
        // Space (32) to tilde (126) - 95 chars
        for c in 32u8..=126 {
            let ch = char::from(c);
            let id = tokenizer.encode_char(ch);
            // IDs should be >= 12 (after special tokens) and <= 106
            assert!(
                (12..=106).contains(&id),
                "Failed for char: {ch} (ASCII {c}), got ID {id}",
            );
        }
    }

    #[test]
    fn test_style_token_lookup() {
        let tokenizer = AsciiTokenizer::new();
        // Style should be a single token
        assert_eq!(tokenizer.get_style_token_id("art"), Some(8));
        assert_eq!(tokenizer.get_style_token_id("banner"), Some(9));
        assert_eq!(tokenizer.get_style_token_id("simple"), Some(10));
        assert_eq!(tokenizer.get_style_token_id("detailed"), Some(11));
        // Case insensitive
        assert_eq!(tokenizer.get_style_token_id("ART"), Some(8));
        assert_eq!(tokenizer.get_style_token_id("BANNER"), Some(9));
        // Unknown style
        assert_eq!(tokenizer.get_style_token_id("unknown"), None);
    }

    // ==================== Encode/Decode Round-Trip Tests ====================

    #[test]
    fn test_simple_roundtrip() {
        let tokenizer = AsciiTokenizer::new();
        let text = "Hello World";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_ascii_art_roundtrip() {
        let tokenizer = AsciiTokenizer::new();
        let art = " /\\_/\\ \n( o.o )\n > ^ < ";
        let ids = tokenizer.encode(art);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, art);
    }

    #[test]
    fn test_all_printable_chars_roundtrip() {
        let tokenizer = AsciiTokenizer::new();
        // All printable ASCII
        let text: String = (32u8..=126).map(char::from).collect();
        let ids = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_multiline_roundtrip() {
        let tokenizer = AsciiTokenizer::new();
        let art = "Line1\nLine2\nLine3";
        let ids = tokenizer.encode(art);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, art);
    }

    // ==================== Prompt Encoding Tests ====================

    #[test]
    fn test_encode_prompt_structure() {
        let tokenizer = AsciiTokenizer::new();
        let prompt = tokenizer.encode_prompt(40, 20, "art", "a cute cat");

        // Should start with BOS
        assert_eq!(prompt[0], tokenizer.bos_id());

        // Should contain WIDTH token
        assert!(prompt.contains(&tokenizer.width_id()));

        // Should contain HEIGHT token
        assert!(prompt.contains(&tokenizer.height_id()));

        // Should contain STYLE_ART token
        assert!(prompt.contains(&tokenizer.style_art_id()));

        // Should contain SEP token
        assert!(prompt.contains(&tokenizer.sep_id()));

        // Should NOT end with EOS (inference prompt)
        assert_ne!(prompt.last(), Some(&tokenizer.eos_id()));

        // SEP should be last
        assert_eq!(prompt.last(), Some(&tokenizer.sep_id()));
    }

    #[test]
    fn test_style_comes_before_sep() {
        let tokenizer = AsciiTokenizer::new();
        let prompt = tokenizer.encode_prompt(40, 20, "banner", "HELLO");

        let style_pos = prompt
            .iter()
            .position(|&x| x == tokenizer.style_banner_id())
            .expect("Style token not found");
        let sep_pos = prompt
            .iter()
            .position(|&x| x == tokenizer.sep_id())
            .expect("SEP token not found");

        // Style token should come before SEP
        assert!(
            style_pos < sep_pos,
            "Style at {style_pos} should be before SEP at {sep_pos}",
        );

        // Description chars should be between style and SEP
        let h_id = tokenizer.encode_char('H');
        let h_pos = prompt.iter().position(|&x| x == h_id).expect("H not found");
        assert!(h_pos > style_pos, "H should come after style token");
        assert!(h_pos < sep_pos, "H should come before SEP");
    }

    #[test]
    fn test_encode_prompt_with_different_styles() {
        let tokenizer = AsciiTokenizer::new();

        let styles = [
            ("art", tokenizer.style_art_id()),
            ("banner", tokenizer.style_banner_id()),
            ("simple", tokenizer.style_simple_id()),
            ("detailed", tokenizer.style_detailed_id()),
        ];

        for (style_name, expected_id) in styles {
            let prompt = tokenizer.encode_prompt(80, 24, style_name, "test");
            assert!(
                prompt.contains(&expected_id),
                "Prompt for style '{style_name}' should contain token {expected_id}",
            );
        }
    }

    #[test]
    fn test_encode_training_has_eos() {
        let tokenizer = AsciiTokenizer::new();
        let tokens = tokenizer.encode_training(40, 20, "art", "cat", " /\\_/\\ ");

        // Training sequence should end with EOS
        assert_eq!(tokens.last(), Some(&tokenizer.eos_id()));

        // Should also have BOS at start
        assert_eq!(tokens[0], tokenizer.bos_id());

        // Should contain SEP between prompt and art
        assert!(tokens.contains(&tokenizer.sep_id()));
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_empty_string() {
        let tokenizer = AsciiTokenizer::new();
        let ids = tokenizer.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_non_ascii_to_unk() {
        let tokenizer = AsciiTokenizer::new();
        let text = "Hello\u{1F600}World"; // Emoji
        let ids = tokenizer.encode(text);
        // Should contain UNK for emoji
        assert!(ids.contains(&tokenizer.unk_id()));
        let decoded = tokenizer.decode(&ids);
        // UNK is skipped in decode, so we get HelloWorld
        assert_eq!(decoded, "HelloWorld");
    }

    #[test]
    fn test_control_chars_to_unk() {
        let tokenizer = AsciiTokenizer::new();
        // Tab and other control chars should be UNK
        let text = "A\tB\rC";
        let ids = tokenizer.encode(text);
        // Tab and CR are not printable, should be UNK
        assert_eq!(ids.iter().filter(|&&x| x == UNK_ID).count(), 2);
    }

    #[test]
    fn test_newline_encoding() {
        let tokenizer = AsciiTokenizer::new();
        let text = "Line1\nLine2";
        let ids = tokenizer.encode(text);
        // Should contain exactly one NEWLINE token
        let newline_count = ids.iter().filter(|&&x| x == tokenizer.newline_id()).count();
        assert_eq!(newline_count, 1);
    }

    #[test]
    fn test_decode_skips_control_tokens() {
        let tokenizer = AsciiTokenizer::new();
        let ids = vec![
            BOS_ID,
            PRINTABLE_ASCII_START,
            NEWLINE_ID,
            WIDTH_ID,
            HEIGHT_ID,
            EOS_ID,
        ];
        // Only space and newline should be decoded
        assert_eq!(tokenizer.decode(&ids), " \n");
    }

    #[test]
    fn test_is_output_token() {
        let tokenizer = AsciiTokenizer::new();

        // Newline is output
        assert!(tokenizer.is_output_token(NEWLINE_ID));

        // Printable ASCII is output
        assert!(tokenizer.is_output_token(PRINTABLE_ASCII_START)); // space
        assert!(tokenizer.is_output_token(PRINTABLE_ASCII_END)); // tilde

        // Special tokens are NOT output
        assert!(!tokenizer.is_output_token(PAD_ID));
        assert!(!tokenizer.is_output_token(BOS_ID));
        assert!(!tokenizer.is_output_token(EOS_ID));
        assert!(!tokenizer.is_output_token(UNK_ID));
        assert!(!tokenizer.is_output_token(SEP_ID));
        assert!(!tokenizer.is_output_token(WIDTH_ID));
        assert!(!tokenizer.is_output_token(HEIGHT_ID));
        assert!(!tokenizer.is_output_token(STYLE_ART_ID));
    }

    // ==================== Character Mapping Tests ====================

    #[test]
    fn test_specific_char_mappings() {
        let tokenizer = AsciiTokenizer::new();
        // Verify specific important characters
        assert_eq!(tokenizer.encode_char(' '), 12); // space -> 12
        assert_eq!(tokenizer.encode_char('!'), 13);
        assert_eq!(tokenizer.encode_char('0'), 28); // '0' is 0x30 = 48, 48 - 32 + 12 = 28
        assert_eq!(tokenizer.encode_char('A'), 45); // 'A' is 0x41 = 65, 65 - 32 + 12 = 45
        assert_eq!(tokenizer.encode_char('a'), 77); // 'a' is 0x61 = 97, 97 - 32 + 12 = 77
        assert_eq!(tokenizer.encode_char('~'), 106); // tilde -> 106
    }

    #[test]
    fn test_decode_out_of_range() {
        let tokenizer = AsciiTokenizer::new();
        // Token IDs outside valid range should return None
        assert_eq!(tokenizer.decode_token(107), None); // just past tilde
        assert_eq!(tokenizer.decode_token(1000), None);
        assert_eq!(tokenizer.decode_token(0), None); // PAD
        assert_eq!(tokenizer.decode_token(1), None); // BOS
    }

    #[test]
    fn test_width_height_digits_in_prompt() {
        let tokenizer = AsciiTokenizer::new();
        let prompt = tokenizer.encode_prompt(80, 24, "art", "x");

        // After WIDTH_ID should come '8' and '0'
        let width_pos = prompt.iter().position(|&x| x == WIDTH_ID).unwrap();
        assert_eq!(prompt[width_pos + 1], tokenizer.encode_char('8'));
        assert_eq!(prompt[width_pos + 2], tokenizer.encode_char('0'));

        // After HEIGHT_ID should come '2' and '4'
        let height_pos = prompt.iter().position(|&x| x == HEIGHT_ID).unwrap();
        assert_eq!(prompt[height_pos + 1], tokenizer.encode_char('2'));
        assert_eq!(prompt[height_pos + 2], tokenizer.encode_char('4'));
    }
}
