#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]

pub mod inference;
pub mod model;
pub mod quantized;
pub mod tokenizer;
pub mod weights;
