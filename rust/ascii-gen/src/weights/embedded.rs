// Generated at build time by build.rs.
//
// This file always exists in the crate so downstream code can refer to the same symbols regardless
// of whether embedding is enabled. When the `embedded-weights` feature is disabled (or assets are
// missing), build.rs generates a stub with `EMBEDDED_MODEL_PRESENT = false`.
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/embedded_assets.rs"));
