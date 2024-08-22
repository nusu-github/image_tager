use anyhow::Result;
use dotenvy::dotenv;
use indicatif::ProgressStyle;
use serde::Deserialize;

pub use crate::qdrant_wrapper::*;
pub use crate::s3client::*;

mod qdrant_wrapper;
mod s3client;

#[derive(Deserialize, Debug)]
pub struct Config {
    pub aws_access_key_id: String,
    pub aws_secret_access_key: String,
    pub aws_region: String,
    pub s3_bucket_name: String,
    pub s3_endpoint: String,
    pub qdrant_url: String,
    pub collection_name: String,
}

impl Config {
    pub fn new() -> Result<Self> {
        dotenv().ok();
        Ok(config::Config::builder()
            .add_source(config::Environment::default())
            .build()?
            .try_deserialize()?)
    }
}

pub fn progress_style() -> Result<ProgressStyle> {
    Ok(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}",
    )?)
}
