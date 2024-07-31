use std::{env, sync::LazyLock};

use dotenvy::dotenv;

static ACCESS_KEY_ID: LazyLock<String> =
    LazyLock::new(|| env::var("ACCESS_KEY_ID").unwrap_or("".to_string()));
static BUCKET_NAME: LazyLock<String> =
    LazyLock::new(|| env::var("BUCKET_NAME").unwrap_or("".to_string()));
static ENDPOINT: LazyLock<String> =
    LazyLock::new(|| env::var("ENDPOINT").unwrap_or("".to_string()));
static REGION: LazyLock<String> = LazyLock::new(|| env::var("REGION").unwrap_or("".to_string()));
static SECRET_ACCESS_KEY: LazyLock<String> =
    LazyLock::new(|| env::var("SECRET_ACCESS_KEY").unwrap_or("".to_string()));
static QDRANT_URL: LazyLock<String> =
    LazyLock::new(|| env::var("QDRANT_URL").unwrap_or("".to_string()));
static COLLECTION_NAME: LazyLock<String> =
    LazyLock::new(|| env::var("COLLECTION_NAME").unwrap_or("".to_string()));
static DEVICE_ID: LazyLock<String> =
    LazyLock::new(|| env::var("DEVICE_ID").unwrap_or("0".to_string()));

pub struct Config {
    pub access_key_id: &'static str,
    pub bucket_name: &'static str,
    pub collection_name: &'static str,
    pub device_id: &'static str,
    pub endpoint: &'static str,
    pub qdrant_url: &'static str,
    pub region: &'static str,
    pub secret_access_key: &'static str,
}

impl Config {
    pub fn new() -> Self {
        dotenv().ok();
        Self {
            access_key_id: ACCESS_KEY_ID.as_str(),
            bucket_name: BUCKET_NAME.as_str(),
            collection_name: COLLECTION_NAME.as_str(),
            device_id: DEVICE_ID.as_str(),
            endpoint: ENDPOINT.as_str(),
            qdrant_url: QDRANT_URL.as_str(),
            region: REGION.as_str(),
            secret_access_key: SECRET_ACCESS_KEY.as_str(),
        }
    }
}
