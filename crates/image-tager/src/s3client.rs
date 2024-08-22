use anyhow::Result;
use aws_config::Region;
use aws_sdk_s3::{config::Credentials, primitives::ByteStream, Client};

use crate::Config;

pub struct S3Client {
    client: Client,
    bucket: String,
}

impl S3Client {
    pub fn new() -> Result<Self> {
        let app_config = Config::new()?;
        let client = create_s3_client(&app_config);
        Ok(Self {
            client,
            bucket: app_config.s3_bucket_name.clone(),
        })
    }

    pub async fn upload_file(&self, key: &str, data: &[u8]) -> Result<()> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(data.to_vec()))
            .send()
            .await?;
        Ok(())
    }

    pub async fn download_file(&self, key: &str) -> Result<Vec<u8>> {
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await?;
        Ok(output.body.collect().await?.to_vec())
    }

    pub async fn search_file(&self, key: &str) -> Result<bool> {
        let output = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await;
        Ok(output.is_ok())
    }

    pub async fn list_files(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let mut request = self.client.list_objects_v2().bucket(&self.bucket);
        if let Some(prefix) = prefix {
            request = request.prefix(prefix);
        }
        let output = request.send().await?;
        Ok(output
            .contents()
            .iter()
            .filter_map(|obj| obj.key().map(String::from))
            .collect())
    }
}

fn create_s3_client(config: &Config) -> Client {
    let credentials = Credentials::new(
        &config.aws_access_key_id,
        &config.aws_secret_access_key,
        None,
        None,
        "example",
    );
    let config = aws_sdk_s3::Config::builder()
        .behavior_version_latest()
        .credentials_provider(credentials)
        .region(Region::new(config.aws_region.clone()))
        .force_path_style(true)
        .endpoint_url(&config.s3_endpoint)
        .build();
    Client::from_conf(config)
}
