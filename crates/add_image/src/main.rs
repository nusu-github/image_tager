use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use clap::Parser;
use image::{ImageFormat, RgbImage};
use indicatif::ProgressIterator;
use qdrant_client::qdrant::PointStruct;
use uuid::Uuid;
use walkdir::WalkDir;

use image_tager::{progress_style, Config as AppConfig, QdrantWrapper, S3Client};
use models::WdTagger;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CliConfig {
    input_dir: PathBuf,
    #[arg(short, long, default_value_t = 128)]
    batch_size: usize,
    #[arg(short, long, default_value_t = 0)]
    device_id: i32,
    #[arg(short, long, default_value_t = 16)]
    num_threads: usize,
}

struct ImageProcessor {
    s3_client: Arc<S3Client>,
    qdrant_client: Arc<QdrantWrapper>,
    model: Arc<WdTagger>,
    num_threads: usize,
    app_config: AppConfig,
    base_url: String,
}

impl ImageProcessor {
    fn new(device_id: i32, num_threads: usize) -> Result<Self> {
        let app_config = AppConfig::new()?;
        let base_url = format!("{}/{}", &app_config.s3_endpoint, &app_config.s3_bucket_name);

        Ok(Self {
            s3_client: Arc::from(S3Client::new()?),
            qdrant_client: Arc::from(QdrantWrapper::new()?),
            model: Arc::from(WdTagger::new(device_id, num_threads)?),
            num_threads,
            app_config,
            base_url,
        })
    }

    async fn process(&self, config: &CliConfig) -> Result<()> {
        let input_dir = self.canonicalize_input_dir(&config.input_dir)?;
        self.ensure_image_collection_exists().await?;

        let entries = self.get_image_entries(&input_dir);
        self.process_entries(entries, config.batch_size).await
    }

    fn canonicalize_input_dir(&self, input_dir: &Path) -> Result<PathBuf> {
        dunce::canonicalize(input_dir).context("Failed to canonicalize input directory")
    }

    async fn ensure_image_collection_exists(&self) -> Result<()> {
        if self
            .qdrant_client
            .get_collection_info(&self.app_config.collection_name)
            .await
            .is_err()
        {
            self.qdrant_client
                .create_collection(
                    &self.app_config.collection_name,
                    self.model.output_size as u64,
                )
                .await?;
        }
        Ok(())
    }

    fn get_image_entries(&self, input_dir: &Path) -> Vec<PathBuf> {
        WalkDir::new(input_dir)
            .into_iter()
            .filter_map(Result::ok)
            .map(|e| e.into_path())
            .filter(|e| ImageFormat::from_path(e).is_ok())
            .collect()
    }

    async fn process_entries(&self, entries: Vec<PathBuf>, batch_size: usize) -> Result<()> {
        for batch in entries
            .chunks(batch_size)
            .progress_with_style(progress_style()?)
        {
            let processed_batch = self.process_batch(batch).await?;
            self.upload_and_index_batch(processed_batch).await?;
        }
        Ok(())
    }

    async fn process_batch(&self, batch: &[PathBuf]) -> Result<Vec<ProcessedImage>> {
        let mut processed_batch = Vec::new();
        for paths in batch.chunks(self.num_threads) {
            let datas = self.load_and_hash_images(paths).await?;
            let vectors = self
                .model
                .predicts(&datas.iter().map(|d| d.image.clone()).collect::<Vec<_>>())
                .await?;

            processed_batch.extend(datas.into_iter().zip(vectors).map(|(data, vector)| {
                ProcessedImage {
                    path: data.path,
                    vector,
                    hash: data.hash,
                }
            }));
        }
        Ok(processed_batch)
    }

    async fn load_and_hash_images(&self, paths: &[PathBuf]) -> Result<Vec<ImageData>> {
        futures_util::future::try_join_all(paths.iter().map(|path| self.load_and_hash_image(path)))
            .await
    }

    async fn load_and_hash_image(&self, path: &Path) -> Result<ImageData> {
        tokio::task::spawn_blocking({
            let path = path.to_owned();
            move || -> Result<ImageData> {
                let mut hasher = blake3::Hasher::new();
                let hash = hasher.update_mmap(&path)?.finalize().to_string();
                let image = image::open(&path)?.into_rgb8();
                Ok(ImageData { path, image, hash })
            }
        })
            .await?
    }

    async fn upload_and_index_batch(&self, batch: Vec<ProcessedImage>) -> Result<()> {
        let qdrant_points: Vec<_> = futures_util::future::join_all(
            batch.into_iter().map(|img| self.prepare_qdrant_point(img)),
        )
            .await;

        self.qdrant_client
            .add_points(&self.app_config.collection_name, qdrant_points)
            .await
    }

    async fn prepare_qdrant_point(&self, img: ProcessedImage) -> PointStruct {
        let filename = format!(
            "{}.{}",
            img.hash,
            img.path.extension().unwrap().to_str().unwrap()
        );
        let full_url = format!("{}/{}", self.base_url, filename);

        // Upload file to S3 if it doesn't exist
        if let Err(e) = self.upload_to_s3_if_not_exists(&img.path, &filename).await {
            eprintln!("Failed to upload file to S3: {}", e);
        }

        self.create_qdrant_point(
            &img.hash,
            img.vector,
            img.path.file_name().unwrap().to_str().unwrap(),
            &full_url,
        )
    }

    async fn upload_to_s3_if_not_exists(&self, path: &Path, filename: &str) -> Result<()> {
        if !self.s3_client.search_file(filename).await? {
            let data = tokio::fs::read(path).await?;
            self.s3_client.upload_file(filename, &data).await?;
        }
        Ok(())
    }

    fn create_qdrant_point(
        &self,
        hash: &str,
        vector: Vec<f32>,
        path_str: &str,
        full_url: &str,
    ) -> PointStruct {
        PointStruct::new(
            Uuid::new_v5(&Uuid::NAMESPACE_DNS, hash.as_ref()).to_string(),
            vector,
            [
                ("path", path_str.into()),
                ("hash", hash.into()),
                ("url", full_url.into()),
            ],
        )
    }
}

struct ProcessedImage {
    path: PathBuf,
    vector: Vec<f32>,
    hash: String,
}

struct ImageData {
    path: PathBuf,
    image: RgbImage,
    hash: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = CliConfig::parse();
    let processor = ImageProcessor::new(config.device_id, config.num_threads)?;
    processor.process(&config).await
}
