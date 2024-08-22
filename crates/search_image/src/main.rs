use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use image::ImageFormat;
use image_tager::{progress_style, Config as AppConfig, Payload, QdrantWrapper, S3Client, SearchParams};
use indicatif::ProgressBar;
use models::WdTagger;
use tokio::fs;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CliConfig {
    input: PathBuf,
    output: Option<PathBuf>,
    #[arg(short, long, default_value_t = 100)]
    limit: usize,
    #[arg(short, long, default_value_t = 0.5)]
    score_threshold: f32,
    #[arg(long)]
    use_reqwest: bool,
    #[arg(short, long, default_value_t = 128)]
    batch_size: usize,
    #[arg(short, long, default_value_t = 0)]
    device_id: i32,
    #[arg(short, long, default_value_t = 16)]
    num_threads: usize,
    #[arg(short, long, default_value_t = false)]
    exact: bool,
    #[arg(short, long, default_value_t = 32)]
    hnsw_ef: u64,
}

struct ImageSearcher {
    qdrant_client: QdrantWrapper,
    s3_client: S3Client,
    model: WdTagger,
    app_config: AppConfig,
}

impl ImageSearcher {
    fn new(device_id: i32, num_threads: usize) -> Result<Self> {
        let app_config = AppConfig::new()?;
        let qdrant_client = QdrantWrapper::new()?;
        let s3_client = S3Client::new()?;
        let model = WdTagger::new(device_id, num_threads)?;

        Ok(Self {
            qdrant_client,
            s3_client,
            model,
            app_config,
        })
    }

    async fn process(&self, config: &CliConfig) -> Result<()> {
        let input =
            dunce::canonicalize(&config.input).context("Failed to canonicalize input path")?;
        let output = self.get_output_path(&input, &config.output)?;
        fs::create_dir_all(&output)
            .await
            .context("Failed to create output directory")?;

        let input_entries = self.get_input_entries(&input)?;

        for (entry, files) in input_entries {
            let entry_output = output.join(&entry);
            fs::create_dir_all(&entry_output)
                .await
                .context("Failed to create entry output directory")?;

            self.process_entry(&entry, &files, &entry_output, config)
                .await
                .with_context(|| format!("Failed to process entry: {}", entry))?;
        }

        Ok(())
    }

    fn get_output_path(&self, input: &Path, output: &Option<PathBuf>) -> Result<PathBuf> {
        match output {
            None => Ok(if input.is_dir() {
                input.parent().unwrap().join("output")
            } else {
                input.parent().unwrap().to_path_buf()
            }),
            Some(output) => Ok(output.clone()),
        }
    }

    fn validate_single_image_input(&self, input: &Path) -> Result<Vec<(String, Vec<PathBuf>)>> {
        anyhow::ensure!(
            ImageFormat::from_path(input).is_ok(),
            "Invalid image format"
        );
        Ok(vec![(
            input
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
            vec![input.to_path_buf()],
        )])
    }

    async fn process_entry(
        &self,
        tag: &str,
        files: &[PathBuf],
        output: &Path,
        config: &CliConfig,
    ) -> Result<()> {
        let progress_bar = ProgressBar::new(files.len() as u64);
        progress_bar.set_style(progress_style()?);
        progress_bar.set_message(tag.to_string());

        let vectors = self.process_images(files, &progress_bar).await?;
        let files_to_download = self
            .search_similar_images(vectors, config.limit as u64, config.score_threshold, config.exact, config.hnsw_ef)
            .await?;

        self.download_files(&files_to_download, output, &progress_bar, config.use_reqwest)
            .await
    }

    async fn process_images(&self, files: &[PathBuf], pb: &ProgressBar) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::new();
        for file in files {
            let image = image::open(file)?.into_rgb8();
            let vector = self.model.predict(&image).await?;
            vectors.push(vector);
            pb.inc(1);
        }
        Ok(vectors)
    }

    async fn search_similar_images(
        &self,
        vectors: Vec<Vec<f32>>,
        limit: u64,
        score_threshold: f32,
        exact: bool,
        hnsw_ef: u64,
    ) -> Result<Vec<Payload>> {
        let params = SearchParams {
            score_threshold,
            exact,
            hnsw_ef,
            limit,
        };
        self.qdrant_client
            .search_points(&self.app_config.collection_name, vectors, &params)
            .await
    }

    async fn download_files(
        &self,
        files: &[Payload],
        output: &Path,
        progress_bar: &ProgressBar,
        use_reqwest: bool,
    ) -> Result<()> {
        for payload in files {
            let path = output.join(&payload.path);
            let data = if use_reqwest {
                reqwest::get(&payload.url).await?.bytes().await?.to_vec()
            } else {
                self.s3_client.download_file(&payload.hash).await?
            };
            fs::create_dir_all(path.parent().unwrap()).await?;
            fs::write(&path, data).await?;
            progress_bar.inc(1);
        }
        Ok(())
    }

    fn get_input_entries(&self, input: &Path) -> Result<Vec<(String, Vec<PathBuf>)>> {
        if input.is_dir() {
            Ok(self.get_image_entries(input))
        } else {
            self.validate_single_image_input(input)
        }
    }

    fn get_image_entries(&self, root_path: &Path) -> Vec<(String, Vec<PathBuf>)> {
        WalkDir::new(root_path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_dir())
            .filter_map(|entry| {
                let path = entry.path();
                let folder_name = path.file_name()?.to_str()?.to_string();

                let image_files: Vec<_> = path
                    .read_dir()
                    .ok()?
                    .filter_map(Result::ok)
                    .map(|file_entry| file_entry.path())
                    .filter(|file_path| ImageFormat::from_path(file_path).is_ok())
                    .collect();

                (!image_files.is_empty()).then_some((folder_name, image_files))
            })
            .collect()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = CliConfig::parse();
    let searcher = ImageSearcher::new(config.device_id, config.num_threads)?;
    searcher.process(&config).await
}
