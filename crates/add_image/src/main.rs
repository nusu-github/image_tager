use std::{
    env,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use aws_config::Region;
use aws_sdk_s3::{config::Credentials, primitives::ByteStream, Client};
use clap::Parser;
use dotenvy::dotenv;
use futures_util::{stream, StreamExt};
use image::ImageFormat;
use indicatif::ProgressBar;
use memmap2::Mmap;
use qdrant_client::{
    qdrant::{
        quantization_config::Quantization, CreateCollectionBuilder, Distance, PointStruct,
        ScalarQuantizationBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
    Qdrant,
};
use walkdir::WalkDir;

use models::WdTagger;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Config {
    #[arg()]
    input_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv()?;
    let config = Config::parse();
    let input_dir = dunce::canonicalize(Path::new(&config.input_dir))?;

    let s3_client = Arc::new(create_s3_client()?);
    let qdrant_client = Arc::new(Qdrant::from_url(env::var("QDRANT_URL")?.as_str()).build()?);
    let model = Arc::new(WdTagger::new(0, 16)?);

    let collection_name = Arc::new(env::var("COLLECTION_NAME")?);
    ensure_image_collection_exists(
        &qdrant_client,
        model.output_size as u64,
        collection_name.as_str(),
    )
    .await?;

    let entries = get_image_entries(&input_dir);
    let pb = Arc::new(ProgressBar::new(entries.len() as u64));
    pb.set_style(indicatif::ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}",
    )?);

    process_images(
        entries,
        s3_client,
        qdrant_client,
        model,
        collection_name,
        pb,
    )
    .await?;

    Ok(())
}

fn create_s3_client() -> Result<Client> {
    let credentials_provider = Credentials::new(
        env::var("ACCESS_KEY_ID")?,
        env::var("SECRET_ACCESS_KEY")?,
        None,
        None,
        "s3",
    );
    let config = aws_sdk_s3::Config::builder()
        .behavior_version_latest()
        .credentials_provider(credentials_provider)
        .force_path_style(true)
        .endpoint_url(env::var("ENDPOINT")?)
        .region(Region::new(env::var("REGION")?))
        .build();

    Ok(Client::from_conf(config))
}

async fn ensure_image_collection_exists(
    client: &Qdrant,
    dimension: u64,
    collection_name: &str,
) -> Result<()> {
    if !client.collection_exists(collection_name).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(
                        VectorParamsBuilder::new(dimension, Distance::Cosine).on_disk(true),
                    )
                    .quantization_config(Quantization::Scalar(
                        ScalarQuantizationBuilder::default().build(),
                    )),
            )
            .await?;
    }
    Ok(())
}

fn get_image_entries(input_dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.and_then(|e| Ok(e.into_path())).ok())
        .filter(|e| ImageFormat::from_path(e).is_ok())
        .collect()
}

async fn process_images(
    entries: Vec<PathBuf>,
    s3_client: Arc<Client>,
    qdrant_client: Arc<Qdrant>,
    model: Arc<WdTagger>,
    collection_name: Arc<String>,
    pb: Arc<ProgressBar>,
) -> Result<()> {
    let bucket = Arc::new(env::var("BUCKET_NAME")?);
    let url = Arc::new(format!("{}/{}", env::var("ENDPOINT")?, bucket));
    let cpu = std::thread::available_parallelism()?.get();

    pb.wrap_stream(stream::iter(entries))
        .map(|entry| {
            let s3_client = s3_client.clone();
            let bucket = bucket.clone();
            tokio::spawn(async move { process_single_image(entry, s3_client, bucket).await })
        })
        .buffer_unordered(cpu * 2)
        .map(|x| x?)
        .map(|result| {
            let model = model.clone();
            tokio::spawn(async move { generate_vector(result?, model).await })
        })
        .buffer_unordered(cpu)
        .map(|x| x?)
        .map(|result| {
            let s3_client = s3_client.clone();
            let qdrant_client = qdrant_client.clone();
            let url = url.clone();
            let bucket = bucket.clone();
            let collection_name = collection_name.clone();

            tokio::spawn(async move {
                upload_and_index(
                    result?,
                    s3_client,
                    qdrant_client,
                    url,
                    bucket,
                    collection_name,
                )
                .await
            })
        })
        .buffer_unordered(cpu)
        .map(|x| x?)
        .then(|result| handle_result(result, pb.clone()))
        .collect::<Vec<_>>()
        .await;

    pb.finish();

    Ok(())
}

async fn process_single_image(
    entry: PathBuf,
    s3_client: Arc<Client>,
    bucket: Arc<String>,
) -> Result<(PathBuf, Option<image::DynamicImage>, String)> {
    let mut hasher = blake3::Hasher::new();
    let hash = hasher.update_mmap(&entry)?.finalize().to_hex().to_string();

    let filename = format!("{}.{}", hash, entry.extension().unwrap().to_str().unwrap());

    let object = s3_client
        .get_object()
        .bucket(bucket.to_string())
        .key(&filename)
        .send()
        .await;

    match object {
        Ok(_) => Ok((entry, None, hash)),
        Err(_) => {
            tokio::task::spawn_blocking(move || {
                let file = File::open(&entry)?;
                let mmap = unsafe { Mmap::map(&file)? };
                let img =
                    image::load_from_memory_with_format(&mmap, ImageFormat::from_path(&entry)?)?;
                Ok((entry, Some(img), hash))
            })
            .await?
        }
    }
}

async fn generate_vector(
    data: (PathBuf, Option<image::DynamicImage>, String),
    model: Arc<WdTagger>,
) -> Result<(PathBuf, Option<Vec<f32>>, String)> {
    let (path, img, hash) = data;
    let vec = match img {
        Some(img) => Some(model.predict(&img.into_rgb8()).await?),
        None => None,
    };
    Ok((path, vec, hash))
}

async fn upload_and_index(
    data: (PathBuf, Option<Vec<f32>>, String),
    s3_client: Arc<Client>,
    qdrant_client: Arc<Qdrant>,
    url: Arc<String>,
    bucket: Arc<String>,
    collection_name: Arc<String>,
) -> Result<()> {
    let (path, vec, hash) = data;

    let vec = match vec {
        Some(vec) => vec,
        None => return Ok(()),
    };

    let filename = format!("{}.{}", hash, path.extension().unwrap().to_str().unwrap());
    let full_url = format!("{url}/{filename}");

    // Upload to S3
    let file = File::open(&path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    s3_client
        .put_object()
        .bucket(bucket.to_string())
        .key(&filename)
        .body(ByteStream::from(mmap.to_vec()))
        .send()
        .await?;

    // Index in Qdrant
    let path_str = path.file_name().unwrap().to_str().unwrap().to_string();
    qdrant_client
        .upsert_points(UpsertPointsBuilder::new(
            collection_name.as_str(),
            vec![PointStruct::new(
                uuid::Uuid::new_v4().to_string(),
                vec,
                [
                    ("hash", hash.into()),
                    ("path", path_str.into()),
                    ("url", full_url.into()),
                ],
            )],
        ))
        .await?;

    Ok(())
}

async fn handle_result(result: Result<()>, pb: Arc<ProgressBar>) {
    if let Err(e) = result {
        pb.println(format!("Error: {:?}", e));
    }
}
