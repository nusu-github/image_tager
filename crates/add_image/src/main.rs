use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::LazyLock,
};

use anyhow::Result;
use aws_config::Region;
use aws_sdk_s3::{config::Credentials, primitives::ByteStream, Client};
use clap::Parser;
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

static ENV: LazyLock<utils::Config> = LazyLock::new(|| utils::Config::new());

static BASE_URL: LazyLock<String> =
    LazyLock::new(|| format!("{}/{}", ENV.endpoint, ENV.bucket_name));

static S3_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    let credentials_provider =
        Credentials::new(ENV.access_key_id, ENV.secret_access_key, None, None, "s3");
    let config = aws_sdk_s3::Config::builder()
        .behavior_version_latest()
        .credentials_provider(credentials_provider)
        .force_path_style(true)
        .endpoint_url(ENV.endpoint)
        .region(Region::new(ENV.region))
        .build();

    Client::from_conf(config)
});

static QDRANT_CLIENT: LazyLock<Qdrant> =
    LazyLock::new(|| Qdrant::from_url(ENV.qdrant_url).build().unwrap());

static MODEL: LazyLock<WdTagger> =
    LazyLock::new(|| WdTagger::new(ENV.device_id.parse().unwrap()).unwrap());

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Config {
    input_dir: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();
    let input_dir = dunce::canonicalize(&config.input_dir)?;

    ensure_image_collection_exists().await?;

    let entries = get_image_entries(&input_dir);
    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(indicatif::ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
    )?);

    process_images(entries, pb).await?;

    Ok(())
}

async fn ensure_image_collection_exists() -> Result<()> {
    if !QDRANT_CLIENT.collection_exists(ENV.collection_name).await? {
        QDRANT_CLIENT
            .create_collection(
                CreateCollectionBuilder::new(ENV.collection_name)
                    .vectors_config(
                        VectorParamsBuilder::new(MODEL.output_size as u64, Distance::Cosine)
                            .on_disk(true),
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

async fn process_images(entries: Vec<PathBuf>, pb: ProgressBar) -> Result<()> {
    let cpu = num_cpus::get();

    pb.wrap_stream(stream::iter(entries))
        .map(|path| {
            tokio::spawn(async move {
                process_single_image(&path)
                    .await
                    .map(|(image, hash)| (path, image, hash))
            })
        })
        .buffer_unordered(cpu)
        .map(|x| x?)
        .map(|result| {
            tokio::spawn(async move {
                let (path, image, hash) = result?;
                match image {
                    Some(image) => MODEL
                        .predict(&image.into_rgb8())
                        .await
                        .map(|vector| (path, Some(vector), hash)),
                    None => Ok((path, None, hash)),
                }
            })
        })
        .buffer_unordered(cpu)
        .map(|x| x?)
        .map(|result| {
            tokio::spawn(async move {
                let (path, vector, hash) = result?;
                match vector {
                    Some(vector) => upload_and_index(&path, vector, &hash).await,
                    None => Ok(()),
                }
            })
        })
        .buffer_unordered(cpu)
        .map(|x| x?)
        .then(|result| {
            let pb = pb.clone();
            async move {
                if let Err(e) = result {
                    pb.println(format!("Error: {:?}", e));
                }
            }
        })
        .collect::<Vec<_>>()
        .await;

    pb.finish();

    Ok(())
}

async fn process_single_image(path: &Path) -> Result<(Option<image::DynamicImage>, String)> {
    let mut hasher = blake3::Hasher::new();
    let hash = hasher.update_mmap(path)?.finalize().to_hex().to_string();

    let filename = format!("{}.{}", hash, path.extension().unwrap().to_str().unwrap());

    let object = S3_CLIENT
        .get_object()
        .bucket(ENV.bucket_name)
        .key(&filename)
        .send()
        .await;

    match object {
        Ok(_) => Ok((None, hash)),
        Err(_) => {
            let mmap = unsafe { Mmap::map(&File::open(path)?)? };
            let img = image::load_from_memory_with_format(&mmap, ImageFormat::from_path(path)?)?;
            Ok((Some(img), hash))
        }
    }
}

async fn upload_and_index(path: &Path, vector: Vec<f32>, hash: &str) -> Result<()> {
    let filename = format!("{}.{}", hash, path.extension().unwrap().to_str().unwrap());
    let full_url = format!("{}/{}", BASE_URL.as_str(), filename);

    // Upload to S3
    let mmap = unsafe { Mmap::map(&File::open(&path)?)? };
    S3_CLIENT
        .put_object()
        .bucket(ENV.bucket_name)
        .key(&filename)
        .body(ByteStream::from(mmap.to_vec()))
        .send()
        .await?;

    // Index in Qdrant
    let path_str = path.file_name().unwrap().to_str().unwrap().to_string();
    QDRANT_CLIENT
        .upsert_points(UpsertPointsBuilder::new(
            ENV.collection_name,
            vec![PointStruct::new(
                uuid::Uuid::new_v4().to_string(),
                vector,
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
