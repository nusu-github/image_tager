use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Ok, Result};
use clap::Parser;
use dotenvy::dotenv;
use futures_util::{stream, StreamExt, TryStreamExt};
use image::ImageFormat;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::{prelude::*, stack};
use qdrant_client::{
    qdrant::{
        RecommendExample, RecommendPointsBuilder, RecommendResponse, RecommendStrategy,
        SearchParamsBuilder,
    },
    Qdrant,
};
use tokio::{fs, io};
use walkdir::WalkDir;

use models::WdTagger;

// Constants
const DEFAULT_LIMIT: usize = 512;
const QDRANT_SEARCH_EXACT: bool = true;
const QDRANT_SEARCH_HNSW_EF: u64 = 64;
const QDRANT_SCORE_THRESHOLD: f32 = 0.5;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Config {
    #[arg()]
    input: String,

    #[arg()]
    output: Option<String>,

    #[arg(short, long, default_value_t = DEFAULT_LIMIT)]
    limit: usize,

    #[arg(short, long, default_value_t = QDRANT_SCORE_THRESHOLD)]
    score_threshold: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv()?;
    let config = Config::parse();

    let input = dunce::canonicalize(Path::new(&config.input))
        .with_context(|| "Failed to canonicalize input path")?;
    let output = get_output_path(&input, &config.output)
        .with_context(|| "Failed to determine output path")?;
    fs::create_dir_all(&output)
        .await
        .with_context(|| "Failed to create output directory")?;

    let input_entries = if input.is_dir() {
        get_image_entries(&input)
    } else {
        validate_single_image_input(&input)?
    };

    let model = WdTagger::new(0, 2).with_context(|| "Failed to initialize WdTagger model")?;

    let client = create_qdrant_client().with_context(|| "Failed to create Qdrant client")?;
    let collection_name = env::var("COLLECTION_NAME")
        .with_context(|| "Failed to get COLLECTION_NAME from environment")?;

    let batch_size = std::thread::available_parallelism()
        .with_context(|| "Failed to get available parallelism")?
        .get();

    let bars = MultiProgress::new();

    for (entry, files) in input_entries {
        let entry_output = output.join(&entry);
        fs::create_dir_all(&entry_output)
            .await
            .with_context(|| "Failed to create entry output directory")?;

        process_entry(
            &entry,
            &files,
            &entry_output,
            batch_size,
            config.limit as u64,
            config.score_threshold,
            &client,
            &collection_name,
            &model,
            &bars,
        )
        .await
        .with_context(|| "Failed to process entry")?;
    }

    Ok(())
}

fn get_output_path(input: &Path, output: &Option<String>) -> Result<PathBuf> {
    match output {
        None => Ok(match input.is_dir() {
            true => input.parent().unwrap().join("output"),
            false => input.parent().unwrap().to_path_buf(),
        }),
        Some(output) => Ok(PathBuf::from(output)),
    }
}

fn validate_single_image_input(input: &Path) -> Result<Vec<(String, Vec<PathBuf>)>> {
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

fn create_qdrant_client() -> Result<Qdrant> {
    let qdrant_url =
        env::var("QDRANT_URL").with_context(|| "Failed to get QDRANT_URL from environment")?;
    Qdrant::from_url(qdrant_url.as_str())
        .build()
        .with_context(|| "Failed to build Qdrant client")
}

async fn process_entry(
    tag: &str,
    files: &[PathBuf],
    output: &PathBuf,
    batch_size: usize,
    limit: u64,
    score_threshold: f32,
    client: &Qdrant,
    collection_name: &str,
    model: &WdTagger,
    bars: &MultiProgress,
) -> Result<()> {
    let pb = bars.add(create_progress_bar()?);
    pb.set_message(tag.to_string());

    let vecs = process_images(files, batch_size, model, &pb).await?;
    let vecs = reduce_vectors(vecs, model.output_size as usize)?;

    let req = build_recommend_request(collection_name, limit, score_threshold, vecs);
    let res = client
        .recommend(req)
        .await
        .with_context(|| format!("Tag: {tag} failed to search"))?;

    let files = filter_and_extract_files(&res);

    download_files(files, output, &pb).await?;

    pb.finish();
    Ok(())
}

fn create_progress_bar() -> Result<ProgressBar> {
    let pb = ProgressBar::new(0).with_finish(indicatif::ProgressFinish::Abandon);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise} {bar:60.green/blue}] {pos:5}/{len:5} {msg}",
    )?);
    Ok(pb)
}

async fn process_images(
    files: &[PathBuf],
    batch_size: usize,
    model: &WdTagger,
    pb: &ProgressBar,
) -> Result<Vec<Vec<f32>>> {
    let images_batch = files
        .chunks(batch_size)
        .map(|chunk| {
            chunk
                .iter()
                .map(|e| Ok(image::open(e)?.into_rgb8()))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    pb.set_length(images_batch.len() as u64);
    pb.set_position(0);

    pb.wrap_stream(stream::iter(images_batch))
        .then(|batch| async move { model.predicts(&batch).await })
        .try_fold(vec![], |mut vecs, batch| async move {
            vecs.extend(batch);
            Ok(vecs)
        })
        .await
}

fn reduce_vectors(vecs: Vec<Vec<f32>>, output_size: usize) -> Result<Vec<Vec<f32>>> {
    if vecs.len() <= 32 {
        return Ok(vecs);
    }

    let len = vecs.len();
    let chunk_size = 1 + len.div_ceil(32);

    vecs.chunks(chunk_size)
        .map(|chunk| {
            let vecs = chunk
                .into_iter()
                .map(|x| Ok(ArrayView1::from_shape(output_size, x)?))
                .collect::<Result<Vec<_>>>()?;
            let vecs = stack(Axis(0), &vecs)?;
            Ok(vecs.mean_axis(Axis(0)).unwrap().into_raw_vec())
        })
        .collect::<Result<Vec<_>>>()
}

fn build_recommend_request(
    collection_name: &str,
    limit: u64,
    score_threshold: f32,
    vecs: Vec<Vec<f32>>,
) -> RecommendPointsBuilder {
    vecs.into_iter()
        .fold(
            RecommendPointsBuilder::new(collection_name, limit),
            |builder, vec| builder.add_positive(RecommendExample::from(vec)),
        )
        .with_payload(true)
        .params(
            SearchParamsBuilder::default()
                .exact(QDRANT_SEARCH_EXACT)
                .hnsw_ef(QDRANT_SEARCH_HNSW_EF),
        )
        .strategy(RecommendStrategy::BestScore)
        .score_threshold(score_threshold)
}

fn filter_and_extract_files(res: &RecommendResponse) -> Vec<(String, String)> {
    res.result
        .iter()
        .map(|x| {
            let url = x.payload.get("url").unwrap().as_str().unwrap();
            let path = x.payload.get("path").unwrap().as_str().unwrap();
            (url.to_string(), path.to_string())
        })
        .collect()
}

async fn download_files(
    files: Vec<(String, String)>,
    output: &PathBuf,
    pb: &ProgressBar,
) -> Result<()> {
    pb.set_length(files.len() as u64);
    pb.set_position(0);

    pb.wrap_stream(stream::iter(files))
        .map(|x| {
            let output = output.clone();
            tokio::spawn(async move {
                let (url, file_path) = x;
                let path = output.join(file_path);
                let response: reqwest::Response = reqwest::get(url).await?;
                let bytes = response.bytes().await?;
                let mut file = fs::File::create(&path).await?;
                io::copy(&mut bytes.as_ref(), &mut file).await?;
                anyhow::Ok(())
            })
        })
        .buffer_unordered(4)
        .try_collect::<Vec<_>>()
        .await?;

    Ok(())
}

fn get_image_entries<P: AsRef<Path>>(root_path: P) -> Vec<(String, Vec<PathBuf>)> {
    WalkDir::new(root_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_dir())
        .filter_map(|entry| {
            let path = entry.path();
            let folder_name = path.file_name()?.to_str()?.to_string();

            let image_files: Vec<PathBuf> = path
                .read_dir()
                .ok()?
                .filter_map(Result::ok)
                .map(|file_entry| file_entry.path())
                .filter(|file_path| {
                    file_path.is_file() && ImageFormat::from_path(file_path).is_ok()
                })
                .collect();

            (!image_files.is_empty()).then_some((folder_name, image_files))
        })
        .collect()
}
