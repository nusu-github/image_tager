use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use image::{imageops, Rgb, RgbImage};
use ndarray::{prelude::*, stack};
use num_traits::AsPrimitive;
use ort::Session;

const MODEL_NAME: &str = "SmilingWolf/wd-swinv2-tagger-v3";

pub struct Model {
    session: Session,
    pub target_size: u32,
    pub output_size: u32,
    input_name: String,
    output_name: String,
}

impl Model {
    pub fn new(device_id: i32, num_threads: usize) -> Result<Self> {
        let api = Api::new().context("Failed to initialize API")?;
        let model_path = api
            .model(MODEL_NAME.parse()?)
            .get("model.onnx")
            .context("Failed to get model")?;

        let session = Session::builder()?
            .with_execution_providers([ort::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .with_intra_threads(num_threads)?
            .commit_from_file(model_path)?;

        let target_size = session.inputs[0]
            .input_type
            .tensor_dimensions()
            .context("Failed to get input tensor dimensions")?[1]
            .as_();
        let output_size = session.outputs[0]
            .output_type
            .tensor_dimensions()
            .context("Failed to get output tensor dimensions")?[1]
            .as_();
        let input_name = session.inputs[0].name.to_string();
        let output_name = session.outputs[0].name.to_string();

        Ok(Self {
            session,
            target_size,
            output_size,
            input_name,
            output_name,
        })
    }

    pub async fn predict(&self, image: &RgbImage) -> Result<Vec<f32>> {
        let input = stack(Axis(0), &[preprocess(image, self.target_size)?.view()])
            .context("Failed to stack input tensors")?;
        let outputs = self
            .session
            .run_async(ort::inputs![self.input_name.clone() => input.view()]?)
            .context("Failed to run session")?
            .await?;
        outputs[self.output_name.as_ref()]
            .try_extract_raw_tensor()
            .map(|(_, tensor)| tensor.to_vec())
            .context("Failed to extract raw tensor")
    }

    pub async fn predicts(&self, images: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
        let images: Vec<_> = images
            .iter()
            .map(|image| preprocess(image, self.target_size))
            .collect::<Result<_>>()?;
        let batch = stack(
            Axis(0),
            &images.iter().map(ArrayBase::view).collect::<Vec<_>>(),
        )
            .context("Failed to stack batch of images")?;
        let outputs = self
            .session
            .run_async(ort::inputs![self.input_name.clone() => batch.view()]?)
            .context("Failed to run session")?
            .await?;
        let outputs = outputs[self.output_name.as_ref()]
            .try_extract_tensor::<f32>()
            .context("Failed to extract tensor")?
            .into_dimensionality::<Ix2>()
            .context("Failed to convert tensor dimensionality")?
            .axis_iter(Axis(0))
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        Ok(outputs)
    }
}

fn preprocess(image: &RgbImage, size: u32) -> Result<Array3<f32>> {
    let (w, h) = image.dimensions();
    let max_dim = w.max(h);
    let pad = |x| ((max_dim - x) / 2) as i64;
    let mut padded = RgbImage::from_pixel(max_dim, max_dim, Rgb([255, 255, 255]));
    imageops::overlay(&mut padded, image, pad(w), pad(h));
    let resized = imageops::resize(&padded, size, size, imageops::FilterType::Lanczos3);
    let tensor = Array3::from_shape_vec((size as usize, size as usize, 3), resized.into_raw())
        .context("Failed to create tensor from shape vector")?
        .slice(s![.., .., ..;-1])
        .mapv(AsPrimitive::as_);

    Ok(tensor)
}
