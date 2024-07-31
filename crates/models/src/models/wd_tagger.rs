use std::sync::OnceLock;

use anyhow::Result;
use hf_hub::api::sync::Api;
use image::{imageops, Rgb, RgbImage};
use ndarray::{prelude::*, stack};
use num_traits::AsPrimitive;
use ort::Session;

static SESSION: OnceLock<Session> = OnceLock::new();
const MODEL_NAME: &str = "SmilingWolf/wd-swinv2-tagger-v3";

#[derive(Copy, Clone)]
pub struct Model {
    session: &'static Session,
    pub target_size: u32,
    pub output_size: u32,
    input_name: &'static str,
    output_name: &'static str,
}

impl Model {
    pub fn new(device_id: i32) -> Result<Self> {
        let api = Api::new()?;
        let model_path = api.model(MODEL_NAME.parse()?).get("model.onnx")?;

        let session = Session::builder()?
            .with_execution_providers([ort::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .commit_from_file(model_path)?;

        let target_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[1].as_();
        let output_size = session.outputs[0].output_type.tensor_dimensions().unwrap()[1].as_();
        let input_name = session.inputs[0].name.to_string();
        let output_name = session.outputs[0].name.to_string();

        Ok(Self {
            session: SESSION.get_or_init(|| session),
            target_size,
            output_size,
            input_name: Box::leak(input_name.into_boxed_str()),
            output_name: Box::leak(output_name.into_boxed_str()),
        })
    }

    pub async fn predict(&self, image: &RgbImage) -> Result<Vec<f32>> {
        let input = stack(Axis(0), &[preprocess(image, self.target_size)?.view()])?;
        let outputs = self
            .session
            .run_async(ort::inputs![self.input_name => input.view()]?)?
            .await?;
        let outputs = outputs[self.output_name]
            .try_extract_raw_tensor()?
            .1
            .to_vec();

        Ok(outputs)
    }

    pub async fn predicts(&self, image: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
        let images: Vec<_> = image
            .iter()
            .map(|image| preprocess(image, self.target_size).unwrap())
            .collect();
        let batch = stack(
            Axis(0),
            &images.iter().map(|t| t.view()).collect::<Vec<_>>(),
        )?;
        let outputs = self
            .session
            .run_async(ort::inputs![self.input_name => batch.view()]?)?
            .await?;
        let outputs = outputs[self.output_name]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix2>()?
            .axis_iter(Axis(0))
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        Ok(outputs)
    }
}

fn preprocess(image: &RgbImage, size: u32) -> Result<Array3<f32>> {
    let (w, h) = image.dimensions();
    let max_dim = w.max(h);
    let pad = |x| <_ as AsPrimitive<_>>::as_((max_dim - x) / 2);
    let mut padded = RgbImage::from_pixel(max_dim, max_dim, Rgb([255, 255, 255]));
    imageops::overlay(&mut padded, image, pad(w), pad(h));
    let resized = imageops::resize(&padded, size, size, imageops::FilterType::Lanczos3);
    let tensor = Array3::from_shape_vec((size.as_(), size.as_(), 3), resized.into_raw())?
        .slice(s![.., .., ..;-1])
        .mapv(AsPrimitive::as_);

    Ok(tensor)
}
