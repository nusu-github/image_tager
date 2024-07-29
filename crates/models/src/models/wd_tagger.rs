use anyhow::Result;
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
    pub fn new(device_id: i32, threads: usize) -> Result<Self> {
        let api = Api::new()?;
        let model_path = api.model(MODEL_NAME.parse()?).get("model.onnx")?;

        let session = Session::builder()?
            .with_execution_providers([ort::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;

        let target_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[1].as_();
        let output_size = session.outputs[0].output_type.tensor_dimensions().unwrap()[1].as_();
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
        let input = stack(Axis(0), &[preprocess(image, self.target_size)?.view()])?;
        let outputs = self
            .session
            .run_async(ort::inputs![self.input_name.to_string() => input.view()]?)?
            .await?;
        let outputs = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix2>()?
            .remove_axis(Axis(0))
            .to_vec();

        Ok(outputs)
    }

    pub async fn predicts(&self, image: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
        let images = image
            .iter()
            .map(|image| preprocess(image, self.target_size).unwrap())
            .collect::<Vec<_>>();
        let batch = stack(
            Axis(0),
            &images.iter().map(|t| t.view()).collect::<Vec<_>>(),
        )?;
        let outputs = self
            .session
            .run_async(ort::inputs![self.input_name.to_string() => batch.view()]?)?
            .await?;
        let outputs = outputs[self.output_name.as_str()]
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
    let resized = match max_dim != size {
        true => imageops::resize(&padded, size, size, imageops::FilterType::Lanczos3),
        false => padded,
    };
    let tensor =
        unsafe { ArrayView3::from_shape_ptr((size.as_(), size.as_(), 3), resized.as_ptr()) }
            .slice(s![.., .., ..;-1])
            .mapv(AsPrimitive::as_);

    Ok(tensor)
}
