// basics
use std::error::Error;
use std::path::{Path, PathBuf};
// arrays/vectors/tensors
use ndarray::{array, Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayView};
use ndarray::{s, Axis, Dim, IxDyn, IxDynImpl};
// images
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, RgbImage};
// machine learning
use ort::{Session, GraphOptimizationLevel};

use crate::utils::*;


pub struct OrtModel {
    onnx_file: PathBuf,
    model: Session
}

impl OrtModel {
    pub fn new(mode: &str) -> Result<Self, Box<dyn Error>> {
        let name = match mode {
            "relative" => "depth_anything_small",
            "metric" => "depth_anything_small", // TODO
            _ => return Err("Invalid model".into())
        };
        let onnx_file = PathBuf::from(format!("./assets/models/{}.onnx", name));
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(onnx_file.clone())?;
        
        Ok(OrtModel { onnx_file, model })
    }

    pub fn process(self, image: DynamicImage) -> Result<DynamicImage, Box<dyn Error>> {
        let (width, height) = image.dimensions();
        let image_array = image_to_onnx_input(image.clone());
        let inputs = ort::inputs!["image" => image_array.view()]?; 
        let outputs = self.model.run(inputs)?;
        
        let pred = outputs["depth"].try_extract_tensor::<f32>()?.view().clone().into_owned(); // rename; good or bad?
        let pred_image = ndarray_to_dynamic_image(pred.clone(), width, height); // rename; good or bad?
        // println!("{:?}", pred_image.clone());

        // let pred_2d = pred.into_dimensionality::<Dim<[usize; 2]>>().unwrap();

        // maximize_contrast(pred_2d);

        Ok( pred_image )
    }
        
}