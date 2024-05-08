// arrays/vectors/tensors
use ndarray::{array, Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayView};
use ndarray::{s, Axis, Dim, IxDyn, IxDynImpl};
// images
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, RgbImage};


pub fn image_to_onnx_input(image: DynamicImage) -> Array4<f32> { 
  let mut img_arr = image.to_rgb8().into_vec();
  let (width, height) = image.dimensions();
  let channels = 3;
  let mut onnx_input = Array::zeros((1, channels, height as _, width as _));
  for (x, y, pixel) in image.into_rgb8().enumerate_pixels() {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
    // Set the RGB values in the array
    onnx_input[[0, 0, y as _, x as _]] = (r as f32) / 255.;
    onnx_input[[0, 1, y as _, x as _]] = (g as f32) / 255.;
    onnx_input[[0, 2, y as _, x as _]] = (b as f32) / 255.;
  };
  onnx_input
  //   x_d = np.array(img_d).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)/256 // HWC -> NCHW
  }

pub fn ndarray_to_dynamic_image(data: Array<f32, Dim<IxDynImpl>>, width: u32, height: u32) -> DynamicImage {
  // Assuming `data` is of shape (height, width * 3) and holds RGB data
  let mut img_buf: RgbImage = ImageBuffer::new(width, height);

  // for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
  //     let base_index = (y * width + x) as usize * 3; // 3 channels per pixel
  //     *pixel = image::Rgb([
  //         data[(base_index + 0 as usize)],
  //         data[(base_index + 1 as _)],
  //         data[(base_index + 2 as _)],
  //     ]);
  // }

  for y in 0..height {
    for x in 0..width {
      // let base_index = (y * width + x) as usize * 3; // 3 channels per pixel
      let r = (data[[0, 0, y as usize, x as usize]] * 1.0) as u8;
      let g = (data[[0, 0, y as usize, x as usize]] * 1.0) as u8;
      let b = (data[[0, 0, y as usize, x as usize]] * 1.0) as u8;
      img_buf.put_pixel(x, y, image::Rgb([r, g, b]));
    }
  }

  DynamicImage::ImageRgb8(img_buf)
}