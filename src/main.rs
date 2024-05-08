// basics
use std::error::Error;
use std::time::Instant;
// images
use image::io::Reader as ImageReader;

mod inference;
mod utils;

fn benchmark_and_profile<F, T>(func: F) -> (T, f64)
where
    F: FnOnce() -> T,
{
    let start_time = Instant::now();
    let result = func();
    let end_time = Instant::now();
    let execution_time = end_time.duration_since(start_time).as_secs_f64();

    println!("Function: {:?}", std::any::type_name::<F>());
    println!("Execution time: {:.5} seconds", execution_time);

    (result, execution_time)
}

fn main() -> Result<(), Box<dyn Error>> {
    let image = ImageReader::open("./test/test_images/bench.jpeg").unwrap().decode().unwrap();
    let ort_model = inference::OrtModel::new("relative").unwrap();
    benchmark_and_profile(|| ort_model.process(image));

    Ok(())
}
