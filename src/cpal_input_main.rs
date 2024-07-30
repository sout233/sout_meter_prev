use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

fn main() {
    let host = cpal::default_host();

    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

    let device = host.default_input_device().unwrap();

    let mut supported_configs_range = device.supported_input_configs().unwrap();
    let supported_config = supported_configs_range
        .next()
        .unwrap()
        .with_max_sample_rate();

    let sample_format = supported_config.sample_format();

    let config = supported_config.into();

    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let output_fell_behind = false;
        for &sample in data {
            let sample: f32 = sample.into();
            println!("{:?}", sample);
        }
        if output_fell_behind {
            eprintln!("output stream fell behind: try increasing latency");
        }
    };

    let stream = match sample_format {
        SampleFormat::U8 => device.build_input_stream(&config, input_data_fn, err_fn, None),
        _ => device.build_input_stream(&config, input_data_fn, err_fn, None),
        // sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
    .unwrap();

    stream.play().unwrap();

    loop {
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
