use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::{Arc, Mutex};
use std::thread;
use vizia::prelude::*;
use vizia::vg::{self, Color4f, ColorSpace, Point};

#[allow(unused)]
const FREQ_RANGE_START_HZ: f32 = 30.0;
#[allow(unused)]
const FREQ_RANGE_END_HZ: f32 = 22_000.0;
const LN_FREQ_RANGE_START_HZ: f32 = 3.4011974; // 30.0f32.ln();
const LN_FREQ_RANGE_END_HZ: f32 = 9.998797; // 22_000.0f32.ln();
const LN_FREQ_RANGE: f32 = LN_FREQ_RANGE_END_HZ - LN_FREQ_RANGE_START_HZ;
pub const MINUS_INFINITY_GAIN: f32 = 1e-5; // 10f32.powf(MINUS_INFINITY_DB / 20)

fn main() -> Result<(), ApplicationError> {
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let spectrum_buffer = Arc::new(Mutex::new(Vec::new()));

    // 音频输入线程？启动！
    let buffer_clone = audio_buffer.clone();
    let spectrum_clone = spectrum_buffer.clone();
    thread::spawn(move || {
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
            let mut buffer = buffer_clone.lock().unwrap();
            // Downsample 但是 factor 10
            let downsampled_data: Vec<f32> = data.iter().step_by(10).copied().collect();
            buffer.extend(downsampled_data);

            // 更新spectrum cache
            let mut spectrum = spectrum_clone.lock().unwrap();
            let window_size = 1024;
            if buffer.len() >= window_size {
                let mut fft_buffer: Vec<Complex<f32>> = buffer[buffer.len() - window_size..]
                    .iter()
                    .map(|&x| Complex { re: x, im: 0.0 })
                    .collect();

                let mut planner = FftPlanner::<f32>::new();
                let fft = planner.plan_fft_forward(window_size);
                fft.process(&mut fft_buffer);

                spectrum.clear();
                spectrum.extend(fft_buffer.iter().map(|c| c.norm()));
            }
        };

        let stream = match sample_format {
            _ => device.build_input_stream(&config, input_data_fn, err_fn, None),
        }
        .unwrap();

        stream.play().unwrap();

        loop {
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    });

    Application::new(move |cx| {
        AppData {
            waveform: audio_buffer.clone(),
            spectrum: spectrum_buffer.clone(),
        }
        .build(cx);

        HStack::new(cx, |cx| {
            WaveformView::new(cx, AppData::waveform.clone())
                .size(Pixels(500.0))
                .height(Pixels(100.0));

            SpectrumView::new(cx, AppData::waveform.clone(), AppData::spectrum.clone())
                .size(Pixels(800.0))
                .height(Pixels(100.0))
                .top(Pixels(100.0));
        }).on_press_down(|cx| {
            println!("dragging window");
            cx.emit(WindowEvent::DragWindow);
        });
        let timer = cx.add_timer(Duration::from_millis(33), None, |cx, _| {
            cx.emit(AppEvent::UpdateWaveform);
        });

        cx.start_timer(timer);
    })
    .inner_size((800 + 500, 200))
    .always_on_top(true)
    .title("sout meter test")
    .decorations(false)
    .run()
}

#[derive(Lens)]
struct AppData {
    waveform: Arc<Mutex<Vec<f32>>>,
    spectrum: Arc<Mutex<Vec<f32>>>,
}

impl Model for AppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|app_event, _| match app_event {
            AppEvent::UpdateWaveform => {
                cx.needs_redraw();
            }
        });
    }
}

pub enum AppEvent {
    UpdateWaveform,
}

pub struct WaveformView<L: Lens<Target = Arc<Mutex<Vec<f32>>>>> {
    data: L,
}

impl<L> WaveformView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    pub fn new(cx: &mut Context, data: L) -> Handle<Self> {
        Self { data }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L> View for WaveformView<L>
where
    L: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        let mut data = self.data.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();
        let max_samples = (bounds.w as usize).min(data.len());

        if data.len() > max_samples {
            data.drain(..data.len() - max_samples);
        }

        let mut path = vg::Path::new();
        let step = if data.len() > 1 {
            bounds.w / (data.len() - 1) as f32
        } else {
            0.0
        };

        for (i, &value) in data.iter().enumerate() {
            let x = bounds.x + i as f32 * step;
            let y = bounds.y + bounds.h * (1.0 - value);
            if i == 0 {
                path.move_to(Point::new(x, y));
            } else {
                path.line_to(Point::new(x, y));
            }
        }

        // let paint = vg::Paint::color(Color::black().into()).with_line_width(2.0);
        let mut paint = vg::Paint::new(Color4f::new(0.0, 0.0, 0.0, 1.0), None);
        paint.set_stroke_width(2.0);
        paint.set_anti_alias(true);
        canvas.draw_path(&path, &paint);
    }
}

// 频谱视图
pub struct SpectrumView<L1, L2>
where
    L1: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    L2: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    data: L1,
    spectrum: L2,
}

impl<L1, L2> SpectrumView<L1, L2>
where
    L1: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    L2: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    pub fn new(cx: &mut Context, data: L1, spectrum: L2) -> Handle<Self> {
        Self { data, spectrum }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
            .bind(spectrum, |mut handle, _| handle.needs_redraw())
    }
}

impl<L1, L2> View for SpectrumView<L1, L2>
where
    L1: Lens<Target = Arc<Mutex<Vec<f32>>>>,
    L2: Lens<Target = Arc<Mutex<Vec<f32>>>>,
{
    fn draw(&self, cx: &mut DrawContext, canvas: &Canvas) {
        let data = self.data.get(cx).lock().unwrap().clone();
        let spectrum = self.spectrum.get(cx).lock().unwrap().clone();
        let bounds = cx.bounds();

        if data == spectrum {
            return;
        }

        // 窗函数很重要
        let window_size = 1024;
        if data.len() < window_size {
            return;
        }

        let mut buffer: Vec<Complex<f32>> = data[data.len() - window_size..]
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        fft.process(&mut buffer);

        let magnitude: Vec<f32> = buffer.iter().map(|c| c.norm()).collect();
        let max_magnitude = magnitude.iter().cloned().fold(0. / 0., f32::max);

        let line_width = cx.scale_factor() * 1.5;
        // let text_color: vg::Color = vg::Color::rgb(136,20,0);
        let text_color: vg::Color = vg::Color::BLACK;
        // let bars_paint = vg::Paint::color(text_color).with_line_width(line_width);
        let bars_paint = vg::Paint::new(Color4f::new(0.0, 0.0, 0.0, 1.0), None);
        let mut lighter_text_color = text_color;
        // lighter_text_color.r = (lighter_text_color.r + 0.25) / 1.25;
        // lighter_text_color.g = (lighter_text_color.g + 0.25) / 1.25;
        // lighter_text_color.b = (lighter_text_color.b + 0.25) / 1.25;

        let nyquist_hz = 48000.0 / 2.0; // TODO: 这玩意应该dynamic的
        let bin_frequency = |bin_idx: f32| (bin_idx / (window_size as f32 / 2.0)) * nyquist_hz;
        let bin_t =
            |bin_idx: f32| (bin_frequency(bin_idx).ln() - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE;
        let magnitude_height = |magnitude: f32| {
            let magnitude_db = gain_to_db(magnitude);
            db_to_unclamped_t(magnitude_db).clamp(0.0, 1.0)
        };

        let mesh_start_delta_threshold = line_width + 0.5;
        let mut mesh_bin_start_idx = window_size / 2;
        let mut previous_physical_x_coord = bounds.x - 2.0;

        let mut bars_path = vg::Path::new();
        for (bin_idx, &magnitude) in magnitude.iter().enumerate().take(window_size / 2) {
            let t = bin_t(bin_idx as f32);
            if t <= 0.0 || t >= 1.0 {
                continue;
            }

            let physical_x_coord = bounds.x + (bounds.w * t);
            if physical_x_coord - previous_physical_x_coord < mesh_start_delta_threshold {
                mesh_bin_start_idx = bin_idx.saturating_sub(1);
                previous_physical_x_coord = physical_x_coord;
                break;
            }

            let height = magnitude_height(magnitude / max_magnitude);
            bars_path.move_to(Point::new(
                physical_x_coord,
                bounds.y + (bounds.h * (1.0 - height)),
            ));
            bars_path.line_to(Point::new(physical_x_coord, bounds.y + bounds.h));

            previous_physical_x_coord = physical_x_coord;
        }
        canvas.draw_path(&bars_path, &bars_paint);

        let mut mesh_path = vg::Path::new();
        let mesh_start_x_coordiante = bounds.x + (bounds.w * bin_t(mesh_bin_start_idx as f32));
        let mesh_start_y_coordinate = bounds.y + bounds.h;

        mesh_path.move_to(Point::new(mesh_start_x_coordiante, mesh_start_y_coordinate));
        for (bin_idx, &magnitude) in magnitude
            .iter()
            .enumerate()
            .take(window_size / 2)
            .skip(mesh_bin_start_idx)
        {
            let t = bin_t(bin_idx as f32);
            if t <= 0.0 || t >= 1.0 {
                continue;
            }

            let physical_x_coord = bounds.x + (bounds.w * t);
            previous_physical_x_coord = physical_x_coord;
            let height = magnitude_height(magnitude / max_magnitude);
            if height > 0.0 {
                mesh_path.line_to(Point::new(
                    physical_x_coord,
                    bounds.y + (bounds.h * (1.0 - height) - (line_width / 2.0)).max(0.0),
                ));
            } else {
                mesh_path.line_to(Point::new(physical_x_coord, mesh_start_y_coordinate));
            }
        }

        mesh_path.line_to(Point::new(
            previous_physical_x_coord,
            mesh_start_y_coordinate,
        ));
        mesh_path.close();

        // let mesh_paint = vg::Paint::linear_gradient_stops(
        //     mesh_start_x_coordiante,
        //     0.0,
        //     previous_physical_x_coord,
        //     0.0,
        //     [
        //         (0.0, lighter_text_color),
        //         (0.707, text_color),
        //         (1.0, text_color),
        //     ],
        // )
        // .with_anti_alias(false);
        let mesh_paint = vg::Paint::new(Color4f::new(0.0, 0.0, 0.0, 1.0), None);

        canvas.draw_path(&mesh_path, &mesh_paint);
    }
}

pub fn gain_to_db(gain: f32) -> f32 {
    f32::max(gain, MINUS_INFINITY_GAIN).log10() * 20.0
}

#[inline]
fn db_to_unclamped_t(db_value: f32) -> f32 {
    (db_value + 80.0) / 100.0
}
