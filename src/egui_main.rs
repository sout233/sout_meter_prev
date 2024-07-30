use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use hound;
use std::borrow::Borrow;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "WAV Waveform",
        options,
        Box::new(|_cc| Ok(Box::new(WaveformApp::default()))),
    );
}

struct WaveformApp {
    samples: Arc<Mutex<Vec<f32>>>,
    plot_points: Arc<Mutex<PlotPoints>>,
    last_update: Instant,
}

impl Default for WaveformApp {
    fn default() -> Self {
        let samples = read_wav_samples("test.wav");
        let plot_points = Arc::new(Mutex::new(calculate_plot_points(&samples)));
        Self {
            samples: Arc::new(Mutex::new(samples)),
            plot_points,
            last_update: Instant::now(),
        }
    }
}

impl eframe::App for WaveformApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f32();
        if elapsed < (1.0 / 30.0) {
            ctx.request_repaint_after(Duration::from_secs_f32((1.0 / 30.0) - elapsed));
            return;
        }
        self.last_update = Instant::now();

        egui::CentralPanel::default().show(ctx, |ui| {
            let plot_points = self.plot_points.lock().unwrap();
            let points = plot_points.points().iter().map(|p|{
                [p.x, p.y]
            }).collect::<Vec<_>>();
            let line = Line::new(PlotPoints::new(points)); // Clone the plot points for Line
            Plot::new("waveform_plot")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(line);
                });
        });

        ctx.request_repaint();
    }
}

fn read_wav_samples(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / i32::MAX as f32)
        .collect()
}


fn calculate_plot_points(samples: &Vec<f32>) -> PlotPoints {
    (0..samples.len())
        .map(|i| [i as f64, samples[i] as f64])
        .collect()
}