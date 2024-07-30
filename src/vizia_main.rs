use std::time::Duration;
use vizia::prelude::*;
use vizia::vg;

mod wav_process;

pub struct WaveformView<L: Lens<Target = Vec<f32>>> {
    data: L,
}

impl<L: Lens<Target = Vec<f32>>> WaveformView<L> {
    pub fn new(cx: &mut Context, data: L) -> Handle<Self> {
        Self { data }
            .build(cx, |_| {})
            .bind(data, |mut handle, _| handle.needs_redraw())
    }
}

impl<L: Lens<Target = Vec<f32>>> View for WaveformView<L> {
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let data = self.data.get(cx);
        let bounds = cx.bounds();
        let mut path = vg::Path::new();
        let step = bounds.w / data.len() as f32;

        for (i, &value) in data.iter().enumerate() {
            let x = bounds.x + i as f32 * step;
            let y = bounds.y + bounds.h * (1.0 - value);
            if i == 0 {
                path.move_to(x, y);
            } else {
                path.line_to(x, y);
            }
        }

        let paint = vg::Paint::color(Color::black().into()).with_line_width(2.0);
        canvas.stroke_path(&path, &paint);
    }
}

#[derive(Lens)]
struct AppData {
    waveform: Vec<f32>,
}

impl Model for AppData {}

fn main() -> Result<(), ApplicationError> {
    Application::new(|cx| {
        AppData {
            waveform: wav_process::read_wav("test.wav"),
        }
        .build(cx);
        WaveformView::new(cx, AppData::waveform)
            .size(Pixels(400.0))
            .height(Pixels(200.0));
    })
    .run()
}
