use std::time::Duration;
use vizia::prelude::*;
use vizia::vg;

pub struct LineView<L: Lens<Target = f32>> {
    position: L,
}

impl<L: Lens<Target = f32>> LineView<L> {
    pub fn new(cx: &mut Context, position: L) -> Handle<Self> {
        Self { position }
            .build(cx, |_| {})
            .bind(position, |mut handle, _| handle.needs_redraw())
    }
}

impl<L: Lens<Target = f32>> View for LineView<L> {
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let position = self.position.get(cx);
        let bounds = cx.bounds();
        let mut path = vg::Path::new();
        path.move_to(bounds.x + position, bounds.y + 10.0);
        path.line_to(bounds.x + position + 200.0, bounds.y + 200.0);
        let paint = vg::Paint::color(Color::black().into()).with_line_width(2.0);
        canvas.stroke_path(&path, &paint);
    }
}

#[derive(Lens)]
struct AppData {
    position: f32,
    direction: f32,
}

impl Model for AppData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|app_event, _| match app_event {
            AppEvent::UpdatePosition => {
                self.position += self.direction;
                if self.position > 200.0 || self.position < 0.0 {
                    self.direction = -self.direction;
                }
            }
        })
    }
}

pub enum AppEvent {
    UpdatePosition,
}

fn main() -> Result<(), ApplicationError> {
    Application::new(|cx| {
        AppData {
            position: 50.0,
            direction: 2.0,
        }
        .build(cx);
        LineView::new(cx, AppData::position).size(Pixels(400.0));
        let timer = cx.add_timer(Duration::from_millis(33), None, |cx, _| {
            cx.emit(AppEvent::UpdatePosition);
        });
        cx.start_timer(timer);
    })
    .run()
}