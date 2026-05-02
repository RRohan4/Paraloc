use paraloc::renderer::RendererPlugin;

fn main() {
    bevy::app::App::new()
        .add_plugins(RendererPlugin)
        .run();
}
