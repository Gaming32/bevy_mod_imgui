// Example that matches the minimal docking example in the documentation.
// In the documentation, the #cfg and not docking main function is stripped for brevity.
#[cfg(feature = "docking")]
use bevy::prelude::*;

#[cfg(feature = "docking")]
use bevy_mod_imgui::prelude::*;

#[cfg(not(feature = "docking"))]
fn main() -> Result<(), ()> {
    println!("Error: `docking` feature is not enabled. Please compile with `--features docking`");
    Err(())
}

#[cfg(feature = "docking")]
fn main() {
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgba(0.2, 0.2, 0.2, 1.0)))
        .add_plugins(DefaultPlugins)
        .add_plugins(bevy_mod_imgui::ImguiPlugin {
            ..Default::default()
        })
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera3d::default());
        })
        .add_systems(Update, imgui_example_ui);
    app.run();
}

#[cfg(feature = "docking")]
fn imgui_example_ui(mut context: NonSendMut<ImguiContext>) {
    let ui = context.ui();
    ui.dockspace_over_main_viewport();
    let window = ui.window("Drag me");
    window
        .size([300.0, 100.0], imgui::Condition::FirstUseEver)
        .position([0.0, 0.0], imgui::Condition::FirstUseEver)
        .build(|| {
            ui.text("Drag the window title-bar to dock it!");
        });
}
