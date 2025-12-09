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
        .add_systems(Startup, |mut imgui: NonSendMut<ImguiContext>| {
            // Use `with_io_mut` to get mutable access to the underlying ImGui io structure
            // for configuration...
            imgui.with_io_mut(|io| {
                io.config_docking_always_tab_bar = true;
            });
        })
        .add_systems(Update, imgui_example_ui);
    app.run();
}

#[cfg(feature = "docking")]
fn imgui_example_ui(mut context: NonSendMut<ImguiContext>) {
    let ui = context.ui();

    // Create a dockspace that covers the entire viewport (so that windows can be docked to it)
    ui.dockspace_over_main_viewport();

    // Create some windows that can be docked...
    let window_a = ui.window("Window A");
    window_a
        .size([300.0, 100.0], imgui::Condition::FirstUseEver)
        .position([0.0, 0.0], imgui::Condition::FirstUseEver)
        .build(|| {
            ui.text("This is Window A!");
            ui.text("Drag the window tabs to dock them");
        });

    let window_b = ui.window("Window B");
    window_b
        .size([300.0, 100.0], imgui::Condition::FirstUseEver)
        .position([350.0, 0.0], imgui::Condition::FirstUseEver)
        .build(|| {
            ui.text("This is Window B!");
        });

    let window_c = ui.window("Window C");
    window_c
        .size([300.0, 100.0], imgui::Condition::FirstUseEver)
        .position([0.0, 150.0], imgui::Condition::FirstUseEver)
        .build(|| {
            ui.text("This is Window C!");
        });
}
