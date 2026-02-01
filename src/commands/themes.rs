use anyhow::Result;

pub async fn run() -> Result<()> {
    let themes = syntect::highlighting::ThemeSet::load_defaults();
    println!("Available themes:");
    println!();
    let mut names: Vec<&String> = themes.themes.keys().collect();
    names.sort();
    for name in names {
        println!("  - {}", name);
    }
    Ok(())
}
