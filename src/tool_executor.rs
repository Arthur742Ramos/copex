use std::io;
use std::process::Command;

#[allow(dead_code)]
pub fn read_file(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}

#[allow(dead_code)]
pub fn write_file(path: &str, content: &str) -> Result<(), io::Error> {
    std::fs::write(path, content)
}

#[allow(dead_code)]
pub fn run_command(command: &str) -> Result<String, io::Error> {
    let output = Command::new("sh").arg("-c").arg(command).output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let message = format!(
            "Command failed (status {:?}): {}{}",
            output.status.code(),
            stdout.trim_end(),
            if stderr.is_empty() {
                String::new()
            } else {
                format!("\n{}", stderr.trim_end())
            }
        );
        return Err(io::Error::new(io::ErrorKind::Other, message.trim().to_string()));
    }
    let mut combined = stdout.to_string();
    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&stderr);
    }
    Ok(combined)
}
