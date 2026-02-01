use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const STATE_FILE_NAME: &str = ".copex-state.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub number: usize,
    pub description: String,
    pub status: StepStatus,
    pub result: Option<String>,
    pub error: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub tokens_used: u64,
}

impl PlanStep {
    pub fn new(number: usize, description: String) -> Self {
        Self {
            number,
            description,
            status: StepStatus::Pending,
            result: None,
            error: None,
            started_at: None,
            completed_at: None,
            tokens_used: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub task: String,
    pub steps: Vec<PlanStep>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub total_tokens: u64,
}

impl Plan {
    pub fn new(task: String, steps: Vec<PlanStep>) -> Self {
        Self {
            task,
            steps,
            created_at: Utc::now(),
            completed_at: None,
            total_tokens: 0,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.steps.iter().all(|step| {
            matches!(step.status, StepStatus::Completed | StepStatus::Skipped)
        })
    }

    pub fn current_step(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|step| step.status == StepStatus::Pending)
    }

    #[allow(dead_code)]
    pub fn completed_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| matches!(step.status, StepStatus::Completed | StepStatus::Skipped))
            .count()
    }

    #[allow(dead_code)]
    pub fn failed_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| step.status == StepStatus::Failed)
            .count()
    }

    #[allow(dead_code)]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let payload = serde_json::to_string_pretty(self)?;
        fs::write(path, payload).context("Failed to write plan")?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let data = fs::read_to_string(path).context("Failed to read plan")?;
        Ok(serde_json::from_str(&data)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanState {
    pub task: String,
    pub plan: Plan,
    pub completed: Vec<usize>,
    pub current_step: usize,
    pub step_results: std::collections::HashMap<String, String>,
    pub started_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub total_tokens: u64,
}

impl PlanState {
    pub fn from_plan(plan: Plan) -> Self {
        Self {
            task: plan.task.clone(),
            completed: plan
                .steps
                .iter()
                .filter(|s| s.status == StepStatus::Completed)
                .map(|s| s.number)
                .collect(),
            current_step: plan
                .current_step()
                .map(|s| s.number)
                .unwrap_or(plan.steps.len() + 1),
            step_results: plan
                .steps
                .iter()
                .filter_map(|s| s.result.as_ref().map(|r| (s.number.to_string(), r.clone())))
                .collect(),
            started_at: plan.created_at,
            last_updated: Utc::now(),
            total_tokens: plan.total_tokens,
            plan,
        }
    }

    pub fn save(&mut self, path: Option<PathBuf>) -> Result<PathBuf> {
        let path = path.unwrap_or_else(|| PathBuf::from(STATE_FILE_NAME));
        self.last_updated = Utc::now();
        let payload = serde_json::to_string_pretty(self)?;
        fs::write(&path, payload)?;
        Ok(path)
    }

    #[allow(dead_code)]
    pub fn load(path: Option<PathBuf>) -> Result<Option<Self>> {
        let path = path.unwrap_or_else(|| PathBuf::from(STATE_FILE_NAME));
        if !path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&path)?;
        let state = serde_json::from_str(&data)?;
        Ok(Some(state))
    }

    pub fn cleanup(path: Option<PathBuf>) -> Result<bool> {
        let path = path.unwrap_or_else(|| PathBuf::from(STATE_FILE_NAME));
        if path.exists() {
            fs::remove_file(path)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn update_step(&mut self, step: &PlanStep) {
        self.current_step = step.number + 1;
        if step.status == StepStatus::Completed {
            if !self.completed.contains(&step.number) {
                self.completed.push(step.number);
            }
            if let Some(result) = &step.result {
                self.step_results
                    .insert(step.number.to_string(), result.chars().take(500).collect());
            }
            self.total_tokens += step.tokens_used;
        }
        self.last_updated = Utc::now();
    }
}

pub struct PlanExecutor {
    #[allow(dead_code)]
    pub max_iterations_per_step: usize,
}

impl PlanExecutor {
    pub fn new() -> Self {
        Self {
            max_iterations_per_step: 10,
        }
    }

    pub fn new_with_max(max_iterations_per_step: usize) -> Self {
        Self {
            max_iterations_per_step,
        }
    }

    pub fn parse_steps(&self, content: &str) -> Vec<PlanStep> {
        let mut steps = Vec::new();
        let re =
            regex::Regex::new(r"^(?:STEP\s*)?(\d+)\s*[.:\)-]\s*(.+)$").expect("regex");
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(caps) = re.captures(line) {
                let desc = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");
                if !desc.is_empty() {
                    steps.push(PlanStep::new(steps.len() + 1, desc.to_string()));
                }
            }
        }
        if steps.is_empty() {
            for (idx, line) in content.lines().enumerate() {
                let clean = line
                    .trim()
                    .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '-')
                    .trim();
                if !clean.is_empty() {
                    steps.push(PlanStep::new(idx + 1, clean.to_string()));
                }
            }
        }
        steps
    }

    pub fn execute_plan<F>(
        &self,
        plan: &mut Plan,
        mut execute_step: F,
        from_step: usize,
        mut state_path: Option<PathBuf>,
    ) -> Result<()>
    where
        F: FnMut(&PlanStep) -> Result<(String, u64)>,
    {
        let mut state = PlanState::from_plan(plan.clone());
        state.save(state_path.clone())?;
        for idx in 0..plan.steps.len() {
            let mut failure: Option<anyhow::Error> = None;
            {
                let step = &mut plan.steps[idx];
                if step.number < from_step {
                    if step.status == StepStatus::Pending {
                        step.status = StepStatus::Skipped;
                    }
                    continue;
                }
                if matches!(step.status, StepStatus::Completed | StepStatus::Skipped) {
                    continue;
                }
                step.status = StepStatus::Running;
                step.started_at = Some(Utc::now());
                state.current_step = step.number;
                state.save(state_path.clone())?;
                match execute_step(step) {
                    Ok((result, tokens)) => {
                        step.status = StepStatus::Completed;
                        step.result = Some(result);
                        step.tokens_used = tokens;
                    }
                    Err(err) => {
                        step.status = StepStatus::Failed;
                        step.error = Some(err.to_string());
                        step.completed_at = Some(Utc::now());
                        failure = Some(err);
                    }
                }
                step.completed_at = Some(Utc::now());
                if failure.is_none() {
                    state.update_step(step);
                }
            }
            state.plan = plan.clone();
            state.save(state_path.clone())?;
            if let Some(err) = failure {
                return Err(err);
            }
        }
        if plan.is_complete() {
            plan.completed_at = Some(Utc::now());
            plan.total_tokens = state.total_tokens;
            PlanState::cleanup(state_path.take())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_steps_extracts_numbered_lines() {
        let executor = PlanExecutor::new();
        let content = "STEP 1: Do thing\nSTEP 2: Do other";
        let steps = executor.parse_steps(content);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].description, "Do thing");
    }

    #[test]
    fn execute_plan_tracks_state() {
        let mut plan = Plan::new(
            "Test".into(),
            vec![PlanStep::new(1, "Step one".into())],
        );
        let executor = PlanExecutor::new();
        executor
            .execute_plan(
                &mut plan,
                |_step| Ok(("done".into(), 10)),
                1,
                Some(PathBuf::from("test-state.json")),
            )
            .expect("execute");
        assert!(plan.is_complete());
        let _ = fs::remove_file("test-state.json");
    }
}
