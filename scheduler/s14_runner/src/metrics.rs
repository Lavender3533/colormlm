use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferObservation {
    pub disk_bytes: u64,
    pub host_to_device_bytes: u64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub miss_stall_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterReport {
    pub format: String,
    pub committed_tokens: u64,
    pub first_token_latency_ms: Option<f64>,
    pub steady_state_sample_tokens: u64,
    pub steady_state_elapsed_ms: Option<f64>,
    pub steady_state_tokens_per_second: Option<f64>,
    pub disk_bytes: u64,
    pub host_to_device_bytes: u64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub miss_stall_ms: f64,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeCounters {
    started: Instant,
    first_commit: Option<Instant>,
    last_commit: Option<Instant>,
    committed_tokens: u64,
    transfer: TransferObservation,
}

impl RuntimeCounters {
    pub fn start_now() -> Self {
        Self::with_start(Instant::now())
    }

    pub fn with_start(started: Instant) -> Self {
        Self {
            started,
            first_commit: None,
            last_commit: None,
            committed_tokens: 0,
            transfer: TransferObservation::default(),
        }
    }

    pub fn observe_transfer(&mut self, observation: TransferObservation) {
        self.transfer.disk_bytes = self
            .transfer
            .disk_bytes
            .saturating_add(observation.disk_bytes);
        self.transfer.host_to_device_bytes = self
            .transfer
            .host_to_device_bytes
            .saturating_add(observation.host_to_device_bytes);
        self.transfer.expert_cache_hits = self
            .transfer
            .expert_cache_hits
            .saturating_add(observation.expert_cache_hits);
        self.transfer.expert_cache_misses = self
            .transfer
            .expert_cache_misses
            .saturating_add(observation.expert_cache_misses);
        self.transfer.miss_stall_ns = self
            .transfer
            .miss_stall_ns
            .saturating_add(observation.miss_stall_ns);
    }

    pub fn commit_now(&mut self) {
        self.commit_at(Instant::now());
    }

    pub fn commit_at(&mut self, instant: Instant) {
        if self.first_commit.is_none() {
            self.first_commit = Some(instant);
        }
        self.last_commit = Some(instant);
        self.committed_tokens = self.committed_tokens.saturating_add(1);
    }

    pub fn report(&self) -> CounterReport {
        let first_latency = self
            .first_commit
            .map(|instant| instant.duration_since(self.started));
        let steady = match (self.first_commit, self.last_commit, self.committed_tokens) {
            (Some(first), Some(last), tokens) if tokens >= 2 && last > first => {
                Some((tokens - 1, last.duration_since(first)))
            }
            _ => None,
        };
        CounterReport {
            format: "polaris-s14-observed-counters-v1".into(),
            committed_tokens: self.committed_tokens,
            first_token_latency_ms: first_latency.map(duration_ms),
            steady_state_sample_tokens: steady.map_or(0, |(tokens, _)| tokens),
            steady_state_elapsed_ms: steady.map(|(_, elapsed)| duration_ms(elapsed)),
            steady_state_tokens_per_second: steady
                .map(|(tokens, elapsed)| tokens as f64 / elapsed.as_secs_f64()),
            disk_bytes: self.transfer.disk_bytes,
            host_to_device_bytes: self.transfer.host_to_device_bytes,
            expert_cache_hits: self.transfer.expert_cache_hits,
            expert_cache_misses: self.transfer.expert_cache_misses,
            miss_stall_ms: self.transfer.miss_stall_ns as f64 / 1_000_000.0,
            source: if self.committed_tokens == 0 {
                "unmeasured_no_committed_token".into()
            } else {
                "observed_wall_clock_and_provider_counters".into()
            },
        }
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_never_invent_speed_before_observation() {
        let start = Instant::now();
        let mut counters = RuntimeCounters::with_start(start);
        let empty = counters.report();
        assert_eq!(empty.first_token_latency_ms, None);
        assert_eq!(empty.steady_state_tokens_per_second, None);
        assert_eq!(empty.source, "unmeasured_no_committed_token");

        counters.commit_at(start + Duration::from_millis(250));
        let first = counters.report();
        assert_eq!(first.first_token_latency_ms, Some(250.0));
        assert_eq!(first.steady_state_tokens_per_second, None);

        counters.commit_at(start + Duration::from_millis(300));
        let steady = counters.report();
        assert_eq!(steady.steady_state_sample_tokens, 1);
        assert_eq!(steady.steady_state_tokens_per_second, Some(20.0));
    }
}
