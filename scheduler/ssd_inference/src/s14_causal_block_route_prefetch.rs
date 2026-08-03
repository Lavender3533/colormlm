//! Causally safe cache-only prefetch for one authoritative K=4/8 route.
//!
//! The ticket may warm loose Range payloads after the real GPU router has
//! returned, but it owns no model state, mmap lease, union upload or route
//! decision.  `finish` binds it to the independently rebuilt authoritative
//! identity plan; the normal union materializer remains the only publisher.

use crate::{
    s14_dynamic_page_cache_readiness::{
        fetch_dynamic_page_plans_batched_only, DynamicPageFetchMode, DynamicPageFetchOnlyReceipt,
    },
    s14_dynamic_routed_page_plan::DynamicRoutedPagePlan,
};
use anyhow::{bail, Context, Result};
use std::{path::PathBuf, thread};

#[derive(Debug)]
pub struct S14CausalBlockRoutePrefetchTicket {
    layer: u8,
    base_position: u32,
    route_plans: Vec<DynamicRoutedPagePlan>,
    handle: Option<thread::JoinHandle<std::result::Result<DynamicPageFetchOnlyReceipt, String>>>,
    local_only_receipt: Option<DynamicPageFetchOnlyReceipt>,
}

impl S14CausalBlockRoutePrefetchTicket {
    pub fn start(
        base_position: u32,
        route_plans: Vec<DynamicRoutedPagePlan>,
        cache_root: PathBuf,
        fetch_mode: DynamicPageFetchMode,
    ) -> Result<Self> {
        if !matches!(route_plans.len(), 4 | 8) {
            bail!("causal-block route prefetch 要求精确K=4/8 plans");
        }
        let layer = route_plans[0].layer;
        for (lane, plan) in route_plans.iter().enumerate() {
            let expected_position = u64::from(base_position)
                .checked_add(lane as u64)
                .context("causal-block route prefetch position overflow")?;
            if plan.layer != layer || plan.position != expected_position {
                bail!("causal-block route prefetch layer/position identity 漂移");
            }
        }

        let (handle, local_only_receipt) = match fetch_mode {
            DynamicPageFetchMode::LocalOnly => (
                None,
                Some(DynamicPageFetchOnlyReceipt {
                    layer,
                    position: u64::from(base_position),
                    ..DynamicPageFetchOnlyReceipt::default()
                }),
            ),
            DynamicPageFetchMode::ExplicitFetch => {
                let worker_plans = route_plans.clone();
                let handle = thread::Builder::new()
                    .name(format!("s14-route-prefetch-l{layer}"))
                    .spawn(move || {
                        fetch_dynamic_page_plans_batched_only(
                            &worker_plans,
                            &cache_root,
                            DynamicPageFetchMode::ExplicitFetch,
                        )
                        .map_err(|error| error.to_string())
                    })
                    .context("启动causal-block exact-route prefetch ticket失败")?;
                (Some(handle), None)
            }
        };
        Ok(Self {
            layer,
            base_position,
            route_plans,
            handle,
            local_only_receipt,
        })
    }

    pub fn finish(
        mut self,
        authoritative_route_plans: &[DynamicRoutedPagePlan],
    ) -> Result<DynamicPageFetchOnlyReceipt> {
        if authoritative_route_plans != self.route_plans {
            bail!("causal-block route prefetch ticket 与 authoritative union identity 漂移");
        }
        let receipt = self.join_inner()?;
        if receipt.layer != self.layer || receipt.position != u64::from(self.base_position) {
            bail!("causal-block route prefetch receipt layer/position 漂移");
        }
        Ok(receipt)
    }

    /// Abort/destroy path: never detach a worker.  A fetch error is returned to
    /// the owner after the bounded worker timeout has reclaimed the child.
    pub fn drain(mut self) -> Result<()> {
        self.join_inner().map(|_| ())
    }

    fn join_inner(&mut self) -> Result<DynamicPageFetchOnlyReceipt> {
        if let Some(receipt) = self.local_only_receipt.take() {
            return Ok(receipt);
        }
        let handle = self
            .handle
            .take()
            .context("causal-block route prefetch ticket 已被消费")?;
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("causal-block route prefetch worker panic"))?
            .map_err(anyhow::Error::msg)
    }
}

impl Drop for S14CausalBlockRoutePrefetchTicket {
    fn drop(&mut self) {
        // Safety first: an error path must not detach a downloader that can
        // outlive the block and collide with the next authoritative request.
        let _ = self.join_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plans(base_position: u32) -> Vec<DynamicRoutedPagePlan> {
        (0..4)
            .map(|lane| DynamicRoutedPagePlan {
                layer: 7,
                position: u64::from(base_position) + lane,
                expert_ids: [0, 1, 2, 3, 4, 5],
                route_weights: [0.25; 6],
                pages: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn local_ticket_requires_exact_authoritative_route_identity() {
        let authoritative = plans(9);
        let ticket = S14CausalBlockRoutePrefetchTicket::start(
            9,
            authoritative.clone(),
            PathBuf::from("unused-local-cache"),
            DynamicPageFetchMode::LocalOnly,
        )
        .unwrap();
        assert_eq!(
            ticket.finish(&authoritative).unwrap().transport_invocations,
            0
        );

        let mut drift = authoritative.clone();
        drift[0].expert_ids.swap(0, 1);
        let ticket = S14CausalBlockRoutePrefetchTicket::start(
            9,
            authoritative,
            PathBuf::from("unused-local-cache"),
            DynamicPageFetchMode::LocalOnly,
        )
        .unwrap();
        assert!(ticket.finish(&drift).is_err());
    }
}
