//! S14 routed expert pages on top of the existing Vulkan `Loading -> Ready`
//! publication discipline.
//!
//! This is deliberately not a native DeepSeek executor. It only adapts an
//! already-produced official route to the existing fenced VRAM page pool.

use crate::{ExpertKey, VramPool};
use anyhow::{anyhow, Result};
use polaris_s14_runner::{GraphProfile, RouteDecision, EXPERT_PAGE_BYTES};

const S14_ROUTED_EXPERT_KIND: u8 = 0x53;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageState {
    CachedReady { slot: u32 },
    Loading { slot: u32, key: ExpertKey },
}

/// A route-bound upload ticket. It can only be created after a validated
/// `RouteDecision`; an ABI sample or guessed E0 can never create this ticket.
#[derive(Debug)]
pub struct VulkanRouteLoad {
    pub layer: u8,
    pub expert_ids: Vec<u16>,
    pages: Vec<PageState>,
}

pub struct S14VulkanExpertPages<'a> {
    pool: &'a mut VramPool,
    profile: GraphProfile,
}

impl<'a> S14VulkanExpertPages<'a> {
    pub fn new(pool: &'a mut VramPool, profile: GraphProfile) -> Result<Self> {
        if pool.slot_bytes() < EXPERT_PAGE_BYTES {
            return Err(anyhow!(
                "S14 expert slot is {} B, requires at least {} B",
                pool.slot_bytes(),
                EXPERT_PAGE_BYTES
            ));
        }
        Ok(Self { pool, profile })
    }

    /// Reserve misses as hidden Loading pages. Cached Ready pages remain
    /// visible; newly reserved pages remain invisible until `publish_after_fence`.
    pub fn begin_after_official_route(&mut self, route: &RouteDecision) -> Result<VulkanRouteLoad> {
        route
            .validate_for(self.profile)
            .map_err(|error| anyhow!(error.to_string()))?;
        let mut pages = Vec::with_capacity(route.expert_ids.len());
        for &expert in &route.expert_ids {
            let key = ExpertKey {
                layer: route.layer as u32,
                kind: S14_ROUTED_EXPERT_KIND,
                slot: expert as u32,
            };
            if let Some(slot) = self.pool.lookup(key) {
                pages.push(PageState::CachedReady { slot });
                continue;
            }
            match self.pool.reserve_loading(key) {
                Ok((slot, _evicted)) => pages.push(PageState::Loading { slot, key }),
                Err(error) => {
                    for page in &pages {
                        if let PageState::Loading { slot, key } = *page {
                            self.pool.cancel_loading(slot, key);
                        }
                    }
                    return Err(error);
                }
            }
        }
        Ok(VulkanRouteLoad {
            layer: route.layer,
            expert_ids: route.expert_ids.clone(),
            pages,
        })
    }

    /// Call only after the transfer fence covering every Loading page has
    /// completed. This is the sole publication point.
    pub fn publish_after_fence(&mut self, ticket: VulkanRouteLoad) -> Result<Vec<u32>> {
        let mut slots = Vec::with_capacity(ticket.pages.len());
        for page in ticket.pages {
            match page {
                PageState::CachedReady { slot } => slots.push(slot),
                PageState::Loading { slot, key } => {
                    self.pool.mark_ready(slot, key)?;
                    slots.push(slot);
                }
            }
        }
        Ok(slots)
    }

    pub fn cancel(&mut self, ticket: VulkanRouteLoad) {
        for page in ticket.pages {
            if let PageState::Loading { slot, key } = page {
                self.pool.cancel_loading(slot, key);
            }
        }
    }
}
