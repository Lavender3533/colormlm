//! VRAM expert slot pool with LRU eviction.
//!
//! Holds a fixed number of fixed-size slots in VRAM. Each slot can hold ONE
//! expert. Maps `(layer, kind, expert_slot) → vram_slot_idx` and tracks LRU
//! to evict cold slots when a new expert needs to land but pool is full.
//!
//! NOTE: this is a simple direct-mapped pool. The `expert_cache` crate has a
//! more sophisticated tier model; we'll integrate it when we add RAM tier.

use crate::buffer::GpuBuffer;
use crate::device::VulkanContext;
use anyhow::{anyhow, Result};
use ash::vk;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExpertKey {
    pub layer: u32,
    pub kind: u8, // ExpertKind as u8 to keep this struct cheap
    pub slot: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotState {
    Empty,
    Loading(ExpertKey),
    Ready(ExpertKey),
}

/// Generation-stamped slot metadata suitable for descriptor validation.
/// It remains valid only while its owning [`SlotLease`] is pinned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotBinding {
    pub key: ExpertKey,
    pub slot: u32,
    pub generation: u64,
}

/// A compute-side page pin. The scheduler must consume it through
/// `release_after_compute_fence` after the covering Vulkan fence completes.
#[derive(Debug)]
pub struct SlotLease {
    binding: SlotBinding,
}

impl SlotLease {
    pub fn binding(&self) -> SlotBinding {
        self.binding
    }
}

/// Hidden upload reservation. A late transfer callback cannot publish it
/// after cancellation/reuse because both slot and generation are checked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LoadingReservation {
    pub key: ExpertKey,
    pub slot: u32,
    pub generation: u64,
    pub evicted: Option<ExpertKey>,
}

pub struct VramPool {
    pub buffer: GpuBuffer,
    slot_bytes: u64,
    n_slots: u32,
    /// slot_idx → transfer lifecycle. Only `Ready` slots are visible to lookup.
    states: Vec<SlotState>,
    last_access: Vec<u64>,
    pin_count: Vec<u32>,
    generations: Vec<u64>,
    /// expert_key → slot_idx
    index: HashMap<ExpertKey, u32>,
    clock: u64,
    n_loaded: u32,
}

impl VramPool {
    pub fn new(ctx: &VulkanContext, n_slots: u32, slot_bytes: u64) -> Result<Self> {
        if n_slots == 0 {
            return Err(anyhow!("VRAM pool must contain at least one slot"));
        }
        if slot_bytes == 0 {
            return Err(anyhow!("VRAM slot size must be greater than zero"));
        }
        let total = (n_slots as u64)
            .checked_mul(slot_bytes)
            .ok_or_else(|| anyhow!("VRAM pool size overflow"))?;
        let qfs = if ctx.has_dedicated_transfer() {
            vec![ctx.qf_graphics, ctx.qf_transfer]
        } else {
            vec![]
        };
        let usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
        let buffer = if qfs.len() >= 2 {
            GpuBuffer::new_vram_shared(ctx, total, usage, &qfs)?
        } else {
            GpuBuffer::new_vram(ctx, total, usage)?
        };
        Ok(Self {
            buffer,
            slot_bytes,
            n_slots,
            states: vec![SlotState::Empty; n_slots as usize],
            last_access: vec![0; n_slots as usize],
            pin_count: vec![0; n_slots as usize],
            generations: vec![0; n_slots as usize],
            index: HashMap::with_capacity(n_slots as usize),
            clock: 0,
            n_loaded: 0,
        })
    }

    pub fn capacity(&self) -> u32 {
        self.n_slots
    }
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }
    pub fn n_loaded(&self) -> u32 {
        self.n_loaded
    }
    pub fn total_bytes(&self) -> u64 {
        self.n_slots as u64 * self.slot_bytes
    }

    /// Look up where an expert lives. Updates LRU clock if found.
    pub fn lookup(&mut self, key: ExpertKey) -> Option<u32> {
        if let Some(&idx) = self.index.get(&key) {
            self.clock = self.clock.saturating_add(1);
            self.last_access[idx as usize] = self.clock;
            Some(idx)
        } else {
            None
        }
    }

    /// Look up and pin a Ready page for a future compute submission.
    pub fn lookup_and_pin(&mut self, key: ExpertKey) -> Result<Option<SlotLease>> {
        let Some(&slot) = self.index.get(&key) else {
            return Ok(None);
        };
        if self.states[slot as usize] != SlotState::Ready(key) {
            return Err(anyhow!(
                "VRAM index for {:?} points at non-ready slot {}",
                key,
                slot
            ));
        }
        self.pin_count[slot as usize] = self.pin_count[slot as usize]
            .checked_add(1)
            .ok_or_else(|| anyhow!("VRAM slot {} pin count overflow", slot))?;
        self.clock = self.clock.saturating_add(1);
        self.last_access[slot as usize] = self.clock;
        Ok(Some(SlotLease {
            binding: SlotBinding {
                key,
                slot,
                generation: self.generations[slot as usize],
            },
        }))
    }

    /// Reserve a slot for an upload without publishing it to readers.
    /// Loading slots are never eviction candidates. The caller must invoke
    /// `mark_ready` only after the transfer fence has completed.
    pub fn reserve_loading(&mut self, key: ExpertKey) -> Result<(u32, Option<ExpertKey>)> {
        let reservation = self.reserve_loading_generation(key)?;
        Ok((reservation.slot, reservation.evicted))
    }

    /// Generation-aware reservation used by S14 transfer batches.
    pub fn reserve_loading_generation(&mut self, key: ExpertKey) -> Result<LoadingReservation> {
        if self.index.contains_key(&key) {
            return Err(anyhow!("expert {:?} is already ready", key));
        }
        if self
            .states
            .iter()
            .any(|state| *state == SlotState::Loading(key))
        {
            return Err(anyhow!("expert {:?} is already loading", key));
        }

        let slot_idx = if let Some(idx) = self
            .states
            .iter()
            .position(|state| *state == SlotState::Empty)
        {
            idx as u32
        } else {
            self.states
                .iter()
                .enumerate()
                .filter_map(|(idx, state)| match state {
                    SlotState::Ready(_) if self.pin_count[idx] == 0 => {
                        Some((idx, self.last_access[idx]))
                    }
                    SlotState::Empty | SlotState::Loading(_) => None,
                    SlotState::Ready(_) => None,
                })
                .min_by_key(|&(_, access)| access)
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| {
                    anyhow!(
                        "all {} VRAM slots are Loading or pinned Ready",
                        self.n_slots
                    )
                })?
        };

        let evicted = match self.states[slot_idx as usize] {
            SlotState::Ready(old_key) => {
                self.index.remove(&old_key);
                Some(old_key)
            }
            SlotState::Empty => {
                self.n_loaded += 1;
                None
            }
            SlotState::Loading(_) => unreachable!("loading slots are not reservation candidates"),
        };

        let generation = self.generations[slot_idx as usize]
            .checked_add(1)
            .ok_or_else(|| anyhow!("VRAM slot {} generation overflow", slot_idx))?;
        self.generations[slot_idx as usize] = generation;
        self.pin_count[slot_idx as usize] = 0;
        self.states[slot_idx as usize] = SlotState::Loading(key);
        Ok(LoadingReservation {
            key,
            slot: slot_idx,
            generation,
            evicted,
        })
    }

    /// Publish a completed upload. Before this call `lookup` must miss.
    pub fn mark_ready(&mut self, slot_idx: u32, key: ExpertKey) -> Result<()> {
        let state = self
            .states
            .get(slot_idx as usize)
            .ok_or_else(|| anyhow!("invalid VRAM slot {}", slot_idx))?;
        if *state != SlotState::Loading(key) {
            return Err(anyhow!(
                "slot {} cannot publish {:?}: current state is {:?}",
                slot_idx,
                key,
                *state
            ));
        }
        let reservation = LoadingReservation {
            key,
            slot: slot_idx,
            generation: self.generations[slot_idx as usize],
            evicted: None,
        };
        let lease = self.publish_batch_and_pin(&[reservation])?.remove(0);
        self.release_after_compute_fence(lease)?;
        Ok(())
    }

    /// Preflight an entire transfer batch without publishing any page.
    pub fn validate_publish_batch(&self, reservations: &[LoadingReservation]) -> Result<()> {
        if reservations.is_empty() {
            return Err(anyhow!("cannot publish an empty VRAM batch"));
        }
        let mut slots = HashSet::with_capacity(reservations.len());
        let mut keys = HashSet::with_capacity(reservations.len());
        for reservation in reservations {
            if !slots.insert(reservation.slot) || !keys.insert(reservation.key) {
                return Err(anyhow!(
                    "VRAM publish batch contains a duplicate slot or key"
                ));
            }
            let state = self
                .states
                .get(reservation.slot as usize)
                .ok_or_else(|| anyhow!("invalid VRAM slot {}", reservation.slot))?;
            let current_generation = self.generations[reservation.slot as usize];
            if *state != SlotState::Loading(reservation.key)
                || current_generation != reservation.generation
            {
                return Err(anyhow!(
                    "stale VRAM generation for {:?}: slot={} generation={} current_state={:?} current_generation={}",
                    reservation.key,
                    reservation.slot,
                    reservation.generation,
                    state,
                    current_generation
                ));
            }
            if self.pin_count[reservation.slot as usize] != 0 {
                return Err(anyhow!(
                    "Loading slot {} unexpectedly has compute pins",
                    reservation.slot
                ));
            }
            if self.index.contains_key(&reservation.key) {
                return Err(anyhow!(
                    "cannot publish {:?}: key is already visible",
                    reservation.key
                ));
            }
        }
        Ok(())
    }

    /// Atomically publish a preflighted upload batch and pin every resulting
    /// Ready page for the compute fence that will consume it.
    pub fn publish_batch_and_pin(
        &mut self,
        reservations: &[LoadingReservation],
    ) -> Result<Vec<SlotLease>> {
        self.validate_publish_batch(reservations)?;
        Ok(self.publish_prevalidated_batch_and_pin(reservations))
    }

    pub(crate) fn publish_prevalidated_batch_and_pin(
        &mut self,
        reservations: &[LoadingReservation],
    ) -> Vec<SlotLease> {
        debug_assert!(self.validate_publish_batch(reservations).is_ok());
        let mut leases = Vec::with_capacity(reservations.len());
        for reservation in reservations {
            self.states[reservation.slot as usize] = SlotState::Ready(reservation.key);
            self.index.insert(reservation.key, reservation.slot);
            self.pin_count[reservation.slot as usize] = 1;
            self.clock = self.clock.saturating_add(1);
            self.last_access[reservation.slot as usize] = self.clock;
            leases.push(SlotLease {
                binding: SlotBinding {
                    key: reservation.key,
                    slot: reservation.slot,
                    generation: reservation.generation,
                },
            });
        }
        leases
    }

    /// Reject descriptor metadata after an eviction/reuse generation change,
    /// or if the compute pin has already been released.
    pub fn validate_binding(&self, binding: SlotBinding) -> Result<()> {
        let state = self
            .states
            .get(binding.slot as usize)
            .ok_or_else(|| anyhow!("invalid VRAM slot {}", binding.slot))?;
        if *state != SlotState::Ready(binding.key)
            || self.generations[binding.slot as usize] != binding.generation
            || self.pin_count[binding.slot as usize] == 0
        {
            return Err(anyhow!(
                "stale or unpinned VRAM binding for {:?}: slot={} generation={}",
                binding.key,
                binding.slot,
                binding.generation
            ));
        }
        Ok(())
    }

    pub fn release_after_compute_fence(&mut self, lease: SlotLease) -> Result<()> {
        self.validate_binding(lease.binding)?;
        self.pin_count[lease.binding.slot as usize] -= 1;
        Ok(())
    }

    /// Roll back a reservation when the disk read or Vulkan submission fails.
    pub fn cancel_loading(&mut self, slot_idx: u32, key: ExpertKey) {
        if self.states.get(slot_idx as usize) == Some(&SlotState::Loading(key)) {
            self.states[slot_idx as usize] = SlotState::Empty;
            self.pin_count[slot_idx as usize] = 0;
            self.n_loaded = self.n_loaded.saturating_sub(1);
        }
    }

    pub fn cancel_reservation(&mut self, reservation: LoadingReservation) -> Result<()> {
        let state = self
            .states
            .get(reservation.slot as usize)
            .ok_or_else(|| anyhow!("invalid VRAM slot {}", reservation.slot))?;
        if *state != SlotState::Loading(reservation.key)
            || self.generations[reservation.slot as usize] != reservation.generation
        {
            return Err(anyhow!(
                "stale VRAM cancellation for {:?}: slot={} generation={}",
                reservation.key,
                reservation.slot,
                reservation.generation
            ));
        }
        self.states[reservation.slot as usize] = SlotState::Empty;
        self.pin_count[reservation.slot as usize] = 0;
        self.n_loaded = self.n_loaded.saturating_sub(1);
        Ok(())
    }

    /// Byte offset of slot in the underlying VRAM buffer.
    pub fn slot_offset(&self, slot_idx: u32) -> u64 {
        slot_idx as u64 * self.slot_bytes
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        self.buffer.destroy(ctx);
    }

    #[cfg(test)]
    pub(crate) fn new_for_tests(n_slots: u32, slot_bytes: u64) -> Self {
        Self {
            buffer: unsafe { std::mem::zeroed() },
            slot_bytes,
            n_slots,
            states: vec![SlotState::Empty; n_slots as usize],
            last_access: vec![0; n_slots as usize],
            pin_count: vec![0; n_slots as usize],
            generations: vec![0; n_slots as usize],
            index: HashMap::new(),
            clock: 0,
            n_loaded: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    // Pure logic tests (no Vulkan). For Vulkan integration coverage see examples.
    use super::*;

    fn key(l: u32, k: u8, s: u32) -> ExpertKey {
        ExpertKey {
            layer: l,
            kind: k,
            slot: s,
        }
    }

    #[test]
    fn reserve_eviction_logic() {
        // Build a pool by hand without Vulkan to test pure book-keeping
        let mut p = VramPool {
            buffer: unsafe { std::mem::zeroed() }, // never used in this test
            slot_bytes: 1024,
            n_slots: 3,
            states: vec![SlotState::Empty; 3],
            last_access: vec![0; 3],
            pin_count: vec![0; 3],
            generations: vec![0; 3],
            index: HashMap::new(),
            clock: 0,
            n_loaded: 0,
        };
        let k1 = key(0, 0, 1);
        let k2 = key(0, 0, 2);
        let k3 = key(0, 0, 3);
        let k4 = key(0, 0, 4);

        let (s1, e1) = p.reserve_loading(k1).unwrap();
        assert_eq!(s1, 0);
        assert!(e1.is_none());
        assert!(p.lookup(k1).is_none());
        p.mark_ready(s1, k1).unwrap();
        let (s2, e2) = p.reserve_loading(k2).unwrap();
        assert_eq!(s2, 1);
        assert!(e2.is_none());
        p.mark_ready(s2, k2).unwrap();
        let (s3, e3) = p.reserve_loading(k3).unwrap();
        assert_eq!(s3, 2);
        assert!(e3.is_none());
        p.mark_ready(s3, k3).unwrap();
        assert_eq!(p.n_loaded(), 3);

        // Touch k2 to make k1 oldest
        p.lookup(k2);
        // k4 should evict k1
        let (s4, e4) = p.reserve_loading(k4).unwrap();
        assert_eq!(s4, 0);
        assert_eq!(e4, Some(k1));
        assert!(p.lookup(k4).is_none());
        p.mark_ready(s4, k4).unwrap();
        assert_eq!(p.n_loaded(), 3);

        // k1 evicted
        assert!(p.lookup(k1).is_none());
        // others still there
        assert!(p.lookup(k2).is_some());
        assert!(p.lookup(k3).is_some());
        assert!(p.lookup(k4).is_some());

        // Avoid running buffer.destroy in this test (it's zeroed)
        std::mem::forget(p);
    }

    #[test]
    fn loading_slots_are_hidden_and_never_evicted() {
        let mut p = VramPool {
            buffer: unsafe { std::mem::zeroed() }, // never used in this test
            slot_bytes: 1024,
            n_slots: 1,
            states: vec![SlotState::Empty; 1],
            last_access: vec![0; 1],
            pin_count: vec![0; 1],
            generations: vec![0; 1],
            index: HashMap::new(),
            clock: 0,
            n_loaded: 0,
        };
        let loading = key(1, 0, 7);
        let other = key(1, 0, 8);

        let (slot, _) = p.reserve_loading(loading).unwrap();
        assert!(p.lookup(loading).is_none());
        assert!(p.reserve_loading(other).is_err());

        p.cancel_loading(slot, loading);
        let (replacement_slot, _) = p.reserve_loading(other).unwrap();
        assert_eq!(replacement_slot, slot);
        p.mark_ready(replacement_slot, other).unwrap();
        assert_eq!(p.lookup(other), Some(slot));

        std::mem::forget(p);
    }

    #[test]
    fn pinned_ready_slot_cannot_be_evicted_until_compute_fence_release() {
        let mut p = VramPool {
            buffer: unsafe { std::mem::zeroed() }, // never used in this test
            slot_bytes: 1024,
            n_slots: 1,
            states: vec![SlotState::Empty; 1],
            last_access: vec![0; 1],
            pin_count: vec![0; 1],
            generations: vec![0; 1],
            index: HashMap::new(),
            clock: 0,
            n_loaded: 0,
        };
        let resident = key(42, 0x53, 126);
        let replacement = key(42, 0x53, 12);

        let reservation = p.reserve_loading_generation(resident).unwrap();
        let lease = p.publish_batch_and_pin(&[reservation]).unwrap().remove(0);
        assert!(p.reserve_loading_generation(replacement).is_err());
        p.release_after_compute_fence(lease).unwrap();
        assert!(p.reserve_loading_generation(replacement).is_ok());

        std::mem::forget(p);
    }

    #[test]
    fn stale_generation_is_rejected_after_slot_reuse() {
        let mut p = VramPool {
            buffer: unsafe { std::mem::zeroed() }, // never used in this test
            slot_bytes: 1024,
            n_slots: 1,
            states: vec![SlotState::Empty; 1],
            last_access: vec![0; 1],
            pin_count: vec![0; 1],
            generations: vec![0; 1],
            index: HashMap::new(),
            clock: 0,
            n_loaded: 0,
        };
        let first = key(42, 0x53, 126);
        let second = key(42, 0x53, 12);

        let first_reservation = p.reserve_loading_generation(first).unwrap();
        let first_lease = p
            .publish_batch_and_pin(&[first_reservation])
            .unwrap()
            .remove(0);
        let stale_binding = first_lease.binding();
        p.release_after_compute_fence(first_lease).unwrap();
        let second_reservation = p.reserve_loading_generation(second).unwrap();
        p.publish_batch_and_pin(&[second_reservation]).unwrap();

        assert!(p.validate_binding(stale_binding).is_err());
        assert!(p
            .release_after_compute_fence(SlotLease {
                binding: stale_binding,
            })
            .is_err());

        std::mem::forget(p);
    }

    #[test]
    fn batch_publish_preflight_prevents_partial_visibility() {
        let mut p = VramPool {
            buffer: unsafe { std::mem::zeroed() }, // never used in this test
            slot_bytes: 1024,
            n_slots: 2,
            states: vec![SlotState::Empty; 2],
            last_access: vec![0; 2],
            pin_count: vec![0; 2],
            generations: vec![0; 2],
            index: HashMap::new(),
            clock: 0,
            n_loaded: 0,
        };
        let first = key(42, 0x53, 126);
        let second = key(42, 0x53, 12);
        let first_reservation = p.reserve_loading_generation(first).unwrap();
        let second_reservation = p.reserve_loading_generation(second).unwrap();
        let mut stale_second = second_reservation;
        stale_second.generation += 1;

        assert!(p
            .publish_batch_and_pin(&[first_reservation, stale_second])
            .is_err());
        assert!(p.lookup(first).is_none());
        assert!(p.lookup(second).is_none());
        p.cancel_reservation(first_reservation).unwrap();
        p.cancel_reservation(second_reservation).unwrap();

        std::mem::forget(p);
    }
}
