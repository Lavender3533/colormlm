use crate::{
    ChatEngine, EngineChatRequest, EngineError, EngineErrorKind, EngineEventReceiver,
    EngineEventSender, EngineHealth, EngineRequestLease, EngineStartFuture,
};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, RwLock},
};
use tokio::sync::mpsc;

/// 常驻单 worker 实际拥有的同步后端。
///
/// 后端对象在专用 OS 线程内完成构造并始终留在该线程，因此 Vulkan context、paged arena
/// 和 codec 不需要跨线程移动。错误由 worker 作为失败事件发送，绝不会补发 `Done`。
pub trait ResidentChatBackend: 'static {
    fn run_chat(
        &mut self,
        request: EngineChatRequest,
        events: &EngineEventSender,
    ) -> Result<(), EngineError>;

    /// deadline/cancellation 只在同步 backend 的安全边界观察；默认入口只在开始前拒绝
    /// 已失效租约。S14 K4 production backend 会进一步在每个 block commit 后检查。
    fn run_chat_with_lease(
        &mut self,
        request: EngineChatRequest,
        events: &EngineEventSender,
        lease: &EngineRequestLease,
    ) -> Result<(), EngineError> {
        if lease.should_stop() || events.is_closed() {
            return Ok(());
        }
        self.run_chat(request, events)
    }
}

struct WorkerCommand {
    request: EngineChatRequest,
    events: EngineEventSender,
    lease: EngineRequestLease,
}

/// 一个有界队列、一个模型所有者线程、一次只执行一个请求的最薄常驻引擎。
///
/// `spawn` 返回时后端可能仍在加载；只有 loader 成功且资源确实由 worker 持有后才发布
/// `ready=true`。加载中、加载失败或 worker panic 都保持/恢复为未就绪。
pub struct ResidentChatEngine {
    commands: Option<mpsc::Sender<WorkerCommand>>,
    health: Arc<RwLock<EngineHealth>>,
}

impl ResidentChatEngine {
    pub fn blocked(reason: impl Into<String>) -> Self {
        Self {
            commands: None,
            health: Arc::new(RwLock::new(EngineHealth {
                ready: false,
                detail: reason.into(),
            })),
        }
    }

    pub fn spawn<F>(queue_capacity: usize, loader: F) -> Result<Self, EngineError>
    where
        F: FnOnce() -> Result<Box<dyn ResidentChatBackend>, EngineError> + Send + 'static,
    {
        if queue_capacity == 0 {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "resident worker queue_capacity 必须大于 0",
            ));
        }
        let (commands, receiver) = mpsc::channel(queue_capacity);
        let health = Arc::new(RwLock::new(EngineHealth {
            ready: false,
            detail: "Polaris S14 resident worker 正在加载".to_owned(),
        }));
        let worker_health = Arc::clone(&health);
        std::thread::Builder::new()
            .name("polaris-s14-resident".to_owned())
            .spawn(move || run_worker(receiver, worker_health, loader))
            .map_err(|error| {
                EngineError::runtime_unavailable(format!(
                    "启动 Polaris S14 resident worker 失败: {error}"
                ))
            })?;
        Ok(Self {
            commands: Some(commands),
            health,
        })
    }

    fn current_health(&self) -> EngineHealth {
        self.health
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl ChatEngine for ResidentChatEngine {
    fn start_chat(&self, request: EngineChatRequest) -> EngineStartFuture<'_> {
        self.start_chat_with_lease(request, EngineRequestLease::unbounded())
    }

    fn start_chat_with_lease(
        &self,
        request: EngineChatRequest,
        lease: EngineRequestLease,
    ) -> EngineStartFuture<'_> {
        let health = self.current_health();
        let commands = self.commands.clone();
        Box::pin(async move {
            if !health.ready {
                return Err(EngineError::runtime_unavailable(health.detail));
            }
            let Some(commands) = commands else {
                return Err(EngineError::runtime_unavailable(health.detail));
            };
            let (events, receiver): (_, EngineEventReceiver) = mpsc::channel(16);
            match commands.try_send(WorkerCommand {
                request,
                events,
                lease,
            }) {
                Ok(()) => Ok(receiver),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    let mut error = EngineError::new(
                        EngineErrorKind::QueueFull,
                        "Polaris S14 单 worker 正忙，等待队列已满",
                    );
                    error.retryable = true;
                    Err(error)
                }
                Err(mpsc::error::TrySendError::Closed(_)) => Err(EngineError::runtime_unavailable(
                    "Polaris S14 resident worker 已停止",
                )),
            }
        })
    }

    fn health(&self) -> EngineHealth {
        self.current_health()
    }
}

fn set_health(health: &RwLock<EngineHealth>, ready: bool, detail: impl Into<String>) {
    let mut state = health
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *state = EngineHealth {
        ready,
        detail: detail.into(),
    };
}

fn run_worker<F>(
    mut commands: mpsc::Receiver<WorkerCommand>,
    health: Arc<RwLock<EngineHealth>>,
    loader: F,
) where
    F: FnOnce() -> Result<Box<dyn ResidentChatBackend>, EngineError>,
{
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let mut backend = match loader() {
            Ok(backend) => backend,
            Err(error) => {
                set_health(
                    &health,
                    false,
                    format!("Polaris S14 resident backend 加载失败: {}", error.message),
                );
                return;
            }
        };
        set_health(
            &health,
            true,
            "Polaris S14 resident backend 已加载并通过调用方数值门",
        );
        while let Some(command) = commands.blocking_recv() {
            if let Err(error) =
                backend.run_chat_with_lease(command.request, &command.events, &command.lease)
            {
                let _ = command.events.blocking_send(Err(error));
            }
        }
        set_health(&health, false, "Polaris S14 resident worker 已停止");
    }));
    if outcome.is_err() {
        set_health(
            &health,
            false,
            "Polaris S14 resident worker panic；已撤销 ready 状态",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineDone, EngineEvent, FinishReason, POLARIS_MODEL_ID};
    use std::time::Duration;

    struct DoneBackend;

    impl ResidentChatBackend for DoneBackend {
        fn run_chat(
            &mut self,
            _request: EngineChatRequest,
            events: &EngineEventSender,
        ) -> Result<(), EngineError> {
            events
                .blocking_send(Ok(EngineEvent::Done(EngineDone {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: Some(1),
                    completion_tokens: Some(0),
                })))
                .map_err(|_| EngineError::runtime_unavailable("测试 receiver 已关闭"))
        }
    }

    #[tokio::test]
    async fn worker_only_publishes_ready_after_backend_load_and_returns_done() {
        let engine = ResidentChatEngine::spawn(1, || Ok(Box::new(DoneBackend))).unwrap();
        for _ in 0..1000 {
            if engine.health().ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(engine.health().ready);
        let request = EngineChatRequest {
            model: POLARIS_MODEL_ID.to_owned(),
            messages: Vec::new(),
            max_tokens: Some(1),
            temperature: Some(0.0),
            stop: Vec::new(),
            tools: None,
            tool_choice: None,
        };
        let mut receiver = engine.start_chat(request).await.unwrap();
        assert!(matches!(
            receiver.recv().await,
            Some(Ok(EngineEvent::Done(_)))
        ));
    }

    #[tokio::test]
    async fn blocked_engine_never_accepts_a_request() {
        let engine = ResidentChatEngine::blocked("测试数值门未通过");
        let request = EngineChatRequest {
            model: POLARIS_MODEL_ID.to_owned(),
            messages: Vec::new(),
            max_tokens: Some(1),
            temperature: None,
            stop: Vec::new(),
            tools: None,
            tool_choice: None,
        };
        let error = engine.start_chat(request).await.unwrap_err();
        assert_eq!(error.kind, EngineErrorKind::RuntimeUnavailable);
        assert!(!engine.health().ready);
    }
}
