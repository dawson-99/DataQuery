"""
智能数据查询系统 - 主应用入口（并发优化版）
提供省间电力数据查询和交易结果信息查询服务
"""

import asyncio
import contextvars
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config import settings
from src.api.routers.query_router import router as query_router, start_cache_cleanup, stop_cache_cleanup
from src.rule_review.router import router as rule_review_router
from src.service.agent_factory import agent_factory
from src.utils.logging_setup import logger, set_trace_id



# ---------- 日志中间件异步优化 ----------
# 使用队列异步写入日志，避免阻塞请求线程
log_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)


async def log_writer():
    """后台日志写入协程"""
    while True:
        try:
            record = await log_queue.get()
            if record is None:  # 停止信号
                break
            level, msg, ctx = record
            ctx.run(logger.log, level, msg)
        except Exception:
            # 避免后台任务崩溃
            pass


async def async_log(level: int, msg: str):
    """非阻塞日志写入，保留请求上下文以确保 trace_id 正确"""
    try:
        log_queue.put_nowait((level, msg, contextvars.copy_context()))
    except asyncio.QueueFull:
        # 队列满时降级为同步日志（极少发生）
        logger.log(level, f"[QUEUE_FULL] {msg}")


# ---------- 应用生命周期管理 ----------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    替代 on_event 的现代化生命周期管理
    """
    # 启动阶段
    writer_task = asyncio.create_task(log_writer())
    start_cache_cleanup()
    logger.info("应用启动完成，后台任务已启动")

    # 存储后台任务以便关闭时取消
    app.state.log_writer_task = writer_task

    yield

    # 关闭阶段
    logger.info("正在关闭应用...")

    # 显式关闭aiohttp.session
    await agent_factory.shutdown()

    # 停止缓存清理任务
    await stop_cache_cleanup()

    # 停止日志写入器
    await log_queue.put(None)  # 发送停止信号
    try:
        await asyncio.wait_for(writer_task, timeout=5.0)
    except asyncio.TimeoutError:
        writer_task.cancel()

    logger.info("应用已关闭")


def create_app() -> FastAPI:
    """创建并配置FastAPI应用"""

    app = FastAPI(
        title="智能数据查询系统",
        description="""
        ## 功能概述
        
        智能数据查询系统提供以下服务：
        
        - 基于自然语言的数据查询
        - 自动意图识别
        - 智能参数提取和URL拼接
        - 支持流式和非流式输出
        
        
        ## 特性
        
        - 🤖 **智能意图识别**：自动识别用户意图
        - 📊 **参数自动提取**：从自然语言中提取结构化参数
        - 🔄 **流式输出**：支持SSE流式响应
        - 💬 **多轮对话**：支持会话上下文管理
        - 📈 **结果格式化**：自动生成Markdown格式结果
        
        """,
        version="1.0.0",
        contact={
            "name": "智能数据查询系统",
            "email": "support@example.com",
        },
        license_info={
            "name": "MIT License",
        },
        lifespan=lifespan,  # 使用现代生命周期管理
    )

    # CORS中间件配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 高性能请求日志中间件（异步日志）
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """异步记录所有HTTP请求，不影响主请求性能"""
        trace_id = str(uuid4())
        set_trace_id(trace_id)

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            process_time = time.perf_counter() - start_time

            await async_log(
                logging.INFO,
                f"[请求完成] {request.method} {request.url.path} "
                f"状态={response.status_code} 耗时={process_time:.3f}s"
            )

            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            return response

        except Exception as e:
            process_time = time.perf_counter() - start_time
            await async_log(
                logging.ERROR,
                f"[请求错误] {request.method} {request.url.path} 耗时={process_time:.3f}s 错误={str(e)}"
            )
            raise

    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理器"""
        await async_log(
            logging.ERROR,
            f"[全局异常] {request.method} {request.url.path} 异常={str(exc)}"
        )
        # 同步记录详细堆栈到原始logger
        logger.error(f"全局异常详情", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "服务器内部错误",
                "detail": str(exc) if settings.DEBUG else "请联系管理员"
            }
        )

    # 注册路由
    app.include_router(query_router)
    app.include_router(rule_review_router)

    # 根路径
    @app.get("/", tags=["系统"])
    async def root():
        """系统根路径，返回API信息"""
        return {
            "name": "智能数据查询系统",
            "version": "1.0.0",
            "status": "running",
            "services": {
                "query": {
                    "path": "/v1/query",
                    "description": "智能查询服务（自动识别意图并路由）",
                    "endpoints": [
                        "POST /v1/query"
                    ]
                },
                "rule_review": {
                    "path": "/v1/rule-review",
                    "description": "电力规则审查服务（RAG + LLM + Judge）",
                    "endpoints": [
                        "GET /v1/rule-review"
                    ]
                }
            },
            "docs": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }

    return app


# 创建应用实例
app = create_app()


# ---------- 生产级 Uvicorn 配置 ----------
def run():
    """生产环境启动入口（可通过命令行调用）"""
    uvicorn.run(
        "app:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        workers=settings.UVICORN_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,                                  # 关闭 uvicorn 自带访问日志，使用自定义中间件
        proxy_headers=True,                                # 信任代理头（X-Forwarded-*）
        forwarded_allow_ips="*",                           # 生产环境如有固定代理 IP 应替换为具体地址
        server_header=False,                               # 隐藏 uvicorn 版本号
        limit_concurrency=settings.MAX_CONCURRENT_REQUESTS,
        backlog=settings.BACKLOG,
        timeout_keep_alive=settings.TIMEOUT_KEEP_ALIVE,
        limit_max_requests=settings.UVICORN_LIMIT_MAX_REQUESTS,
        timeout_graceful_shutdown=settings.UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT,
    )


if __name__ == "__main__":
    # 开发环境直接运行（单进程）
    if settings.DEBUG:
        uvicorn.run(
            app,
            host=settings.SERVER_HOST,
            port=settings.SERVER_PORT,
            log_level="debug",
            access_log=True,
            reload=True,  # 开发模式热重载
        )
    else:
        run()
