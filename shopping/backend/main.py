"""
Shopping AI Agent - FastAPI Backend Server
메인 진입점
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from agents import SupervisorAgent


class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    success: bool
    message: str
    products: list
    intent: dict = None


# Supervisor Agent 인스턴스
supervisor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    global supervisor
    supervisor = SupervisorAgent()
    yield
    # 리소스 정리 (Playwright 브라우저 등)
    if supervisor:
        await supervisor.close()
    supervisor = None


app = FastAPI(
    title="Shopping AI Agent",
    description="AI 쇼핑 어시스턴트 - 제품 검색 및 추천",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """헬스 체크"""
    return {"status": "ok", "message": "Shopping AI Agent is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 엔드포인트
    
    사용자 메시지를 받아 제품 검색 및 추천 결과를 반환합니다.
    """
    if not supervisor:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        result = await supervisor.process_request(request.message)
        return ChatResponse(
            success=result["success"],
            message=result["message"],
            products=result["products"],
            intent=result.get("intent")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """상태 체크"""
    return {
        "status": "healthy",
        "naver_api": bool(os.getenv("NAVER_CLIENT_ID")),
        "openai_api": bool(os.getenv("OPENAI_API_KEY")),
        "google_api": bool(os.getenv("GOOGLE_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
