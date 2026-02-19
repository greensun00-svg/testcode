"""
Supervisor Agent - 멀티 에이전트 시스템의 총괄 에이전트
OpenAI GPT-4o를 사용하여 사용자 의도를 파악하고 하위 에이전트를 조율합니다.
"""
import os
import sys
import json
from typing import Any, Dict, List, Optional

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from agents.search_agent import SearchAgent
from agents.price_agent import PriceAgent
from agents.rank_agent import RankAgent
from agents.detail_agent import DetailAgent

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SupervisorAgent:
    """
    멀티 에이전트 시스템의 총괄 에이전트
    
    OpenAI GPT-4o를 사용하여:
    1. 사용자 의도 파악
    2. 하위 에이전트 조율
    3. 상세 페이지 분석 (Detail Agent)
    4. 최종 응답 생성
    """
    
    def __init__(self, enable_detail_analysis: bool = True):
        self.model = "gpt-4o"
        self.search_agent = SearchAgent()
        self.price_agent = PriceAgent()
        self.rank_agent = RankAgent()
        self.detail_agent = DetailAgent() if enable_detail_analysis else None
        self.enable_detail_analysis = enable_detail_analysis
    
    async def analyze_intent(self, user_message: str) -> Dict[str, Any]:
        """
        사용자 메시지의 의도 분석
        
        Returns:
            - intent: 의도 유형 (search, compare, recommend)
            - keywords: 추출된 키워드
            - preferences: 사용자 선호사항 (가격대, 브랜드 등)
            - needs_detail: 상세 분석 필요 여부
        """
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """당신은 쇼핑 AI 어시스턴트입니다.
사용자의 메시지를 분석하여 의도와 요구사항을 파악해주세요.
응답은 JSON 형식으로:
{
    "intent": "search|compare|recommend",
    "keywords": ["키워드1", "키워드2"],
    "preferences": {
        "min_price": 0,
        "max_price": 0,
        "brand": "",
        "priority": "price|quality|balance"
    },
    "needs_detail": true/false,
    "detail_criteria": ["확인할 조건1", "확인할 조건2"],
    "summary": "사용자 요청 요약"
}

needs_detail은 다음 경우 true:
- 구체적인 스펙 요구사항 (용량, 크기, 재질 등)
- 특정 기능 요구 (방수, 무선충전 등)
- 상세 조건 확인 필요 시
"""
                },
                {"role": "user", "content": user_message}
            ]
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "intent": "search",
                "keywords": [user_message],
                "preferences": {"priority": "balance"},
                "needs_detail": False,
                "detail_criteria": [],
                "summary": user_message
            }
    
    async def process_request(
        self, 
        user_message: str,
        enable_detail: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        사용자 요청을 처리하고 제품 추천 결과 반환
        
        Args:
            user_message: 사용자의 자연어 메시지
            enable_detail: 상세 분석 활성화 여부 (None이면 자동 판단)
            
        Returns:
            처리 결과 (제품 목록, 응답 메시지 등)
        """
        # 1. 의도 분석
        intent_data = await self.analyze_intent(user_message)
        
        # 2. 제품 검색 (Search Agent)
        products = await self.search_agent.search(user_message, max_results=20)
        
        if not products:
            return {
                "success": False,
                "message": "죄송합니다. 검색 결과가 없습니다. 다른 키워드로 시도해주세요.",
                "products": [],
                "intent": intent_data
            }
        
        # 3. 가격 필터링 및 분석 (Price Agent)
        preferences = intent_data.get("preferences", {})
        min_price = preferences.get("min_price", 0)
        max_price = preferences.get("max_price", float('inf'))
        
        if max_price > 0:
            products = self.price_agent.filter_by_price_range(
                products, min_price, int(max_price)
            )
        
        # 가치 분석
        analyzed_products = await self.price_agent.analyze_value(
            products, user_message
        )
        
        # 4. 랭킹 (Rank Agent)
        ranked_products = await self.rank_agent.calculate_relevance(
            analyzed_products, user_message
        )
        
        # 가중치 설정
        priority = preferences.get("priority", "balance")
        if priority == "price":
            weights = {"relevance": 0.2, "value": 0.3, "price": 0.5}
        elif priority == "quality":
            weights = {"relevance": 0.5, "value": 0.3, "price": 0.2}
        else:
            weights = {"relevance": 0.4, "value": 0.3, "price": 0.3}
        
        final_products = self.rank_agent.rank_products(ranked_products, weights)
        
        # 5. 상세 분석 (Detail Agent) - 조건부 실행
        should_analyze = enable_detail if enable_detail is not None else intent_data.get("needs_detail", False)
        
        if should_analyze and self.detail_agent and self.enable_detail_analysis:
            try:
                # 상위 5개 제품만 상세 분석
                detail_criteria = " ".join(intent_data.get("detail_criteria", []))
                analysis_requirements = f"{user_message}\n확인 조건: {detail_criteria}"
                
                analyzed_with_detail = await self.detail_agent.analyze_multiple(
                    final_products[:10],
                    analysis_requirements,
                    max_products=5
                )
                
                # 상세 분석 결과 기반 재랭킹
                final_products = self._rerank_with_detail(analyzed_with_detail)
                intent_data["detail_analysis_performed"] = True
            except Exception as e:
                intent_data["detail_analysis_error"] = str(e)
        
        # 6. 최종 응답 생성
        top_products = final_products[:5]
        response_message = await self._generate_response(
            user_message, intent_data, top_products
        )
        
        return {
            "success": True,
            "message": response_message,
            "products": final_products[:10],
            "intent": intent_data
        }
    
    def _rerank_with_detail(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """상세 분석 결과를 반영하여 재랭킹"""
        for product in products:
            detail = product.get("detail_analysis")
            if detail and detail.get("match_score"):
                # 상세 분석 점수를 종합 점수에 반영 (30% 가중치)
                base_score = product.get("total_score", 5)
                detail_score = detail["match_score"] / 10  # 0-100 → 0-10
                product["total_score"] = round(
                    base_score * 0.7 + detail_score * 0.3, 2
                )
                product["is_detail_match"] = detail.get("is_match", False)
        
        # 재정렬: 상세 매치 우선, 그 다음 점수순
        return sorted(
            products,
            key=lambda p: (
                p.get("is_detail_match", False),
                p.get("total_score", 0)
            ),
            reverse=True
        )
    
    async def _generate_response(
        self,
        user_message: str,
        intent_data: Dict[str, Any],
        top_products: List[Dict[str, Any]]
    ) -> str:
        """최종 응답 메시지 생성"""
        if not top_products:
            return "검색 조건에 맞는 제품을 찾지 못했습니다."
        
        # 상세 분석 여부에 따른 정보 포함
        products_info = []
        for i, p in enumerate(top_products[:5]):
            info = f"{i+1}. {p['title']} - {p['lprice']:,}원"
            if p.get("detail_analysis") and p["detail_analysis"].get("summary"):
                info += f" (분석: {p['detail_analysis']['summary'][:50]}...)"
            products_info.append(info)
        
        products_text = "\n".join(products_info)
        detail_note = ""
        if intent_data.get("detail_analysis_performed"):
            detail_note = "\n(상세 페이지 분석 완료)"
        
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 친절한 쇼핑 AI 어시스턴트입니다.
검색 결과를 바탕으로 사용자에게 자연스럽고 도움이 되는 답변을 해주세요.
답변은 간결하고 핵심적인 정보만 포함하세요.
상세 분석이 수행된 경우, 분석 결과를 간단히 언급해주세요."""
                },
                {
                    "role": "user",
                    "content": f"""사용자 요청: {user_message}

검색 결과:{detail_note}
{products_text}

사용자에게 전달할 추천 메시지를 작성해주세요."""
                }
            ]
        )
        
        return response.choices[0].message.content
    
    async def close(self):
        """리소스 정리"""
        if self.detail_agent:
            await self.detail_agent.close()
