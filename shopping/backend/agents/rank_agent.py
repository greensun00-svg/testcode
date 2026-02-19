"""
Rank Agent - 결과 랭킹 전문 에이전트
사용자 요구사항과의 유사도를 계산하고 최종 추천 순위를 결정합니다.
"""
import os
from typing import Any, List, Dict, Optional
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RankAgent:
    """결과 랭킹 전문 에이전트"""
    
    def __init__(self):
        self.model = "gpt-4o-mini"
    
    async def calculate_relevance(
        self,
        products: List[Dict[str, Any]],
        user_query: str
    ) -> List[Dict[str, Any]]:
        """
        사용자 쿼리와의 관련성 점수 계산
        
        Args:
            products: 제품 정보 목록 (딕셔너리)
            user_query: 사용자의 원본 쿼리
            
        Returns:
            관련성 점수가 추가된 제품 정보 목록
        """
        if not products:
            return []
        
        # 제품 정보 요약
        products_summary = "\n".join([
            f"{i+1}. {p['title']} - {p['lprice']:,}원 (브랜드: {p.get('brand', '미상')})"
            for i, p in enumerate(products[:15])
        ])
        
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 제품 추천 전문가입니다.
사용자의 요구사항을 분석하여 가장 적합한 제품을 찾아주세요.
각 제품에 대해 사용자 요구사항과의 관련성을 1-10점으로 평가해주세요.
응답 형식: 제품번호,관련성점수
한 줄에 하나씩 작성하세요.
"""
                },
                {
                    "role": "user",
                    "content": f"""사용자 요청: {user_query}

제품 목록:
{products_summary}

각 제품의 관련성 점수를 매겨주세요."""
                }
            ]
        )
        
        relevance_text = response.choices[0].message.content
        relevance_scores = {}
        
        for line in relevance_text.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    idx = int(parts[0].strip().rstrip(".")) - 1
                    score = int(parts[1].strip())
                    relevance_scores[idx] = score
                except ValueError:
                    continue
        
        # 점수 적용
        for i, product in enumerate(products):
            product["relevance_score"] = relevance_scores.get(i, 5)
        
        return products
    
    def rank_products(
        self,
        products: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        종합점수를 계산하여 제품 랭킹
        
        Args:
            products: 제품 정보 목록
            weights: 가중치 설정 (기본: relevance=0.4, value=0.3, price=0.3)
            
        Returns:
            종합점수로 정렬된 제품 목록
        """
        if not products:
            return []
        
        if weights is None:
            weights = {
                "relevance": 0.4,  # 관련성
                "value": 0.3,      # 가성비
                "price": 0.3       # 가격 (낮을수록 좋음)
            }
        
        # 가격 정규화 (가장 비싼 제품 기준)
        max_price = max(p.get("lprice", 1) for p in products) or 1
        
        for product in products:
            relevance = product.get("relevance_score", 5)
            value = product.get("value_score", 5)
            price = product.get("lprice", max_price)
            
            # 가격 점수: 낮을수록 높은 점수 (0-10)
            price_score = 10 * (1 - price / max_price)
            
            # 종합 점수 계산
            total_score = (
                weights["relevance"] * relevance +
                weights["value"] * value +
                weights["price"] * price_score
            )
            
            product["total_score"] = round(total_score, 2)
        
        # 종합점수로 정렬
        ranked = sorted(products, key=lambda p: p.get("total_score", 0), reverse=True)
        
        # 순위 추가
        for i, product in enumerate(ranked):
            product["rank"] = i + 1
        
        return ranked
    
    def get_top_recommendations(
        self,
        products: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """상위 N개 추천 제품 반환"""
        ranked = self.rank_products(products)
        return ranked[:top_n]
