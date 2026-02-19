"""
Price Agent - 가격 비교 전문 에이전트
검색 결과에서 최저가 제품을 찾고 가격 대비 가치를 분석합니다.
"""
import os
import sys
from typing import Any, List

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from services.naver_api import Product

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PriceAgent:
    """가격 비교 전문 에이전트"""
    
    def __init__(self):
        self.model = "gpt-4o-mini"
    
    def filter_by_price_range(
        self,
        products: List[Product],
        min_price: int = 0,
        max_price: float = float('inf')
    ) -> List[Product]:
        """가격 범위로 제품 필터링"""
        return [
            p for p in products
            if min_price <= p.lprice <= max_price
        ]
    
    def sort_by_price(
        self,
        products: List[Product],
        ascending: bool = True
    ) -> List[Product]:
        """가격순으로 정렬"""
        return sorted(products, key=lambda p: p.lprice, reverse=not ascending)
    
    def get_lowest_price_products(
        self,
        products: List[Product],
        top_n: int = 5
    ) -> List[Product]:
        """최저가 제품 N개 반환"""
        sorted_products = self.sort_by_price(products, ascending=True)
        return sorted_products[:top_n]
    
    async def analyze_value(
        self,
        products: List[Product],
        user_requirements: str
    ) -> List[dict]:
        """
        가격 대비 가치 분석
        
        Args:
            products: 분석할 제품 목록
            user_requirements: 사용자 요구사항
            
        Returns:
            가치 분석이 추가된 제품 정보 목록
        """
        if not products:
            return []
        
        # 제품 정보 요약
        products_summary = "\n".join([
            f"- {p.title}: {p.lprice:,}원 (브랜드: {p.brand or '미상'}, 판매처: {p.mall_name})"
            for p in products[:10]
        ])
        
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 쇼핑 가격 비교 전문가입니다.
제품 목록을 보고 가격 대비 가치를 분석해주세요.
각 제품에 대해 1-10점으로 가성비 점수를 매겨주세요.
응답 형식: 제품명|가성비점수|간단한이유
각 제품을 줄바꿈으로 구분하세요.
"""
                },
                {
                    "role": "user",
                    "content": f"""사용자 요구사항: {user_requirements}

제품 목록:
{products_summary}

각 제품의 가성비를 분석해주세요."""
                }
            ]
        )
        
        analysis_text = response.choices[0].message.content
        results = []
        
        for i, line in enumerate(analysis_text.strip().split("\n")):
            if i >= len(products):
                break
            
            parts = line.split("|")
            score = 5  # 기본 점수
            reason = ""
            
            if len(parts) >= 2:
                try:
                    score = int(parts[1].strip())
                except ValueError:
                    score = 5
            
            if len(parts) >= 3:
                reason = parts[2].strip()
            
            product = products[i]
            results.append({
                **product.model_dump(),
                "value_score": score,
                "value_reason": reason
            })
        
        # 분석되지 않은 제품들 추가
        for i in range(len(results), len(products)):
            results.append({
                **products[i].model_dump(),
                "value_score": 5,
                "value_reason": ""
            })
        
        return results
