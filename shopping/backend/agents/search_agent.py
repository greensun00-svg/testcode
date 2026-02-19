"""
Search Agent - 제품 검색 전문 에이전트
사용자 쿼리를 분석하여 적절한 검색어로 변환하고 네이버 쇼핑 API를 호출합니다.
"""
import os
import sys
from typing import Any, List

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from services.naver_api import NaverShoppingAPI, Product

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SearchAgent:
    """제품 검색 전문 에이전트"""
    
    def __init__(self):
        self.naver_api = NaverShoppingAPI()
        self.model = "gpt-4o-mini"  # 검색에는 빠른 모델 사용
    
    async def extract_keywords(self, user_query: str) -> List[str]:
        """
        사용자 쿼리에서 검색 키워드를 추출
        
        Args:
            user_query: 사용자의 자연어 쿼리
            
        Returns:
            검색에 사용할 키워드 목록
        """
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": """당신은 쇼핑 검색 키워드 추출 전문가입니다.
사용자의 요청에서 쇼핑 검색에 적합한 키워드를 추출해주세요.
응답은 키워드만 쉼표로 구분하여 출력하세요.
예시: "가성비 좋은 무선 이어폰 추천해줘" → "무선 이어폰, 블루투스 이어폰"
"""
                },
                {"role": "user", "content": user_query}
            ]
        )
        
        keywords_text = response.choices[0].message.content
        keywords = [kw.strip() for kw in keywords_text.split(",")]
        return keywords
    
    async def search(self, user_query: str, max_results: int = 20) -> List[Product]:
        """
        사용자 쿼리를 기반으로 제품 검색 수행
        
        Args:
            user_query: 사용자의 자연어 쿼리
            max_results: 최대 결과 수
            
        Returns:
            검색된 제품 목록
        """
        # 키워드 추출
        keywords = await self.extract_keywords(user_query)
        
        all_products = []
        seen_ids = set()
        
        # 각 키워드로 검색 수행
        for keyword in keywords[:3]:  # 최대 3개 키워드만 사용
            products = await self.naver_api.search(
                query=keyword,
                display=max_results // len(keywords)
            )
            
            for product in products:
                if product.product_id not in seen_ids:
                    seen_ids.add(product.product_id)
                    all_products.append(product)
        
        return all_products[:max_results]
    
    def to_dict(self, products: List[Product]) -> List[dict]:
        """제품 목록을 딕셔너리 형태로 변환"""
        return [p.model_dump() for p in products]
