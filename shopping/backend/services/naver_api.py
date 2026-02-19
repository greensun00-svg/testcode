"""
Naver Shopping API Service
네이버 쇼핑 검색 API 연동 모듈
"""
import os
import httpx
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Product(BaseModel):
    """제품 정보 모델"""
    title: str
    link: str
    image: str
    lprice: int  # 최저가
    hprice: int  # 최고가
    mall_name: str
    product_id: str
    brand: str
    maker: str
    category1: str
    category2: str
    category3: str
    category4: str


class NaverShoppingAPI:
    """네이버 쇼핑 검색 API 클라이언트"""
    
    BASE_URL = "https://openapi.naver.com/v1/search/shop.json"
    
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("NAVER_CLIENT_ID and NAVER_CLIENT_SECRET must be set")
    
    async def search(
        self,
        query: str,
        display: int = 10,
        start: int = 1,
        sort: str = "sim",  # sim: 정확도, date: 날짜, asc: 가격 오름차순, dsc: 가격 내림차순
        filter_option: Optional[str] = None,
        exclude: Optional[str] = None
    ) -> list[Product]:
        """
        네이버 쇼핑 검색 수행
        
        Args:
            query: 검색어
            display: 결과 개수 (최대 100)
            start: 시작 위치 (최대 1000)
            sort: 정렬 방식 (sim/date/asc/dsc)
            filter_option: 필터 옵션 (naverpay 등)
            exclude: 제외 옵션 (used:rental:cbshop 등)
        
        Returns:
            검색된 제품 목록
        """
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        
        params = {
            "query": query,
            "display": min(display, 100),
            "start": min(start, 1000),
            "sort": sort
        }
        
        if filter_option:
            params["filter"] = filter_option
        if exclude:
            params["exclude"] = exclude
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.BASE_URL,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
        
        products = []
        for item in data.get("items", []):
            # HTML 태그 제거
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            
            product = Product(
                title=title,
                link=item.get("link", ""),
                image=item.get("image", ""),
                lprice=int(item.get("lprice", 0)),
                hprice=int(item.get("hprice", 0) or 0),
                mall_name=item.get("mallName", ""),
                product_id=item.get("productId", ""),
                brand=item.get("brand", ""),
                maker=item.get("maker", ""),
                category1=item.get("category1", ""),
                category2=item.get("category2", ""),
                category3=item.get("category3", ""),
                category4=item.get("category4", "")
            )
            products.append(product)
        
        return products
    
    async def search_by_price_asc(self, query: str, display: int = 10) -> list[Product]:
        """가격 낮은순으로 검색"""
        return await self.search(query, display=display, sort="asc")
    
    async def search_by_price_desc(self, query: str, display: int = 10) -> list[Product]:
        """가격 높은순으로 검색"""
        return await self.search(query, display=display, sort="dsc")
    
    async def search_by_relevance(self, query: str, display: int = 10) -> list[Product]:
        """정확도순으로 검색"""
        return await self.search(query, display=display, sort="sim")


# 테스트용 함수
async def test_search():
    """API 테스트 함수"""
    api = NaverShoppingAPI()
    products = await api.search("노트북", display=5)
    for p in products:
        print(f"{p.title}: {p.lprice:,}원 ({p.mall_name})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_search())
