"""
Detail Agent - 상세페이지 분석 에이전트 (Hybrid Mode)
Playwright 실행 불안정성으로 인해 Naver Catalog API(HTTP)를 주력으로 사용.
"""
import os
import sys
import asyncio
import random
import re
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Playwright 타입은 런타임에 로드하지 않음 (안정성)
if TYPE_CHECKING:
    from playwright.async_api import Page, Browser

from openai import OpenAI
from dotenv import load_dotenv

# NaverCatalogAPI 임포트 (Fallback용)
from services.naver_catalog import NaverCatalogAPI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class ProductAnalysis:
    """제품 분석 결과"""
    product_id: str
    url: str
    match_score: int  # 0-100
    analysis_summary: str
    extracted_specs: Dict[str, Any]
    is_match: bool


class DetailAgent:
    """
    상세페이지 분석 에이전트 (Hybrid Mode)
    
    기본 전략:
    1. 시스템 안정성을 위해 NaverCatalogAPI (HTTP)로 데이터 조회 우선
    2. 필요한 경우에만 Playwright 실행 (현재는 비활성화 상태)
    """
    
    def __init__(self, max_concurrent: int = 3):
        self.model = "gpt-4o"
        self.max_concurrent = max_concurrent
        self.playwright = None
        self.browser = None
        self.catalog_api = NaverCatalogAPI() # API 서비스
    
    async def _get_browser(self):
        """브라우저 인스턴스 가져오기 (싱글톤)"""
        # Playwright 모듈을 여기서 임포트하여 초기 로드 시 크래시 방지
        from playwright.async_api import async_playwright
        
        if self.browser is None:
            self.playwright = await async_playwright().start()
            
            # 안정된 설정을 위한 헤드리스 모드
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                ]
            )
        return self.browser
    
    async def close(self):
        """리소스 정리"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.browser = None
        self.playwright = None
    
    async def extract_page_content(self, url: str, product_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        페이지 콘텐츠 추출
        """
        # 안정성을 위해 Playwright 대신 API를 직접 호출
        return await self._fetch_via_catalog_api(url, product_info, "Playwright skipped (safety mode)")

    async def _fetch_via_catalog_api(self, url: str, product_info: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Catalog API를 통한 데이터 수집"""
        if not product_info:
            return self._create_fallback_content(url, product_info, error)
            
        product_id = product_info.get("product_id")
        if not product_id:
            # URL에서 product_id 추출 시도
            match = re.search(r'catalog/(\d+)', url)
            if match:
                product_id = match.group(1)
            else:
                return self._create_fallback_content(url, product_info, error + " (No product_id)")
        
        # Catalog API 호출
        try:
            specs = await self.catalog_api.get_product_specs(product_id)
            
            # 텍스트로 변환하여 GPT 분석에 활용
            spec_lines = []
            if specs.get("specs"):
                for k, v in specs["specs"].items():
                    spec_lines.append(f"{k}: {v}")
            if specs.get("attributes"):
                for attr in specs["attributes"]:
                    spec_lines.append(f"{attr.get('name')}: {attr.get('value')}")
            
            text_content = f"제품명: {specs.get('name', product_info.get('title'))}\n"
            text_content += f"가격: {product_info.get('lprice', 0)}원\n\n"
            text_content += "=== 상세 스펙 ===\n" + "\n".join(spec_lines)
            
            return {
                "url": url,
                "images": [product_info.get("image")] if product_info.get("image") else [],
                "text": text_content,
                "specs": specs.get("specs", {}),     # 구조화된 스펙
                "size_info": specs.get("size_info", {}), # 사이즈 정보 별도
                "source": "catalog_api_fallback",
                "original_error": error
            }
            
        except Exception as api_error:
            # API마저 실패하면 제목 기반 분석
            return self._create_fallback_content(url, product_info, error + f" / API Error: {str(api_error)}")

    def _create_fallback_content(self, url: str, product_info: Dict[str, Any], error: str) -> Dict[str, Any]:
        """최후의 수단: 기본 정보만 반환"""
        if not product_info:
            return {"url": url, "text": "", "error": error, "source": "empty"}
        
        text = f"제품명: {product_info.get('title', '')}\n가격: {product_info.get('lprice', '')}"
        return {
            "url": url,
            "images": [product_info.get('image')] if product_info.get('image') else [],
            "text": text,
            "specs": {},
            "error": error,
            "source": "fallback_basic"
        }

    async def analyze_with_vision(
        self,
        content: Dict[str, Any],
        user_requirements: str,
        product_info: Dict[str, Any] = None
    ) -> ProductAnalysis:
        """분석 로직"""
        
        # Catalog API 결과 활용 (우선순위 높음)
        if content.get("source") == "catalog_api_fallback" and content.get("specs"):
            # 요구사항 파싱
            reqs = self._extract_requirements_from_query(user_requirements)
            
            # Catalog API의 분석 로직 사용
            match_result = self.catalog_api.analyze_specs_match(
                {"specs": content["specs"], "size_info": content.get("size_info", {})},
                reqs
            )
            
            # 결과 변환
            is_match = match_result["match_score"] >= 50 # 기준점
            
            summary_parts = []
            if match_result["matches"]:
                summary_parts.append(f"일치: {', '.join(match_result['matches'])}")
            if match_result["mismatches"]:
                summary_parts.append(f"불일치: {', '.join(match_result['mismatches'])}")
            
            summary = " / ".join(summary_parts) if summary_parts else "스펙 정보 부족으로 판단 어려움"
            
            return ProductAnalysis(
                product_id=product_info.get("product_id", "") if product_info else "",
                url=content.get("url", ""),
                match_score=min(match_result["match_score"] + 30, 95) if is_match else match_result["match_score"], # 점수 보정
                analysis_summary=summary,
                extracted_specs=content["specs"],
                is_match=is_match
            )

        # 기존 로직 (Playwright 결과 등)
        product_id = product_info.get('product_id', '') if product_info else ''
        url = content.get('url', '')
        text = content.get('text', '')[:4000]
        specs = content.get('specs', {})
        specs_text = "\n".join([f"- {k}: {v}" for k, v in list(specs.items())[:20]])
        
        prompt = f"""제품 정보를 분석하여 사용자 요구사항과 일치하는지 평가해주세요.
사용자 요구사항: {user_requirements}

제품 정보:
{text[:2000]}

스펙 정보:
{specs_text}

JSON 형식으로 응답:
{{"match_score": 0-100, "is_match": true/false, "summary": "분석 요약", "extracted_specs": {{"key": "value"}}}}
"""
        
        try:
            messages = [
                {"role": "system", "content": "당신은 쇼핑 제품 분석 전문가입니다."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            
            # 이미지 추가 고려 (현재 Catalog API 모드에서는 이미지 URL 하나만 옴)
            images = content.get('images', [])
            if images:
                for img_url in images[:1]:
                    if img_url and img_url.startswith('http'):
                        messages[1]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": img_url, "detail": "low"}
                        })
            
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=messages
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return ProductAnalysis(
                product_id=product_id,
                url=url,
                match_score=result.get('match_score', 50),
                analysis_summary=result.get('summary', '분석 완료'),
                extracted_specs=result.get('extracted_specs', {}),
                is_match=result.get('is_match', False)
            )
        except Exception as e:
            return self._analyze_from_title(product_info, user_requirements, str(e))

    def _extract_requirements_from_query(self, user_query: str) -> Dict[str, Any]:
        """사용자 쿼리에서 요구사항 추출"""
        requirements = {}
        query_lower = user_query.lower()
        
        # 가로/넓이
        width_match = re.search(r'가로\s*(?:크기)?[는가이]?\s*(\d+)\s*(?:cm)?', query_lower)
        if width_match: requirements['width'] = int(width_match.group(1))
        
        # 세로/깊이
        depth_match = re.search(r'(?:세로|넓이|깊이)\s*(?:크기)?[는가이]?\s*(\d+)\s*(?:cm)?', query_lower)
        if depth_match: requirements['depth'] = int(depth_match.group(1))
        
        # 높이 최대
        height_match = re.search(r'높이\s*(?:가|는|가|이)?\s*(\d+)\s*(?:cm)?\s*이하', query_lower)
        if height_match: requirements['height_max'] = int(height_match.group(1))
        
        # 단계
        step_match = re.search(r'(\d)\s*(?:단|step|단계)', query_lower)
        if step_match: requirements['steps'] = int(step_match.group(1))
        
        return requirements

    def _analyze_from_title(self, product_info, user_requirements, error=""):
        """제목 기반 기본 분석"""
        if not product_info:
            return ProductAnalysis("", "", 0, f"Error: {error}", {}, False)
            
        title = product_info.get('title', '').lower()
        score = 30
        matches = []
        specs = {}
        
        reqs = self._extract_requirements_from_query(user_requirements)
        
        if reqs.get('steps'):
            t = reqs['steps']
            if f'{t}단' in title or f'{t}step' in title or f'{t}단계' in title:
                score += 25
                matches.append(f"{t}단")
                specs['steps'] = t
                
        if '모션' in title or '전동' in title:
            score += 10
            matches.append("모션데스크")
            
        summary = f"제목 분석: {', '.join(matches)}" if matches else "제목 분석 실패"
        if error: summary += f" ({error})"
        
        return ProductAnalysis(
            product_id=product_info.get('product_id', ''),
            url=product_info.get('link', ''),
            match_score=min(score, 100),
            analysis_summary=summary,
            extracted_specs=specs,
            is_match=score >= 70
        )

    async def analyze_product(self, url, user_requirements, product_info=None):
        content = await self.extract_page_content(url, product_info)
        return await self.analyze_with_vision(content, user_requirements, product_info)

    async def analyze_multiple(self, products, user_requirements, max_products=5):
        to_analyze = products[:max_products]
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def analyze_with_limit(product):
            async with semaphore:
                try:
                    url = product.get('link', '')
                    await asyncio.sleep(0.5)
                    
                    analysis = await self.analyze_product(url, user_requirements, product)
                    return {
                        **product,
                        "detail_analysis": {
                            "match_score": analysis.match_score,
                            "is_match": analysis.is_match,
                            "summary": analysis.analysis_summary,
                            "specs": analysis.extracted_specs
                        }
                    }
                except Exception as e:
                    fallback = self._analyze_from_title(product, user_requirements, str(e))
                    return {
                        **product,
                        "detail_analysis": {
                            "match_score": fallback.match_score,
                            "is_match": fallback.is_match,
                            "summary": fallback.analysis_summary,
                            "specs": fallback.extracted_specs
                        }
                    }

        analyzed = await asyncio.gather(*[analyze_with_limit(p) for p in to_analyze])
        result = list(analyzed)
        for p in products[max_products:]:
            result.append({**p, "detail_analysis": None})
        return result
