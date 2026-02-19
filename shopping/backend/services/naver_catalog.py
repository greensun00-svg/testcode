"""
Naver Catalog Spec Service
네이버 쇼핑 카탈로그에서 제품 상세 스펙 정보 조회
"""
import os
import re
import json
import httpx
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class NaverCatalogAPI:
    """
    네이버 쇼핑 카탈로그 스펙 조회 서비스
    
    네이버 카탈로그 페이지 (search.shopping.naver.com/catalog/{productId})에서
    제품 스펙 정보를 추출합니다.
    """
    
    # 카탈로그 API 엔드포인트 (비공식)
    CATALOG_API_URL = "https://search.shopping.naver.com/api/catalog/{product_id}"
    
    # 모바일 API (더 간결한 데이터)
    MOBILE_API_URL = "https://msearch.shopping.naver.com/catalog/{product_id}"
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://search.shopping.naver.com/",
        }
    
    async def get_product_specs(self, product_id: str) -> Dict[str, Any]:
        """
        제품 ID로 상세 스펙 조회
        
        Args:
            product_id: 네이버 쇼핑 제품 ID
            
        Returns:
            스펙 정보 딕셔너리
        """
        specs = {}
        
        # 1. 카탈로그 API 시도
        try:
            specs = await self._fetch_from_catalog_api(product_id)
            if specs:
                return specs
        except Exception:
            pass
        
        # 2. 페이지 데이터 추출 시도
        try:
            specs = await self._fetch_from_catalog_page(product_id)
            if specs:
                return specs
        except Exception:
            pass
        
        return {"error": "스펙 정보를 가져올 수 없습니다.", "product_id": product_id}
    
    async def _fetch_from_catalog_api(self, product_id: str) -> Dict[str, Any]:
        """카탈로그 API에서 데이터 가져오기"""
        url = f"https://search.shopping.naver.com/api/catalog/{product_id}/spec"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=self.headers,
                timeout=10,
                follow_redirects=True
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_spec_data(data)
        
        return {}
    
    async def _fetch_from_catalog_page(self, product_id: str) -> Dict[str, Any]:
        """카탈로그 페이지에서 __NEXT_DATA__ JSON 추출"""
        url = f"https://search.shopping.naver.com/catalog/{product_id}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=self.headers,
                timeout=15,
                follow_redirects=True
            )
            
            if response.status_code == 200:
                html = response.text
                
                # __NEXT_DATA__ 스크립트에서 JSON 추출
                match = re.search(
                    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                    html,
                    re.DOTALL
                )
                
                if match:
                    try:
                        data = json.loads(match.group(1))
                        return self._extract_specs_from_next_data(data, product_id)
                    except json.JSONDecodeError:
                        pass
        
        return {}
    
    def _parse_spec_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """API 응답에서 스펙 정보 파싱"""
        specs = {
            "product_id": data.get("id", ""),
            "name": data.get("name", ""),
            "specs": {},
            "attributes": []
        }
        
        # 스펙 항목 추출
        spec_groups = data.get("specGroups", [])
        for group in spec_groups:
            group_name = group.get("groupName", "")
            for item in group.get("specs", []):
                key = item.get("name", "")
                value = item.get("value", "")
                if key and value:
                    specs["specs"][f"{group_name}_{key}"] = value
        
        return specs
    
    def _extract_specs_from_next_data(self, data: Dict[str, Any], product_id: str) -> Dict[str, Any]:
        """__NEXT_DATA__에서 스펙 정보 추출"""
        result = {
            "product_id": product_id,
            "name": "",
            "specs": {},
            "size_info": {},
            "attributes": []
        }
        
        try:
            # pageProps에서 데이터 추출
            props = data.get("props", {}).get("pageProps", {})
            
            # 카탈로그 정보
            catalog = props.get("catalog", {})
            result["name"] = catalog.get("name", "")
            
            # 스펙 정보
            spec_info = props.get("spec", {}) or catalog.get("spec", {})
            if spec_info:
                for group in spec_info.get("groups", []):
                    group_name = group.get("name", "")
                    for item in group.get("items", []):
                        key = item.get("name", "")
                        value = item.get("value", "")
                        if key and value:
                            result["specs"][key] = value
                            
                            # 크기 정보 특별 처리
                            if any(x in key.lower() for x in ["크기", "사이즈", "높이", "폭", "깊이", "가로", "세로", "넓이"]):
                                result["size_info"][key] = value
            
            # 속성 정보
            attrs = props.get("attributes", [])
            for attr in attrs:
                result["attributes"].append({
                    "name": attr.get("name", ""),
                    "value": attr.get("value", "")
                })
                
                # 스펙에도 추가
                key = attr.get("name", "")
                value = attr.get("value", "")
                if key and value and key not in result["specs"]:
                    result["specs"][key] = value
            
        except Exception:
            pass
        
        return result
    
    async def get_specs_for_products(
        self,
        products: List[Dict[str, Any]],
        max_products: int = 10
    ) -> List[Dict[str, Any]]:
        """
        여러 제품의 스펙 정보 조회
        
        Args:
            products: 제품 목록 (product_id 필드 필요)
            max_products: 최대 조회 개수
            
        Returns:
            스펙 정보가 추가된 제품 목록
        """
        results = []
        
        for product in products[:max_products]:
            product_id = product.get("product_id", "")
            
            if not product_id:
                results.append({**product, "catalog_specs": None})
                continue
            
            try:
                specs = await self.get_product_specs(product_id)
                results.append({
                    **product,
                    "catalog_specs": specs
                })
            except Exception as e:
                results.append({
                    **product,
                    "catalog_specs": {"error": str(e)}
                })
        
        # 나머지 제품은 스펙 없이 추가
        for product in products[max_products:]:
            results.append({**product, "catalog_specs": None})
        
        return results
    
    def analyze_specs_match(
        self,
        specs: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        스펙이 요구사항과 일치하는지 분석
        
        Args:
            specs: 제품 스펙 정보
            requirements: 사용자 요구사항
                - height_max: 최대 높이 (cm)
                - width: 가로 (cm)
                - depth: 세로 (cm)
                - steps: 단계 수 (예: 3)
                
        Returns:
            분석 결과
        """
        result = {
            "match_score": 0,
            "matches": [],
            "mismatches": [],
            "unknown": []
        }
        
        all_specs = specs.get("specs", {})
        size_info = specs.get("size_info", {})
        spec_text = " ".join(str(v) for v in all_specs.values()).lower()
        
        # 가로 체크
        target_width = requirements.get("width")
        if target_width:
            width_found = self._check_dimension(all_specs, size_info, spec_text, "가로", "폭", "너비", "width", target_value=target_width)
            if width_found == "match":
                result["matches"].append(f"가로 {target_width}cm")
                result["match_score"] += 25
            elif width_found == "mismatch":
                result["mismatches"].append(f"가로 {target_width}cm")
            else:
                result["unknown"].append(f"가로 {target_width}cm")
        
        # 세로/깊이 체크
        target_depth = requirements.get("depth")
        if target_depth:
            depth_found = self._check_dimension(all_specs, size_info, spec_text, "세로", "깊이", "depth", target_value=target_depth)
            if depth_found == "match":
                result["matches"].append(f"세로 {target_depth}cm")
                result["match_score"] += 25
            elif depth_found == "mismatch":
                result["mismatches"].append(f"세로 {target_depth}cm")
            else:
                result["unknown"].append(f"세로 {target_depth}cm")
        
        # 높이 체크 (최소높이)
        height_max = requirements.get("height_max")
        if height_max:
            height_found = self._check_height_range(all_specs, size_info, spec_text, max_height=height_max)
            if height_found == "match":
                result["matches"].append(f"높이 {height_max}cm 이하")
                result["match_score"] += 25
            elif height_found == "mismatch":
                result["mismatches"].append(f"높이 {height_max}cm 이하")
            else:
                result["unknown"].append(f"높이 {height_max}cm 이하")
        
        # 단계 수 체크
        target_steps = requirements.get("steps")
        if target_steps:
            steps_found = self._check_steps(all_specs, spec_text, target_steps)
            if steps_found == "match":
                result["matches"].append(f"{target_steps}단")
                result["match_score"] += 25
            elif steps_found == "mismatch":
                result["mismatches"].append(f"{target_steps}단")
            else:
                result["unknown"].append(f"{target_steps}단")
        
        return result
    
    def _check_dimension(
        self,
        specs: Dict,
        size_info: Dict,
        spec_text: str,
        *keywords,
        target_value: int,
        tolerance: int = 5
    ) -> str:
        """치수 확인 (match/mismatch/unknown)"""
        import re
        
        combined_text = spec_text + " " + " ".join(str(v) for v in size_info.values())
        
        for keyword in keywords:
            # 정확한 값 찾기
            pattern = rf"{keyword}\s*[:\s]*(\d+)"
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                found_value = int(match.group(1))
                if abs(found_value - target_value) <= tolerance:
                    return "match"
                else:
                    return "mismatch"
        
        # 크기 표기 (예: 1400x800)
        size_pattern = r"(\d{3,4})\s*[xX×]\s*(\d{3,4})"
        matches = re.findall(size_pattern, combined_text)
        for w, d in matches:
            w_cm = int(w) if int(w) < 1000 else int(w) / 10
            d_cm = int(d) if int(d) < 1000 else int(d) / 10
            
            if abs(w_cm - target_value) <= tolerance or abs(d_cm - target_value) <= tolerance:
                return "match"
        
        return "unknown"
    
    def _check_height_range(
        self,
        specs: Dict,
        size_info: Dict,
        spec_text: str,
        max_height: int
    ) -> str:
        """높이 범위 확인 (최소 높이가 max_height 이하인지)"""
        import re
        
        combined_text = spec_text + " " + " ".join(str(v) for v in size_info.values())
        
        # 높이 범위 패턴 (예: 60~120cm, 60-120cm)
        range_pattern = r"높이\s*[:\s]*(\d+)\s*[~\-]\s*(\d+)"
        match = re.search(range_pattern, combined_text, re.IGNORECASE)
        if match:
            min_h = int(match.group(1))
            if min_h <= max_height:
                return "match"
            else:
                return "mismatch"
        
        # 최소 높이만 표기
        min_pattern = r"최소\s*높이\s*[:\s]*(\d+)"
        match = re.search(min_pattern, combined_text, re.IGNORECASE)
        if match:
            min_h = int(match.group(1))
            if min_h <= max_height:
                return "match"
            else:
                return "mismatch"
        
        return "unknown"
    
    def _check_steps(
        self,
        specs: Dict,
        spec_text: str,
        target_steps: int
    ) -> str:
        """단계 수 확인"""
        step_keywords = [
            f"{target_steps}단",
            f"{target_steps}step",
            f"{target_steps}-step",
            f"{target_steps}단계"
        ]
        
        text_lower = spec_text.lower()
        for keyword in step_keywords:
            if keyword.lower() in text_lower:
                return "match"
        
        # 다른 단계 수 체크
        for i in [2, 4]:
            other_keywords = [f"{i}단", f"{i}step", f"{i}-step"]
            for keyword in other_keywords:
                if keyword.lower() in text_lower:
                    return "mismatch"
        
        return "unknown"


# 테스트용 함수
async def test_catalog():
    """카탈로그 API 테스트"""
    api = NaverCatalogAPI()
    
    # 테스트용 제품 ID (모션데스크)
    test_ids = ["42594163618", "47098156618"]
    
    for product_id in test_ids:
        print(f"\n=== Product ID: {product_id} ===")
        specs = await api.get_product_specs(product_id)
        print(f"Name: {specs.get('name', 'N/A')}")
        print(f"Specs: {specs.get('specs', {})}")
        print(f"Size Info: {specs.get('size_info', {})}")
        
        # 요구사항 매칭 테스트
        requirements = {
            "width": 140,
            "depth": 80,
            "height_max": 70,
            "steps": 3
        }
        match_result = api.analyze_specs_match(specs, requirements)
        print(f"\nMatch Analysis:")
        print(f"  Score: {match_result['match_score']}")
        print(f"  Matches: {match_result['matches']}")
        print(f"  Mismatches: {match_result['mismatches']}")
        print(f"  Unknown: {match_result['unknown']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_catalog())
