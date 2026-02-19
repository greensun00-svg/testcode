"""
Detail Agent 테스트 스크립트
모션 데스크 검색 결과 분석
"""
import asyncio
import os
import sys
import json

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from agents import SupervisorAgent


async def test_motion_desk():
    """모션 데스크 검색 테스트"""
    query = "모션 데스크를 구매하고 싶어. 높이가 70cm 이하로 내려갈 수 있고 가로 크기는 140cm 넓이는 80cm 였으면 좋겠어. 2 step말고 3step으로 찾아줘."
    
    print("=" * 60)
    print("테스트 쿼리:")
    print(query)
    print("=" * 60)
    
    supervisor = SupervisorAgent(enable_detail_analysis=True)
    
    try:
        result = await supervisor.process_request(query, enable_detail=True)
        
        print(f"\n✅ 성공 여부: {result['success']}")
        print(f"\n📝 AI 응답:\n{result['message']}")
        
        print(f"\n🔍 의도 분석:")
        intent = result.get('intent', {})
        print(f"  - needs_detail: {intent.get('needs_detail', False)}")
        print(f"  - detail_criteria: {intent.get('detail_criteria', [])}")
        print(f"  - detail_analysis_performed: {intent.get('detail_analysis_performed', False)}")
        
        products = result.get('products', [])
        print(f"\n📦 검색된 제품 수: {len(products)}")
        
        print("\n" + "=" * 60)
        print("상위 5개 제품 상세:")
        print("=" * 60)
        
        for i, p in enumerate(products[:5]):
            print(f"\n[{i+1}] {p.get('title', 'N/A')}")
            print(f"    가격: {p.get('lprice', 0):,}원")
            print(f"    링크: {p.get('link', 'N/A')[:80]}...")
            print(f"    종합점수: {p.get('total_score', 0)}")
            
            detail = p.get('detail_analysis')
            if detail:
                print(f"    [상세분석]")
                print(f"      - 매칭점수: {detail.get('match_score', 0)}/100")
                print(f"      - 일치여부: {detail.get('is_match', False)}")
                print(f"      - 분석요약: {detail.get('summary', 'N/A')[:100]}...")
            else:
                print(f"    [상세분석] 미수행")
        
        # 요구사항 체크
        print("\n" + "=" * 60)
        print("요구사항 충족 분석:")
        print("=" * 60)
        requirements = {
            "높이 70cm 이하": False,
            "가로 140cm": False,
            "넓이 80cm": False,
            "3step": False
        }
        
        for p in products[:5]:
            title = p.get('title', '').lower()
            detail = p.get('detail_analysis', {})
            summary = detail.get('summary', '') if detail else ''
            
            # 제목이나 분석에서 요구사항 체크
            text = f"{title} {summary}".lower()
            
            if '3단' in text or '3step' in text or '3-step' in text:
                requirements['3step'] = True
            if '140' in text:
                requirements['가로 140cm'] = True
            if '80' in text:
                requirements['넓이 80cm'] = True
            if '70' in text or '60' in text or '65' in text:
                requirements['높이 70cm 이하'] = True
        
        for req, found in requirements.items():
            status = "✅" if found else "❌"
            print(f"  {status} {req}")
        
        return result
        
    finally:
        await supervisor.close()


if __name__ == "__main__":
    asyncio.run(test_motion_desk())
