"""
API 테스트 스크립트 - 한글 인코딩 정상 저장
"""
import asyncio
import json
import httpx


async def test_api():
    url = "http://localhost:8000/chat"
    query = "모션 데스크를 구매하고 싶어. 높이가 70cm 이하로 내려갈 수 있고 가로 크기는 140cm 넓이는 80cm 였으면 좋겠어. 2 step말고 3step으로 찾아줘."
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            url,
            json={"message": query}
        )
        data = response.json()
    
    # 한글이 정상적으로 보이는 JSON 파일로 저장
    with open("test_result_korean.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("✅ 저장 완료: test_result_korean.json")
    print(f"\n📋 응답 메시지:\n{data.get('message', 'N/A')[:500]}")
    
    # 매칭된 제품 출력
    products = data.get("products", [])
    print(f"\n📦 발견된 제품 ({len(products)}개):")
    for i, p in enumerate(products[:5]):
        detail = p.get("detail_analysis") or {}
        match = "✅" if detail.get("is_match") else "❌"
        score = detail.get("match_score", "-")
        print(f"  {i+1}. {match} {p.get('title', 'N/A')[:40]}... (점수: {score})")


if __name__ == "__main__":
    asyncio.run(test_api())
