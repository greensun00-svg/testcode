"""
Playwright Stealth 테스트 스크립트
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.detail_agent import DetailAgent


async def test_stealth():
    """Stealth 모드 테스트"""
    agent = DetailAgent()
    
    try:
        # 네이버 쇼핑 페이지 테스트
        test_url = "https://search.shopping.naver.com/catalog/42594163618"
        
        print("=== Playwright Stealth Test ===")
        print(f"URL: {test_url}")
        print("")
        
        content = await agent.extract_page_content(
            test_url,
            {"title": "아이픽스 모션데스크 MD301", "product_id": "42594163618", "lprice": 239000}
        )
        
        print(f"Source: {content.get('source')}")
        print(f"Text length: {len(content.get('text', ''))}")
        print(f"Images count: {len(content.get('images', []))}")
        print(f"Specs count: {len(content.get('specs', {}))}")
        
        if content.get("error"):
            print(f"Error: {content.get('error')}")
            print("FAILED - Bot detection may still be active")
        else:
            print("SUCCESS - Page content extracted!")
            print(f"\nText preview:")
            print(content.get('text', '')[:800])
            print("\n...")
            
            if content.get('specs'):
                print("\nSpecs found:")
                for k, v in list(content.get('specs', {}).items())[:10]:
                    print(f"  - {k}: {v}")
            
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(test_stealth())
