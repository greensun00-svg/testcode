"""간단한 테스트 - 결과를 파일로 저장"""
import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from agents import SupervisorAgent

async def test():
    query = "모션 데스크를 구매하고 싶어. 높이가 70cm 이하로 내려갈 수 있고 가로 크기는 140cm 넓이는 80cm 였으면 좋겠어. 2 step말고 3step으로 찾아줘."
    
    supervisor = SupervisorAgent(enable_detail_analysis=True)
    try:
        result = await supervisor.process_request(query, enable_detail=True)
        
        # 결과를 파일로 저장
        with open("test_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        print("결과가 test_result.json에 저장되었습니다.")
        print(f"성공: {result['success']}")
        print(f"제품 수: {len(result.get('products', []))}")
        
    finally:
        await supervisor.close()

if __name__ == "__main__":
    asyncio.run(test())
