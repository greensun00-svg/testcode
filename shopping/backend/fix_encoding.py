"""
JSON 파일 한글 인코딩 수정
"""
import json

# 원본 파일 읽기 (인코딩 자동 감지)
with open('test_result3.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# UTF-8로 올바르게 저장 (한글 유지)
with open('test_result_readable.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("완료! test_result_readable.json 파일을 확인하세요.")
