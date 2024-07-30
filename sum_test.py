# import torch
# from transformers import PreTrainedTokenizerFast
# from transformers import BartForConditionalGeneration

# tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/user/.cache/huggingface/hub/models--gogamza--kobart-summarization/snapshots/31f181b155a0ad74bd93bd90ee04310ff72691f4')
# model = BartForConditionalGeneration.from_pretrained('/home/user/.cache/huggingface/hub/models--gogamza--kobart-summarization/snapshots/31f181b155a0ad74bd93bd90ee04310ff72691f4')

# text = """

# KT가 사업부서에 필요한 기술을 가진 벤처·스타트업에게 직접 협력을 제안하는 리버스피칭 방식의 스타트업 육성 시스템을 선보인다.

# KT는 판교사옥 내 오픈이노베이션 센터에서 이 같은 방식의 'BM Around' 행사를 진행했다. 리버스피칭은 벤처·스타트업이 대기업을 찾아가 자사 제품을 소개 제안하는 것이 아닌, 혁신적 기술을 필요로 하는 수요 기업이 역으로 협력 방안을 제시하는 방식을 말한다.

# 판교 오픈이노베이션 센터에는 인공지능(AI), 클라우드, 모빌리티, 양자컴퓨팅 등 유망 사업 8개 분야의 기술 역량을 보유한 △딥네츄럴 △젠젠에이아이 △라이브데이터 △오투오(이상 AI 분야) △실크로드소프트(클라우드) △에스큐케이(양자컴퓨팅) △지오소프트(모빌리티) △오투플러스(물류) △미러(교육) △원컵(프롭테크) △페보(헬스케어) △띵스넷(IoT) 등 12개 기업이 입주해 있다.

# KT는 PoC(실증사업) 추진과 함께 기업 진단부터 사업 역량 향상을 위한 멘토링 프로그램 등을 지원할 예정이다. 이번 리버스 피칭에는 KT의 AI, 교육, 물류, 양자컴퓨팅 등 사업·기술부서와 함께 KT클라우드, KT에스테이트, 밀리의서재 등 그룹사가 참여했다. 이들은 사업 현황과 계획을 공유하고 벤처·스타트업과의 협력을 위한 교류 시간(밍글링)을 가졌다.

# 이 자리에서 교육 관련 AI 플랫폼을 연구개발하는 KT 부서는AI 교수 학습 플랫폼 고도화를 위한 AI 학습지원 챗봇 기술, 서술·논술형 평가에 필요한 핵심 기술 보유 기업과 협력이 필요하다고 제안했고, 이 같은 기술을 보유한 '라이브데이터'와 후속 미팅을 통해 구체화 논의를 진행하기로 했다.

# 또한 KT는 스타트업 육성 액셀러레이터로 탭엔젤파트너스를 선발해 IR, 멘토링 등의 육성 프로그램과 KT 사업협력 검증을 위한 실증 사업 등을 보다 전문적으로 지원할 계획이다

# 임현규 KT 경영기획부문장(부사장)은 “벤처·스타트업과 실질적 협력관계로 나아가기 위해서는 기업 자체 성장에서부터 실증사업까지 전주기에 걸친 육성프로그램 운영이 필수적”이라며 “KT가 앞장서서 모범적인 대기업·스타트업 사업협력 사례를 만들어 나가겠다”고 말했다.
# """

# len_text = len(text)
# len_sum = int(0.1 * len_text)
# print(len_text, len_sum)

# raw_input_ids = tokenizer.encode(text)
# input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

# summary_ids = model.generate(torch.tensor([input_ids]), max_length=len_sum, do_sample=True, top_k=30, top_p=0.95, num_beams=4, temperature=0.6)
# print(tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True).strip())



import requests

host = "http://127.0.0.1:8899"
url = f"{host}/summarize"
text = """
올해 저는 사업 종목을 확장하려고 합니다. 영업 면허증의 영업 범위를 변경하려고 했을 때 매장이 이상한 상태로 고지되었고 어떻게 해야 하는지 알고 싶습니다." 산동 (山东) 의 한 육류 가게 책임자 왕 닝이 (王宁) 마을 및 마을 주민 서비스 센터에서 질문합니다. "조사 결과, 왕 닝 (王宁)의 가게는 정기 보고서를 제출하지 않아 이상한 경매 명단에 올랐습니다. 현지에서 정기 보고서 감독 "하나의 일"을 시행하고 여러 가지 대책을 시행하여 기업을 지원하여 오염을 제거하고 신뢰를 높이고, 시 및 시 주민 서비스 센터에 "정기 보고서 지침 상담"을 열어 정기 보고서에 관한 정책을 보다 잘 알리고, 현장에서 행동 주체가 온라인 정기 보고서를 작성하고, 온라인 거래 이상 명단에 신용 복구 요청을 접수하도록 지원합니다. "저는 마을 및 마을 주민 서비스 센터의 직원이 현장에서 저희 가게의 정기 보고서를 작성하도록 도와주었고 요청 서류를 제출했습니다. 예상치 못한 것은 백그라운드에서 즉시 승인을 받았고 처리, 가게 상태를 정상화 할 수있었습니다. 새로운 영업 면허증을 쉽게 얻을 수있었습니다." 왕 닝이 말했다. "신용 감독 사무소의 책임자가 설명했습니다. "신용 감독을 향상시키고 여권 복구 보조 수단의 편의성을 높이기 위해 2023년 5월 시장 감독 총국에서 《엄격하게 위반 및 불안정한 행동 명단 및 관리 처분 공개 정보 신용 복구 관리 절차》(시험)를 발표했습니다. 이를 통해 시장 감독 총국 관련 사무소의 책임 분담과 신용 복구 프로세스의 표준화를 추가로 명시하고 국가 기업 신용 정보 공개 시스템의 업그레이드를 통해 "신용 복구" 모듈을 개선해 현재 전국의 신용 복구 "한 곳에서 모든 것"을 실현했습니다. "모든 수준의 시장 감독 부서는 국가 기업 신용 정보 공개 시스템에 기반을 둔 온라인 신용 복구를 실현했습니다. 기업은 국가 기업 신용 정보 공개 시스템에 등록하고 실명으로 로그인하면 온라인에서 신용 복구 서류를 다운로드, 신용 복구 신청서를 제출, 실시간으로 신용 복구 진행 상황을 확인하고 온라인으로 신용 복구 결정 문서를 수령 할 수 있습니다. 동시에 온라인 수령 서비스를 보존하고 있으며 기업은 여전히 지역 시장 감독 부서에 신용 복구를 신청하고 직원이 현장에서 직접 신용 복구 서류를 작성, 신용 복구 신청서를 제출 할 수 있습니다.
"""

# headers = {'Content-Type': 'application/json'}
data = {
    "src": str(text),
    "sum_rate": 0.5,
}

r = requests.post(url=url, json=data)
print(r.json())