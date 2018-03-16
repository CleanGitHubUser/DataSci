# ML02
# 나이브 베이즈 알고리즘

# 동전을 100회 던졌을 때 앞면이 나오는 횟수는 대략 50회 정도로 예상
# 전통적인 확률 계산은 '일어난횟수/전체시도횟수' 를 이용
# 이러한 확률론은 경험확률이라 함 - 일정한 확률로 반복시행

# A라는 도시에 철수라는 아이가 태어났다. 이 아이가 노벨상을
# 받을 확률은 얼마나 될까?
# 이것을 경험확률로 알아보려면 이 아이를 여러명 살게 하고
# 그 중 몇명이 노벨상을 받았는지 평가해야 한다
# 동일한 아이가 전 세계적으로 몇명있는지 파악하고
# 몇명이 커서 노벨상을 받았는지 평가해야 하는데
# 동일한 유전자, 환경에 자란 아이를 만들기엔 불가능함

# 이런 경우, 베이즈 확률론을 이용한다
# '일어나지 않은 일에 대한 확률'을
# 불확실성이라는 개념으로 이야기함
# 즉, 이 사건과 관련된 여러가지 확률을 이용해서
# 새롭게 일어날수 있는 사건에 대해 추정하는 것
# 이러한 일을 베이즈 이론, 베이즈 추론이라 함

# 심리학, 신경과학, 인지과학, 인공지능, 기계학습 등 분야에서는
# 베이즈 정리가 바로 인간이 생각하고 판단하는 근본적인 방식
# 즉, 인간의 사고는 처음에는 아무 정보가 없던 산태에서
# 새로운 정보를 받아들이고 이를 통해 자신이 갖고 있던
# 일종의 사전확률체계를 업데이트 시켜 세상을 해석하거나
# 판단하고 의사결정하는 방법으로 발전해옴

# 인간의 뇌나 마음이 정보를 처리하는 방식이 베이즈 정리를
# 닮아있다는 가설을 실험이나 모델링으로 검증했고
# 인공지능/기계학습 분야에서는 베이즈 정리를 기초로 하는
# 기법들을 많이 발전시켜 옴

# 베이즈 정리 기반 확률문제
# 몬티홀 문제
# 미국의 티비 게임쇼에서 유래한 퍼즐
# 3개의 문 중 하나를 선택하여 문 뒤에 있는 상품을 가질 수 있는 쇼
# 하나의 문 뒤에는 스포츠카가 있고 나머지 두 문 뒤에는 염소가 있음
# 출연자가 한 문을 선택하고 진행자는 다른 문을 열어
# 염소가 있음을 보여주고 다른 문을 선택하겠냐고 물음

# 이 때 출연자가 처음 선택을 고수했을 때 스포츠카를 갖게 될 확률은?
# 출연자가 스포츠카를 가지려면 원래 선택했던 문을 바꾸는 것이 유리?


# 아들 딸 패러독스
# 두 아이가 있는 어떤 집에서 첫째아이가 남자일 때
# 두 아이가 모두 남자일 확률은?

# 두 아이가 있는 어떤 집에서 두 아이 중 한 명이 남자일 때
# 두 아이 모두 남자일 확률은?

# 어떤 청바지가 적재되는 창고가 있는데
# 이 창고의 청바지를 하나 골랐는데 불량 청바지였다
# 청바지는 구미, 대구, 청주에서 생산되어 운송되었는데
# 이 불량 청바지는 어느 지역에서 생산된 것일까?


# 베이즈 정리
# 이전의 경험과 현재의 증거를 토대로
# 어떤 사건의 확률을 추론하는 알고리즘
# 따라서, 사건이 일어날 확률을 토대로 의사결정을 할 경우
# 그와 관련된 사전 정보를 얼마나 알고있나에 따라 크게 좌우
# 베이즈 정리는 조건부 확률로도 불린다

# 다양한 폐암 증상들
# 숨을 제대로 쉬기 어렵다
# 삼키기 어려울 정도의 목이 아프다 - 인후염

# 병원 방문 후 (검사 정확도 90%) 검사 - 검사 결과 : 양성
# 이 결과로 페암일 확률은 10%도 안될 수 있음
# 폐암에 걸린 남성은 성인 남성의 1% 정도 - 추가 검사 시행 : 음성

# 베이즈 정리에 근거, 실제 검사에서 양성이 나왔어도
# 진짜 폐암에 걸릴 확률은 8.3% 밖에 되지 않음

# 시간이 지나 다시 목이 아프고 숨을 쉬기 어려워서 병원에 감
# 검사해보니 역시 양성 - 예전 경험에 비춰 별거 아니라고 생각
# 이 결과로 폐암일 확률은 50% - 절대 심각

# 검사 정확도 : 90% -> 99%
# 양성이면 폐암일 확률이 예전에는 8.3%였지만 지금은 50%


# 확률 이론
# 주사위 1개를 던져 나오는 눈의 수를 생각
# 주사위를 던지는 행위 - 시행
# 시행으로 얻어진 결과 중 조건과 맞는 결과집합 - 사상
# 예) 주사위를 던져 홀수가 나오는 사상 : 1, 3, 5

# 확률 P
# 사상이 일어날 경우의 수 / 일어날 수 있는 모든 경우의 수 (전사상)
# 주사위를 던졌을 때 4 이하 눈이 나올 확률 P(A) : 2/3
# 주사위를 던졌을 때 짝수 눈이 나올 확률 P(B) : 1/2

# 곱사상
# 두 사상 A, B가 동시에 일어날 확률 (동시확률)
# 예) 4 이하이며A 짝수 눈이 나올 확률 : P(A∩B) = 2/6
# 4이하A : 1, 2, 3, 4
# 짝수B : 2, 4, 6

# 합사상
# 사상 A가 일어났다는 조건 아래 B가 일어날 확률 (조건부확률)

# 예) 4이하 눈이 나왔을 때A, 짝수의 눈B이 나올 확률 : P(B|A) = 2/4
# 4이하 눈이 나왔을때 : 1, 2, 3, 4
# 짝수의 눈B이 나올 때 : 2, 4, 6

# 예) 짝수의 눈이 나왔을 때B, 4이하 눈A이 나올 확률 : P(A|B) =  2/3
# 짝수의 눈B이 나올 때 : 2, 4, 6
# 4이하 눈이 나왔을때 : 1, 2, 3, 4

# 승법정리 (확률의 곱법칙, 곱셈정리)
# P(A∩B) = P(A)P(B|A) = P(B)P(A|B)
# A, B가 동시에 일어날 확률은
# A가 일어날 확률에
# A가 일어났을 때 B가 일어날 확률을 곱한 것

# 베이즈 정리 - 승법정리를 활용
# P(A|B) = P(A)P(B|A)/P(B)

# 즉, 사건 B가 일어났을 때 사건 A가 일어날 확률(사후확률)은
# 사건 A가 일어날 확률(사전확률)과
# 사건 A가 일어났을 때 사건 B가 일어날 조건부 확률의 곱을
# 사건B가 일어날 확률로 나누어 알아낼 수 있다는 뜻

# 사전확률이 주어지고, 이에 따른 조건부확률 및 주변확률을 통해
# 조건과 결과를 서로 바꿔서 사후확률을 계산하는 공식

# 보통 베이즈 정리에서 A는 원인/가정, B는 결과/데이터를 의미


# 베이즈 정리를 이용해서 문제를 푸는 방법
# 사전확률 : 관측자가 이미 알고 있는 사건으로부터 나온 확률 P(A), P(B)

# 가능도/우도(주변확률) : 알고 있는 사건이 발생했다는 조건하
# 다른 사건이 발생할 확률 P(B|A), P(A|B), ...

# 사후 확률 : 사전확률과 주변확률로 알게되는 조건부 확률


# 정확도 90% 폐암 검사 - 검사 결과 : 양성
# 예측결과 페암일 확률은 10%도 안될 수 있음
# 폐암에 걸린 남성은 성인 남성의 1% 정도 (사전확률)

# 사전확률 : P(암) = 0.01 = P(A)
#            P(양성) = P(B) = 0.108
# 주변확률 : P(양성|암) = 0.9 = P(B|A)

# P(A|B) = P(A)P(B|A)/P(B)
# 0.01 x 0.9 / 0.108 = 0.083

# 예측값 : P(암|양성) = 8.3% = P(A|B)


# 스팸일 확률 계산 :
# 사전확률 : '내가 오늘 받은 메일 중 스팸 메일이 있을
#             가능성은 대략 10%일 것 같다' : P(스팸) = 0.1

# 주변확률 : P(반짝할인) = 0.05
#            P(반짝할인 : 스팸) = 0.4

# 조건부확률 : P(스팸|반짝할인) = ?\
#              P(A|B) = P(A)P(B|A)/P(B)
#              P(반짝할인 : 스팸) x P(스팸) / P(반짝할인)
#              0.4 * 0.1 /0.05 = 0.8 = P (스팸|반짝할인)

