<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic LSA + Lagrangian 라우터

## 단일 제출 정책

문항 `i`에서 공식 모델 `m` 하나를 선택하는 변수를 `z_im`으로 두고,
`sum_m z_im = 1`을 적용합니다. 학습한 예상 점수 `r_hat_im`과 예상 비용
`c_hat_im`에 대해 다음 MCKP를 풉니다.

```text
maximize  sum_i sum_m z_im * r_hat_im
subject to sum_i sum_m z_im * c_hat_im <= predicted_budget
           z_im in {0, 1}, sum_m z_im = 1
```

Lagrangian relaxation은 비용 제약에 가격 `lambda`를 붙입니다. 주어진
`lambda`에서는 각 문항이 독립이므로 `r_hat_im - lambda*c_hat_im`가 가장 큰
모델을 고릅니다. `lambda`는 100회 이분 탐색으로 찾습니다. 비용 예측 오차로
실제 예산을 넘는 것을 막기 위해 공식 feature-budget baseline과 같은 방식으로
등급별 **증분 예산의 85%**만 사용합니다.

여기서 `score`는 라우터가 새로 만드는 정답 점수가 아니라, 공식 outcome의
문항·모델별 `score` 필드를 뜻합니다. 공식 채점기는 예산을 통과한 경우 선택한
모델들의 이 값을 합산해 문항 수로 나눈 값을 등급 품질 점수로 사용합니다.

## 특징과 회귀기

- 의미 특징: Train에서 학습한 TF-IDF 단어 1-gram·2-gram 8,000개를
  TruncatedSVD로 줄인 64차원 LSA 잠재 표현
- 수치 특징: 문자·UTF-8 byte·단어·문장·행·메시지 수, 숫자, 문장부호,
  기호, 수학·코드 비율, 한글·영문·CJK 비율, 질문·지시문, task·난이도 등 46개
- 최종 특징: 110차원
- 예측기: 각 모델별 Ridge 회귀기 세 종류(점수, log1p 입력 토큰,
  log1p 출력 토큰), `alpha=10`
- 고정 seed: `20260819`

런타임에서는 학습된 `StandardScaler`, 64차원 SVD와 9개 Ridge 선형변환의
행렬곱을 artifact에 미리 저장해 동일한 예측을 더 적은 곱셈으로 계산합니다.
프롬프트별 예측만 공식 2코어에 맞춰 두 프로세스로 나누고, 비용 계산과
Lagrangian 선택은 기존과 같이 부모 프로세스에서 한 번 수행합니다. 이 최적화는
특징 정의, 회귀 가중치, 예산 또는 모델 선택 규칙을 변경하지 않습니다.

비용은 공식 v1 정책 그대로 계산합니다.

| 모델 | 입력 1M 토큰 | 출력 1M 토큰 |
| --- | ---: | ---: |
| `ax31-light` | 1 | 4 |
| `ax31` | 2.127 | 8.509 |
| `axk1-think` | 6.565 | 26.260 |

모델별 비용 예측에는 Train 5-fold OOF에서 얻은 보수적 보정계수를 적용합니다.
공식 Dev outcome은 학습, 특징 선택, 회귀기 적합 또는 보정계수 계산에 쓰지
않았습니다.

## 동결 artifact 출처와 무결성

`src/ossp_router/resources/semantic-lagrangian.v1.json`은 외부 AI 모델이나
사전이 아닙니다. 이 저장소의 공식 Train 1,760문항에서
`tools/train_semantic_lagrangian.py`로 직접 생성한 JSON입니다.

| 항목 | SHA-256 |
| --- | --- |
| materialized Train 입력 | `029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4` |
| Train outcome | `0a35c1ce83e074ffc8e470d5c4f49d35765371384ecff3db91bad9de4ef2ffe7` |
| raw 비용 정책 파일 | `07f1131caf1e6924b0516e9d24270d7faea8837dcc0231053e6f02ad66212fae` |
| 최적화 전 학습 artifact | `98b7be45f3e81c08b812b839023b1547a16b4ffc8598af63aaa9933bf1e6a052` |
| 제출용 runtime artifact | `ca50aacbfd3f0fc32e82f33ce7f63db581f0c34febdc9b1df7c5af2cabd98d0e` |

artifact 크기는 13,377,037 bytes입니다. 원천 데이터의 라이선스와 출처는
[`DATA_LICENSES.md`](../DATA_LICENSES.md)와
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)를 따릅니다.

제출용 runtime artifact는 학습 artifact에
`tools/compile_runtime_projection.py`의 결정적인 선형변환 결합 결과만 추가한
파일입니다. 학습 데이터, 특징, Ridge 계수와 비용 보정값은 변경하지 않습니다.

학습에만 NumPy 2.3.3, SciPy 1.16.2, scikit-learn 1.7.2를 사용하며 모두
BSD-3-Clause 계열 허용 라이선스입니다. 이 패키지들은 제출 이미지에 들어가지
않습니다. 실행 artifact와 직접 작성한 코드는 Apache-2.0으로 배포합니다.

## 검증된 Train·Dev 결과

공식 Train 1,760문항과 Dev 880문항을 각각 공식 self-check로 평가했습니다.

| split | 등급 | 품질 점수 | 실제 총비용 | 실제 비용 비율 | 공식 한도 | 통과 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Train | Fast | 0.658522727273 | 10.093855125 | 1.173187608920 | 1.25 | 예 |
| Train | Balanced | 0.684375000000 | 14.489372470 | 1.684069370159 | 2.0 | 예 |
| Train | Premium | 0.724715909091 | 25.435604008 | 2.956326901669 | 4.0 | 예 |
| Dev | Fast | 0.660511363636 | 5.376827014 | 1.227185980005 | 1.25 | 예 |
| Dev | Balanced | 0.689488636364 | 7.303652704 | 1.666957143653 | 2.0 | 예 |
| Dev | Premium | 0.727556818182 | 15.865380605 | 3.621052452534 | 4.0 | 예 |

최종 가중 점수는 Train `0.686136363636`, Dev `0.689318181818`입니다.
Windows 개발 환경의 순수 Python 실행 시간은 Train에서 등급별 12.6~12.8초,
Dev에서 6.4~6.8초였고 Dev 출력은 등급별 약 67 KiB였습니다. 이는 공식
`linux/arm64` 컨테이너 측정값이 아니므로 최종 push 전 Docker 검사가 별도로
필요합니다. Train 5-fold OOF의 MAE·RMSE·R2 전체 값은
[`verified-public-summary.json`](../reports/verified-public-summary.json)에
보존했습니다.
