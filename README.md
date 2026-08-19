<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic Lagrangian Router

SK Telecom **Efficient LLM Routing Challenge**를 위한 예산 제약형 프롬프트
라우터입니다. 프롬프트 내용만으로 `ax31-light`, `ax31`, `axk1-think` 중
하나를 선택하며, 최종 제출 정책은 **Semantic LSA + Ridge 회귀 +
Lagrangian MCKP** 한 가지로 고정했습니다.

> 공개 Dev 880문항 기준 최종 가중 점수 `0.689318181818`. Fast, Balanced,
> Premium 모두 실제 비용 한도를 통과했습니다.

## 한눈에 보기

| 항목      | 구현                                                  |
| ------- | --------------------------------------------------- |
| 문제 정의   | 프롬프트마다 모델 하나를 선택하는 Multiple-Choice Knapsack Problem |
| 의미 표현   | TF-IDF 1·2-gram 8,000개 → TruncatedSVD 64차원 LSA      |
| 수치 특징   | 길이·언어·수학·코드·질문·지시·난이도 특징 46개                        |
| 최종 특징   | 110차원                                               |
| 예측 모델   | 모델별 Ridge 회귀: 점수, 입력 토큰, 출력 토큰                      |
| 라우팅     | `예상 점수 - λ × 예상 비용` 최대화, λ 이분 탐색                    |
| 예산 안전장치 | 공식 등급별 증분 예산의 85% 사용                                |
| 학습 데이터  | 공식 Train 1,760문항만 사용                                |
| 실행 의존성  | Python 표준 라이브러리만 사용                                 |
| 실행 환경   | 네트워크·GPU 불필요, `linux/arm64` 컨테이너                    |
| 고정 seed | `20260819`                                          |

평가 측이 확인할 파일은 다음과 같습니다.

- 제출 진입점: [`semantic_lagrangian.py`](src/ossp_router/semantic_lagrangian.py)
- 동결 학습 artifact:
  [`semantic-lagrangian.v1.json`](src/ossp_router/resources/semantic-lagrangian.v1.json)
- 학습 코드: [`train_semantic_lagrangian.py`](tools/train_semantic_lagrangian.py)
- 결과와 회귀 지표: [`verified-public-summary.json`](reports/verified-public-summary.json)
- 상세 설계·provenance: [`SEMANTIC_LAGRANGIAN.md`](docs/SEMANTIC_LAGRANGIAN.md)
- 기술 제출 정보: [`submission-ossp-skt.json`](submission-ossp-skt.json)

## 동작 원리

라우터는 모델을 호출하지 않습니다. 평가 컨테이너는 프롬프트만 받아 각
문항의 `model_id`를 출력합니다. 평가자는 컨테이너 밖에서 이 결정과 비공개
outcome을 결합하여 실제 점수와 비용을 계산합니다.

```text
prompt 또는 messages
        ↓
110차원 프롬프트 특징
        ↓
모델별 예상 점수·토큰·비용
        ↓
등급별 Lagrangian MCKP
        ↓
episode_id + model_id
```

컨테이너는 공식 인터페이스로 등급 하나를 처리합니다.

```console
router-run \
  --input /challenge/input/inputs.json \
  --tier fast \
  --output /challenge/output/submission.json
```

같은 이미지가 `fast`, `balanced`, `premium`에 각각 실행됩니다. 입력의
`episode_id`, 입력 순서, `split`은 모델 선택 특징으로 사용하지 않습니다.
출력은 모든 문항을 정확히 한 번 포함하는 공식 v1 JSON입니다.

## 프롬프트 특징은 어떻게 만들어지는가

이 구현의 `semantic embedding`은 외부 API나 사전학습 Transformer 모델을
사용하지 않습니다. 공식 Train 프롬프트에서 직접 학습한 **LSA(Latent
Semantic Analysis)** 표현입니다. 따라서 평가 시 인터넷이나 GPU가 필요하지
않으며, Train에서 동결한 어휘·IDF·SVD 행렬을 그대로 사용합니다.

### 1. 입력 텍스트 구성

`prompt` 형식이면 해당 문자열을 그대로 사용합니다. `messages` 형식이면 각
메시지를 `<role>\ncontent` 형태로 입력 순서대로 연결합니다. `episode_id`,
`split`, 정답 및 outcome은 특징에 넣지 않습니다.

### 2. 64차원 의미 임베딩

Train 텍스트를 소문자 단어로 나눈 뒤 한 단어(unigram)와 연속한 두 단어
(bigram)를 만듭니다. 두 문항 이상에 나온 표현 중 최대 8,000개를 남겨
TF-IDF 벡터로 바꿉니다. 자주 반복되는 단어의 영향은 로그로 완화하고
벡터의 길이는 L2 정규화합니다.

TF-IDF는 단어가 정확히 일치하는지만 보여 주는 고차원 희소 벡터입니다.
여기에 TruncatedSVD를 적용해 함께 등장하는 표현의 패턴을 64개 축으로
압축합니다. 이 64개 값이 LSA 의미 임베딩입니다. 예를 들어 코드 관련
표현들이 자주 함께 등장하면 하나 이상의 잠재 축에서 비슷한 값을 갖게
되어, 단순 길이만으로는 구분하기 어려운 프롬프트 종류를 나타낼 수 있습니다.

```text
Train prompt
  → unigram + bigram
  → TF-IDF 최대 8,000차원
  → TruncatedSVD
  → LSA 64차원
```

### 3. 46차원 구조·난이도 특징

의미 임베딩과 별도로 문자·UTF-8 byte·단어·문장·행·메시지 수, 숫자와
문장부호 비율, 수학·코드 기호 및 키워드, 한글·영문·CJK 비율, 질문·지시문
형태, task·난이도 단서를 46개 숫자로 계산합니다. 이 값은 프롬프트의 주제뿐
아니라 길이와 답변 난이도에 따른 토큰 사용량 차이를 예측하는 데 쓰입니다.

64차원 LSA와 46차원 수치 특징을 이어 붙인 110차원 벡터를 Train 평균과
표준편차로 표준화합니다. 어휘, IDF, SVD 행렬, 평균과 표준편차는 모두 동결
artifact에 저장되므로 학습과 평가가 같은 변환을 사용합니다.

### 4. 특징에서 모델 선택까지

110차원 특징 하나를 세 공식 모델 각각의 Ridge 회귀기에 넣습니다. 모델마다
품질 점수, 입력 토큰, 출력 토큰을 따로 예측하므로 총 9개의 회귀 head가
있습니다. 토큰 회귀는 큰 값의 영향을 줄이기 위해 `log1p(token)`을 학습한
뒤 예측 시 원래 단위로 되돌립니다.

```text
110차원 특징
  ├─ 모델별 score 회귀        → r_hat_im
  ├─ 모델별 input-token 회귀  → P_hat_im
  └─ 모델별 output-token 회귀 → O_hat_im

c_hat_im = alpha_m × P_hat_im + beta_m × O_hat_im
```

`alpha_m`, `beta_m`는 아래 표의 공식 비용 계수입니다. 이렇게 얻은 문항별
예상 점수와 예상 비용 전체를 Lagrangian 라우터가 한꺼번에 받아, 해당 등급의
총예산 안에서 최종 `model_id`를 정합니다. 즉 semantic embedding 자체가
모델을 바로 고르는 것이 아니라, **회귀 예측의 입력 특징**으로 사용됩니다.

## 알고리즘

문항 `i`, 모델 `m`의 선택 변수를 `z_im`으로 두면 목표는 다음과 같습니다.

```text
maximize   sum_i sum_m z_im * r_hat_im
subject to sum_i sum_m z_im * c_hat_im <= B
           sum_m z_im = 1
           z_im in {0, 1}
```

`r_hat_im`은 예상 품질 점수, `c_hat_im`은 예상 비용입니다. 여기서 점수는
라우터가 임의로 만든 값이 아니라 공식 outcome의 문항·모델별 `score`를
예측한 값입니다.

비용 제약에 가격 `λ`를 부여하면 문항별 선택은 다음처럼 분리됩니다.

```text
argmax_m (r_hat_im - λ * c_hat_im)
```

100회 이분 탐색으로 예산을 만족하는 `λ`를 찾습니다. 실제 평가에서는
문항별 토큰 수를 알 수 없으므로 모델별 입력·출력 토큰을 회귀로 예측하고,
Train 5-fold OOF에서 얻은 보수적 비용 보정계수를 적용합니다.

## 공식 비용과 예산

비용은 공식 정책 `ossp-2026-prompt-router-v1`을 그대로 사용합니다.

| 모델 | 입력 1M 토큰 | 출력 1M 토큰 |
| --- | ---: | ---: |
| `ax31-light` | 1 | 4 |
| `ax31` | 2.127 | 8.509 |
| `axk1-think` | 6.565 | 26.260 |

| 등급 | All-Light 대비 최대 비용 | 최종 점수 가중치 |
| --- | ---: | ---: |
| Fast | 1.25 | 0.4 |
| Balanced | 2.0 | 0.3 |
| Premium | 4.0 | 0.3 |

한도를 조금이라도 넘으면 해당 등급의 점수는 0입니다. 비용 예측 오차를
흡수하기 위해 공식 feature-budget baseline과 동일하게 증분 예산의 85%만
라우팅에 사용합니다.

## 공개 Train·Dev 결과

공식 `self-check`로 선택 결과와 실제 outcome을 결합해 검증했습니다. Dev는
학습, 특징 선택, 회귀 적합 또는 비용 보정에 사용하지 않았습니다.

| split | 등급 | 품질 점수 | 실제 총비용 | 비용 비율 | 공식 한도 | 통과 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Train | Fast | 0.658522727273 | 10.093855125 | 1.173187608920 | 1.25 | ✓ |
| Train | Balanced | 0.684375000000 | 14.489372470 | 1.684069370159 | 2.0 | ✓ |
| Train | Premium | 0.724715909091 | 25.435604008 | 2.956326901669 | 4.0 | ✓ |
| Dev | Fast | 0.660511363636 | 5.376827014 | 1.227185980005 | 1.25 | ✓ |
| Dev | Balanced | 0.689488636364 | 7.303652704 | 1.666957143653 | 2.0 | ✓ |
| Dev | Premium | 0.727556818182 | 15.865380605 | 3.621052452534 | 4.0 | ✓ |

최종 가중 점수는 Train `0.686136363636`, Dev `0.689318181818`입니다. 전체
MAE·RMSE·R²와 모델 선택 수는
[`verified-public-summary.json`](reports/verified-public-summary.json)에
기록했습니다.

검증 범위는 다음과 같습니다.

- 공식 Train·Dev split 유지 및 Train-only 학습
- 공개 Train·Dev 공식 `self-check` 예산 통과
- 동일 입력 반복 실행 시 바이트 단위 결정성
- 문항 ID 변경 및 입력 순서 변경 감사 통과
- `linux/arm64` 이미지 빌드 및 toy 입력 컨테이너 smoke test 통과
- GHCR 이미지의 비로그인 ARM64 pull 확인

공개 데이터 결과는 비공개 평가셋의 점수나 예산 통과를 보장하지 않습니다.

## 빠른 실행

소스에서 toy 입력 한 등급을 실행하려면 추가 패키지가 필요하지 않습니다.

```console
PYTHONPATH=src python3 -m ossp_router.semantic_lagrangian \
  --input data/toy/inputs.json \
  --tier fast \
  --output build/toy-fast.json
```

핵심 결정성·감사 테스트를 실행합니다.

```console
PYTHONPATH=src python3 -m unittest tests.test_semantic_lagrangian
```

공개 Dev 세 등급을 실행하고 채점합니다.

```console
for tier in fast balanced premium; do
  PYTHONPATH=src python3 -m ossp_router.semantic_lagrangian \
    --input data/materialized/dev/inputs.json \
    --tier "$tier" \
    --output "build/semantic-dev-submissions/$tier.json"
done

PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --submissions build/semantic-dev-submissions \
  --policy configs/routing-policy.v1.json \
  --report build/semantic-lagrangian-dev-report.json
```

공개 Train·Dev 입력 생성 방법은 [공식 데이터 안내](data/README.md)를
따릅니다.

## 학습 artifact 재현

학습용 NumPy, SciPy, scikit-learn은 제출 이미지에 포함되지 않습니다.

```console
python3 -m venv .venv-train
.venv-train/bin/pip install -r baselines/requirements-semantic-train.txt

PYTHONPATH=src .venv-train/bin/python tools/train_semantic_lagrangian.py \
  --input data/materialized/train/inputs.json \
  --outcomes data/train/outcomes.json \
  --policy configs/routing-policy.v1.json \
  --artifact src/ossp_router/resources/semantic-lagrangian.v1.json \
  --report build/semantic-lagrangian-train-report.json
```

동결 artifact SHA-256은
`98b7be45f3e81c08b812b839023b1547a16b4ffc8598af63aaa9933bf1e6a052`입니다.
원천 입력·outcome·정책 해시와 라이선스는
[`SEMANTIC_LAGRANGIAN.md`](docs/SEMANTIC_LAGRANGIAN.md)에 기록했습니다.

## 저장소 구조

```text
src/ossp_router/semantic_lagrangian.py      제출 라우터와 CLI
src/ossp_router/resources/                  공식 정책과 동결 artifact
tools/train_semantic_lagrangian.py          Train-only 학습 파이프라인
tools/create_technical_submission.py        기술 제출 JSON 생성기
reports/verified-public-summary.json        회귀·라우팅 검증 결과
docs/SEMANTIC_LAGRANGIAN.md                 설계와 provenance
container/Dockerfile                        linux/arm64 제출 이미지
submission-ossp-skt.json                    고정 코드·이미지 식별자
```

## 공식 규격과 라이선스

이 구현은 원본 저장소의 동결된 정책과 프로토콜을 변경하지 않습니다.

- [과제 규칙](docs/CHALLENGE_RULES.md)
- [점수 계산](docs/SCORING.md)
- [컨테이너 실행 규격](docs/RUNTIME.md)
- [제출 안내](docs/SUBMISSION.md)
- [데이터 및 제3자 고지](THIRD_PARTY_NOTICES.md)

프로젝트 코드와 문서는 [Apache License 2.0](LICENSE)으로 배포합니다.
데이터별 라이선스는 [DATA_LICENSES.md](DATA_LICENSES.md)를 따릅니다.
