# 데이터 수집 파이프라인 — FFW-SG2 VR 원격조작

**작성자:** Hun Kim (hun7728@hanyang.ac.kr) · **수정일:** 2026-07-20

NVIDIA Isaac Sim에서 양팔 매니퓰레이션 **+ 모바일 베이스** 시연을 수집하고, ROBOTIS
IsaacLab-Mimic 증강 파이프라인을 돌린 뒤, 모방학습용 **LeRobot** 포맷으로 내보내는 절차.

**이 문서의 범위.** 베이스 시스템(컨테이너, VR 스택, 녹화, Mimic, LeRobot 변환기, 대시보드)은
기존 레포지토리의 기능이라 여기서는 링크와 함께 **간단히만** 요약한다. 상세한 설명은 **이 fork가
추가/수정한 것** — 주행 모바일 베이스, 물리 주행 22차원 datagen, 실물 카메라/베이스 속도 정합 — 에
집중한다. fork 작업은 **[추가]** / **[수정]** 으로 표시하고 [부록 A](#부록-a--추가--수정-파일)에
정리했다.

---

## 0. 출처 & 활용 레포지토리

**이 작업은 새로 구현한 것이 아니라 기존 레포지토리를 최대한 활용했다.** 거의 모든 파이프라인 —
컨테이너, VR 퍼블리셔/컨트롤러, Isaac Sim 녹화 루프, IsaacLab-Mimic 증강, LeRobot 변환기, 대시보드
— 은 아래 베이스 레포에서 온 것이다. 이 문서의 절차는 대부분 **기존 소스에서 파생**된 것이지
처음부터 작성한 것이 아니다.

| 출처 | 링크 | 제공하는 것 |
|---|---|---|
| **EKAIWORKER** (베이스 레포) | https://github.com/Disniekie01/EKAIWORKER | 이 fork가 올라탄 전체 스택: 컨테이너 3개, VR 퍼블리셔/컨트롤러(`robotis_vuer`, `ai_worker`), Isaac Sim 녹화 파이프라인, IsaacLab-Mimic datagen, `isaaclab2lerobot` 변환기, `sg2_ltable_dashboard.py`. ROBOTIS 업스트림 3개(`cyclo_lab`, `ai_worker`, `robotis_applications`)를 고정 커밋으로 핀. |
| **adb_vr_connect** | https://github.com/Disniekie01/EKAIWORKER/tree/main/adb_vr_connect | Meta Quest의 1회성 ADB/udev 설정과 USB 테더 연결 절차. |
| **ROBOTIS VR 원격조작 가이드** | https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation/ | 공식 VR 조작 절차(헤드셋 페어링, 브라우저 흐름, 그립 3초 활성화). |

**그대로 사용(문서화만, 직접 작성 아님):** 컨테이너 실행(`setup.sh`, `docker/`), VR 연결
절차(`adb_vr_connect`), 기본 VR 퍼블리셔/컨트롤러/IK 체인, 녹화 파이프라인(`record_demos.py`,
recorder manager), IsaacLab-Mimic 드라이버 스크립트, stock L-테이블 태스크/USD.

**이 레포.** `AI_HUN`(https://github.com/hun7407-lgtm/AI_HUN)은 **EKAIWORKER** 위에 얹은
오버레이이며, `overlays/` 아래의 모든 것은 `setup.sh`가 핀된 업스트림 위에 적용한다. 팀 레포 사본은
`nimbuslab1/pai-korea_sprint2026`의 `Initial_collecion/HunKim`에 있다.

---

## 1. 이 fork가 추가한 것

아래는 전부 fork의 고유 기여이며, 각 상세 섹션으로 연결된다.

| 기여 | 태그 | 섹션 |
|---|---|---|
| **주행 모바일 베이스**(`FFW_SG2_MOBILE`) — root teleport 대신 물리로 주행하는 스워브 베이스 | [추가] | [§4](#4-주행-모바일-베이스-추가) |
| **물리 주행 22차원 datagen** — Mimic 증강이 베이스를 물리로 주행; 22차원이 전 단계 유지 | [추가] | [§5](#5-모바일-22차원-datagen-추가) |
| **4카메라 실물 정합** — 헤드 스테레오 + 손목 2개, CHW, `ffw_sg2_rev1`과 일치 | [수정] | [§6.1](#61-카메라-수정) |
| **베이스 속도 관측** — `[linear_x, linear_y, angular_z]`, state/action 19 → 22 | [추가] | [§6.2](#62-베이스-속도-추가) |
| **LeRobot 변환기** — 카메라 + 베이스 속도 자동 감지, 실물 스키마로 내보냄 | [수정] | [§6.3](#63-변환기-수정) |
| **VR 베이스 주행** — A/B 회전, Y버튼 모드 토글, `/cmd_vel` RELIABLE QoS | [수정] | [§7](#7-vr-베이스-주행-조작-수정) |
| **세션 기반 데이터셋 이름** + 대시보드에 모바일 태스크 | [수정] | [부록 A](#부록-a--추가--수정-파일) |

---

## 2. 베이스 시스템 (요약 — 링크 참고)

로봇, 태스크, 컨테이너, 셋업, VR 연결, 표준 파이프라인 단계는 모두 베이스 레포 기능이다. 맥락을
위한 간단한 사실만 적고, 전체 절차는 링크를 따르라.

- **로봇** — ROBOTIS **FFW-SG2**, 양팔 모바일 매니퓰레이터. 관절/액션은 **19차원**
  (`arm_l 7, gripper_l 1, arm_r 7, gripper_r 1, head 2, lift 1`); 모바일 태스크는 베이스
  `linear_x, linear_y, angular_z`를 더해 **22차원**. 베이스 = 스워브 모듈 3개(좌/우/후).
- **태스크** — L-테이블 pick & place: 앞 테이블의 박스를 양팔로 집고, 베이스를 이동해 왼쪽("L")
  테이블에 놓기.
- **컨테이너** — ROS 2(`ROS_DOMAIN_ID=30`, Fast DDS) 위 Docker 3개: `cyclo_lab`(Isaac Sim +
  recorder + Mimic + 대시보드:8765), `robotis-applications`(Vuer VR 퍼블리셔:8012),
  `ai_worker`(팔 IK 컨트롤러). 버전: Isaac Sim 5.1.0, Isaac Lab 2.3.0.
- **셋업** — `git clone https://github.com/hun7407-lgtm/AI_HUN.git AIWORKER && cd AIWORKER &&
  ./setup.sh ~/AIWORKER`. 핀된 업스트림을 클론하고 `overlays/`를 그 위에 rsync한다. 사전
  준비(`setup.sh`가 설치 안 함): Linux + X11, NVIDIA RTX GPU + 드라이버, Docker + NVIDIA
  Container Toolkit, NGC 로그인, Meta Quest 3. EKAIWORKER README 참고.
- **VR 연결** — `adb_vr_connect` USB 테더(권장) 또는 WiFi. 전체 절차:
  [adb_vr_connect](https://github.com/Disniekie01/EKAIWORKER/tree/main/adb_vr_connect),
  [EKAIWORKER README](https://github.com/Disniekie01/EKAIWORKER),
  [ROBOTIS VR 가이드](https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation/).
- **녹화** — `sg2_ltable_dashboard.py`에서 실행; 양쪽 컨트롤러 그립 ~3초로 활성화, 조작 후 `N`
  저장 / `R` 폐기. 출력: `datasets/…_raw.hdf5`.
- **Mimic 파이프라인(베이스)** — 5단계, 대시보드나 수동으로 하나씩 실행:
  `IK 변환 → annotate → datagen → joint 변환 → LeRobot 내보내기`
  (`action_data_converter.py`, `annotate_demos.py`, `generate_dataset.py`, `isaaclab2lerobot.py`).
  모바일 22차원 경로용 fork 변경은 §5.

```
[Meta Quest] ─ pose + 조이스틱 ─► [robotis-applications] Vuer 퍼블리셔
    ├─ 팔/손목 pose ─► [ai_worker] IK ─► 관절 명령 ─┐
    └─ /cmd_vel (Twist) ──────────────────────────┤
                                                   ▼
[cyclo_lab] Isaac Sim + record_demos.py + FFWSG2Sdk ─► raw.hdf5
    └─ IK 변환 ─► annotate ─► Mimic datagen ─► joint 변환 ─► LeRobot
                                                   ▼
                      LeRobot 데이터셋 (카메라 4개 + 22차원 state/action)
```

---

## 3. stock 태스크의 모바일 한계 (이 fork가 존재하는 이유)

stock(베이스 레포) 형태에서는 베이스가 **로봇 root를 키네마틱하게 텔레포트하는 스크립트 L-모션**으로
이동한다. 두 가지 문제:
- 텔레포트가 물리와 충돌 — root가 텔레포트되는 동안 들고 있는 박스는 여전히 물리로 움직여서
  그리퍼 대비 박스가 **흔들린다(jitter)**.
- 이 키네마틱 텔레포트 모션이 **데이터 생성에서 깨끗하게 이어지지 않아**, 모바일 베이스 궤적을
  안정적으로 증강할 수 없었다.

이 문서의 나머지가 fork의 해결책이다: 물리 주행 베이스(§4)를 녹화·datagen(§5) 양쪽에 쓰고, 실물
정합 작업(§6)을 더했다.

---

## 4. 주행 모바일 베이스 **[추가]**

**물리로 주행하는** 베이스 USD(`FFW_SG2_MOBILE`) — root 텔레포트 없이 스워브 바퀴를 물리로 굴려
이동하므로 베이스와 들고 있는 박스가 물리적으로 일관되게 유지된다.

### 4.1 USD에서 바뀐 것

stock `FFW_SG2.usd`는 정지 매니퓰레이션용이다. `FFW_SG2_MOBILE`은 stock USD를 참조하는 `~2 KB`
오버라이드 레이어에서 그 잠금을 푼다(stock 원본은 절대 수정 안 함):

| stock 잠금 | 수정 |
|---|---|
| `FixedJoint`가 섀시를 월드에 용접 | `fix_root_link=False` (자유 베이스) |
| 휠 drive 한계 ±1080° | 제거 (연속 회전) |
| 좌/우 휠 콜라이더 꺼짐 | 다시 켬 |
| 중력 꺼짐 | **바디별**: 베이스+휠 6개 ON(접지력), 팔/리프트/헤드/그리퍼 OFF(처짐 방지) |
| — | 자기충돌 **ON**(팔이 몸통 관통 방지); 휠 6개는 모든 몸체 링크에 필터링(휠은 바닥만 충돌) |
| — | reset 이벤트가 `reset_scene_to_default` 후 베이스를 정상 높이로 올림 |

### 4.2 주행 방식

스워브 컨트롤러가 바디 프레임 `cmd_vel` `[linear_x, linear_y, angular_z]`을 모듈별 조향각 + 휠
속도로 변환한다. 녹화 중 작업자가 `/cmd_vel`로 몰면 SDK가 휠 타겟으로 적용
(`_apply_swerve_cmd_vel(..., integrate_root=False)` → 물리 주행). 검증: `root_z ≈ 1.405`로 안정,
10초에 8.23 m 주행(지령 속도의 96%), 홀로노믹 게걸음/제자리회전 확인.

도구: `scripts/tools/build_ffw_sg2_mobile_usd.py`(USD 재생성),
`check_ffw_sg2_mobile.py`(6/6 회귀), `teleop_sg2_mobile.py`(키보드 주행).

---

## 5. 모바일 22차원 datagen **[추가]**

Mimic 파이프라인 전체가 **모바일 22차원** 데이터로 처음부터 끝까지 돌며, 증강 중 베이스를
**물리로 주행**(텔레포트 아님)한다.

### 5.1 두 가지 enabler

**(a) 모바일 Mimic 태스크.** `Cyclo-Real-Mimic-Pick-Place-LTable-Mobile-FFW-SG2-v0`
(`FFWSG2PickPlaceLTableMobileMimicEnvCfg`) — 같은 L-테이블 datagen을 주행 베이스 env 위에 얹어서
생성 데모가 `base_velocity` 관측을 유지.

**(b) 물리 주행 재생(텔레포트 없음).** 이동 서브태스크에서 베이스를 녹화 pose로 텔레포트하는 대신,
mimic env가 **녹화된 베이스 속도를 스워브 `cmd_vel`로 재생**해 물리 주행 — 바퀴가 돌고
teleport/physics 떨림 없음. `pick_place_l_table_mimic_env.py`(`_physics_drive_step`, 배치 스워브)에
구현, 모바일 cfg에서만 켜져 고정 베이스 태스크는 그대로.

### 5.2 전 단계 22차원

베이스 속도는 IK 변환부터 액션의 마지막 3채널로 실려간다(모바일 한정; 고정 베이스는 19차원 유지):

| 단계 | 22차원 생성 방식 |
|---|---|
| IK / Joint 변환 | `action_data_converter.py`가 `obs/base_velocity`가 있으면 액션에 붙임 |
| Annotate | sim 액션 매니저가 19차원이라 env엔 19차원을 먹이고, 에피소드마다 export 액션을 22차원으로 재조립(`annotate_demos.py`) |
| Datagen | `generate_dataset.py`가 sim 앱 종료 후 생성 파일을 22차원으로 후처리 |
| LeRobot | `isaaclab2lerobot.py`가 이미 22차원인 액션을 그대로 수용(이중 append 방지), state는 22차원으로 구성 |

### 5.3 실행

```bash
# Datagen — 모바일 태스크, 물리 주행 베이스. --generation_num_trials = 채울 성공 개수
# (generation_guarantee=True: 실패 시도는 재시도, 카운트 안 함).
python scripts/sim2real/imitation_learning/mimic/generate_dataset.py \
  --device cuda --num_envs 4 --task Cyclo-Real-Mimic-Pick-Place-LTable-Mobile-FFW-SG2-v0 \
  --generation_num_trials 20 --input_file ./datasets/<...>_annotate.hdf5 \
  --output_file ./datasets/<...>_generate.hdf5 --enable_cameras --headless
```
IK 변환 / annotate / joint 변환 / LeRobot은 베이스 파이프라인(§2)과 같은 명령을 모바일 태스크 ID로만
쓰면 되고, 모바일 데이터에서 자동으로 22차원을 만든다.

**실무 참고**
- **env 수는 카메라 렌더 메모리에 묶인다(연산 아님).** env당 RGB-D 카메라 4개 + 생성 관측이 데모
  export 전까지 GPU에 누적(톱니 패턴). 24 GB GPU에서 `--num_envs 10`은 **OOM**, `--num_envs 4`는
  피크 ~20 GB로 안전.
- **물리 주행 성공률은 teleport보다 낮다.** open-loop 속도 재생이 드리프트해 박스를 가끔 잘못 놓고
  그 시도는 `success=False`(다운스트림 제거). 관측치 ~40 %; `generation_guarantee=True`가 요청한
  *성공* 개수를 채울 때까지 재시도. joint 변환 필터에서 성공만 살아남음.
- **목표까지 이어붙이기.** 끊기면 나머지를 두 번째 파일로 생성 후 병합:
  `python merge_hdf5_demos.py --inputs partA.hdf5 partB.hdf5 --output generate.hdf5`.
- **0 근처 속도의 조향 떨림은 sim 전용 현상.** 정책은 `cmd_vel`(베이스 3채널)을 배우지 휠 명령을
  배우지 않는다; 실물 로봇의 자체 스워브 컨트롤러(자체 저속 처리 포함)가 바퀴를 굴린다. 베이스
  궤적과 `base_velocity` 라벨은 매끄럽다.

---

## 6. 실물 정합

실물 `ffw_sg2_rev1`은 RGB 카메라 **4개**와 **22차원** state를 기록한다; stock sim은 1개·19차원.
두 가지 추가 + 변환기 변경으로 실물 스키마에 맞춘다.

### 6.1 카메라 **[수정]**

| LeRobot 키 | sim 카메라 | 마운트 | 해상도 |
|---|---|---|---|
| `…rgb.cam_left_head` | `cam_head` (ZED 좌안) | `head_link2/zed` | 376 × 672 |
| `…rgb.cam_right_head` | `cam_right_head` | `head_link2/zed`, 미러 | 376 × 672 |
| `…rgb.cam_left_wrist` | `cam_left_wrist` | `arm_l_link7` (D405 pose) | 424 × 240 |
| `…rgb.cam_right_wrist` | `cam_right_wrist` | `arm_r_link7` (D405 pose) | 424 × 240 |

- 헤드 카메라는 헤드 ZED의 두 눈으로 `zed` prim 중심 대칭(Y ±0.03 m ⇒ ~0.06 m 베이스라인). 손목
  카메라는 USD의 RealSense **D405** pose 사용(`arm_*_link7/visuals/d405`): 로컬
  `pos (0.10683, 0, -0.07713)`, Y축 180°.
- 카메라는 로봇 링크의 자식이라 렌더/녹화 피드가 헤드/팔 움직임을 따라간다.
- **외부 파라미터는 보정 placeholder** — 교차 도메인 학습 전 실물 대조 필요.

### 6.2 베이스 속도 **[추가]**

모바일 태스크가 `obs/base_velocity = base_planar_velocity(env)` =
`[root_lin_vel_b.x, root_lin_vel_b.y, root_ang_vel_b.z]`(베이스 프레임)을 기록. state와 action에
붙여 22차원 → 실물 일치.

### 6.3 변환기 **[수정]**

`isaaclab2lerobot.py`가 녹화 내용을 자동 감지해 실물 스키마로 내보냄 — 플래그 불필요:

```bash
lerobot-python scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py \
  --task Cyclo-Real-Pick-Place-LTable-Mobile-FFW-SG2-v0 --robot_type FFW_SG2 --fps 15 \
  --dataset_file ./datasets/<...>_joint.hdf5
```
- 카메라 스트림 감지 → 각각 `observation.images.rgb.<name>`, **채널 우선 `[3, H, W]`**로 내보냄
  (`cam_head` → `cam_left_head`).
- `obs/base_velocity` 붙여 **22차원**(없으면 19); 액션이 이미 22차원이면(모바일 datagen이 패킹,
  §5.2) 그대로 사용(이중 append 방지).

### 6.4 출력 스키마 (실물 `ffw_sg2_rev1`과 일치)

| Feature | Shape |
|---|---|
| `observation.state` | (22,) — 19 관절 + `linear_x, linear_y, angular_z` |
| `action` | (22,) — 동일 |
| `observation.images.rgb.cam_left_head` / `cam_right_head` | [3, 376, 672] |
| `observation.images.rgb.cam_left_wrist` / `cam_right_wrist` | [3, 424, 240] |

19 관절 순서: `arm_l_joint1..7, gripper_l_joint1, arm_r_joint1..7, gripper_r_joint1, head_joint1,
head_joint2, lift_joint`.

---

## 7. VR 베이스 주행 조작 **[수정]**

녹화 중 베이스를 몰기 위해(Plan B) VR 퍼블리셔를 확장했다(베이스 조작만; 팔/그리퍼/리프트 조작은
베이스 레포 그대로):

| 입력 | 동작 |
|---|---|
| 왼쪽 썸스틱 | 베이스 이동(전후 = x, 좌우 = y) |
| **A 버튼(홀드)** | 우회전 |
| **B 버튼(홀드)** | 좌회전 |
| **Y 버튼** | `LIFT+HEAD` ↔ `LIFT+CMD_VEL` 베이스 주행 모드 토글 |

핵심 수정: `/cmd_vel`을 **RELIABLE** QoS로 발행(기존 BEST_EFFORT); SDK가 RELIABLE로 구독하므로
DDS 불일치가 모든 베이스 명령을 조용히 버리고 있었다. 배포/테스트 절차는
[`DEPLOY_PLAN_B.md`](DEPLOY_PLAN_B.md).

---

## 8. 알려진 제약

- **모바일 22차원 datagen은 고정 베이스 teleport보다 성공률이 낮다**(~40 %). 물리 주행 베이스가
  open-loop라 드리프트한 시도가 박스를 잘못 놓아 제거됨; `generation_guarantee=True`가 요청 성공
  개수를 채울 때까지 재시도하므로 실행 시간이 그만큼 길어짐.
- **datagen env 수는 연산이 아니라 카메라 렌더 메모리에 제한** — 24 GB GPU에선 작게 유지(4 안전,
  10 OOM). §5.3 참고.
- **카메라 외부 파라미터는 placeholder** — 실물과 대조해 보정 필요.
- **카메라 4개 렌더는 GPU 부하가 크다.** 검증 워크스테이션(**NVIDIA RTX PRO 5000 Blackwell Laptop,
  24 GB**)에선 일반 사용 시 프레임드랍 없음; 저사양에선 카메라 4개가 GPU를 포화시켜 teleop을 느리게
  할 수 있음. 프레임드랍은 조작감에만 영향, **녹화 데이터엔 영향 없음**(각 프레임은 여전히 해당
  시뮬 시점의 올바른 이미지). 필요하면 headless로 GUI 렌더 제거.
- **action의 베이스 속도는 측정 twist**를 명령 대용으로 사용(스워브가 `/cmd_vel`을 잘 추종); 0
  근처 속도의 약한 노이즈는 실물 스워브/관성이 흡수.

---

## 부록 A — 추가 / 수정 파일

모든 경로는 `overlays/`(버전관리 오버레이) 아래이며 `setup.sh` / `sync_overlay.sh`가 라이브
체크아웃에 적용한다. 그 외 레포의 모든 것은 베이스 레포 / 업스트림 코드로 미변경.

### 추가

| 파일 | 용도 |
|---|---|
| `cyclo_lab/…/assets/robots/FFW_SG2_MOBILE.py` | 주행 가능 베이스 아티큘레이션 config |
| `cyclo_lab/…/data/robots/FFW/FFW_SG2_MOBILE.usd` | stock USD 위 ~2 KB 오버라이드 레이어 |
| `cyclo_lab/…/controllers/swerve.py` (+ `__init__.py`) | 3모듈 홀로노믹 스워브 IK 컨트롤러 |
| `cyclo_lab/scripts/tools/build_ffw_sg2_mobile_usd.py` | `FFW_SG2_MOBILE.usd` 재생성 |
| `cyclo_lab/scripts/tools/check_ffw_sg2_mobile.py` | 주행 / 홀로노믹 회귀 검증 (6개) |
| `cyclo_lab/scripts/tools/teleop_sg2_mobile.py` | 키보드 베이스 주행 (개발용) |
| `DEPLOY_PLAN_B.md` | 모바일 녹화 배포 & 테스트 절차 |

### 수정

| 파일 | 변경 내용 |
|---|---|
| `cyclo_lab/…/pick_place_l_table/joint_pos_env_cfg.py` | 카메라 4개; `FFWSG2PickPlaceLTableMobileEnvCfg`; 베이스 속도 관측; reset 이벤트 |
| `cyclo_lab/…/pick_place_l_table/pick_place_env_cfg.py` | 카메라 씬 슬롯 + 관측항; teleop 플래그 |
| `cyclo_lab/…/pick_place_l_table/__init__.py` | mobile 태스크 **및 mobile Mimic 태스크** 등록 |
| `cyclo_lab/…/pick_place_l_table/pick_place_l_table_mimic_env_cfg.py` | `FFWSG2PickPlaceLTableMobileMimicEnvCfg`(모바일 22차원 datagen cfg) 추가 |
| `cyclo_lab/…/pick_place_l_table/mdp/observations.py` | `base_planar_velocity` 관측 |
| `cyclo_lab/…/pick_place_l_table/mdp/ffw_sg2_l_table_events.py` | `reset_mobile_base_standing` 이벤트 |
| `cyclo_lab/scripts/…/mimic/cyclo_mimic_datagen.py` | Datagen: SG2 head/lift 액션 레이아웃, 소스 에피소드 카메라 obs 제거, 박스 carry latch |
| `cyclo_lab/…/pick_place_l_table/ltable_kinematic_l_motion.py` | 양손 파지 판정 강화 (한손/부분 파지 거부) |
| `cyclo_lab/…/pick_place_l_table/pick_place_l_table_mimic_env.py` | 병렬(multi-env) datagen 스텝; **물리 주행 베이스 재생**(`_physics_drive_step`, 배치 스워브)으로 모바일 태스크 teleport 대체 |
| `cyclo_lab/scripts/…/mimic/action_data_converter.py` | IK / joint 변환이 모바일에서 **`obs/base_velocity` → 22차원 append**(그 외 19차원) |
| `cyclo_lab/scripts/…/mimic/annotate_demos.py` | env엔 19차원 입력; **export 액션을 `obs/base_velocity`로 22차원 재조립** |
| `cyclo_lab/scripts/…/mimic/generate_dataset.py` | sim 앱 종료 후 **생성 파일을 22차원 액션으로 후처리** |
| `cyclo_lab/scripts/…/data_converter/isaaclab2lerobot.py` | 카메라 자동 감지(rgb, CHW) + 베이스 속도 → 22차원; **이미 22차원인 액션 수용** |
| `cyclo_lab/scripts/…/dds_sdk/ffw_sg2_sdk.py` | `/cmd_vel` 구독, 물리 스워브 베이스 주행 |
| `cyclo_lab/sg2_ltable_dashboard.py` | 세션 기반 데이터셋 이름; 태스크 목록에 모바일 태스크 |
| `robotis_applications/robotis_vuer/robotis_vuer/vr_publisher_sg2.py` | Y버튼 모드 토글; A/B 베이스 회전; `/cmd_vel` → RELIABLE QoS |

---

*베이스 레포: [`EKAIWORKER`](https://github.com/Disniekie01/EKAIWORKER) · Fork:
[`AI_HUN`](https://github.com/hun7407-lgtm/AI_HUN). 업스트림: ROBOTIS `cyclo_lab`, `ai_worker`,
`robotis_applications` (`setup.sh`에 핀).*
