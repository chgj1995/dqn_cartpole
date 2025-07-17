# cartpole_ddqn.py
# ------------------------------------------------------------
# 이 스크립트는 OpenAI Gym의 CartPole-v1 환경에서
# Double Deep Q-Network(DDQN)를 이용해 막대가 넘어지지 않도록
# 균형을 잡는 에이전트를 학습합니다.
# 에피소드 당 Q값 평균을 계산해, 리워드 평균(avg10r)과
# Q값 평균(avg10q)을 10회 단위로 출력·그래프로 보여줍니다.
# ------------------------------------------------------------

import os, math, random, warnings       # 기본 모듈
from collections import deque           # 경험 리플레이 버퍼
import gym, torch, numpy as np          # RL 환경, 딥러닝, 수치계산
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt         # 학습 결과 시각화
import config as C                      # 공통 하이퍼파라미터·시드 불러오기

# === 불필요한 DeprecationWarning 숨기기 =========================
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === 하이퍼파라미터 ============================================
ENV_NAME          = C.ENV_NAME
GAMMA             = C.GAMMA
LEARNING_RATE     = C.LEARNING_RATE
BATCH_SIZE        = C.BATCH_SIZE
MEMORY_CAPACITY   = C.MEMORY_CAPACITY
EPS_START         = C.EPS_START
EPS_END           = C.EPS_END
EPS_DECAY         = C.EPS_DECAY
TARGET_UPDATE_EPS = C.TARGET_UPDATE_EPS
TOTAL_EPISODES    = C.TOTAL_EPISODES
SAVE_INTERVAL     = C.SAVE_INTERVAL
MODEL_DIR         = "model_dddqn"
os.makedirs(MODEL_DIR, exist_ok=True)   # 체크포인트 폴더 생성

# === 환경 및 디바이스 설정 =====================================
env       = gym.make(ENV_NAME)                              # CartPole 환경 생성
num_state = env.observation_space.shape[0]                  # 상태 벡터 차원(4)
num_act   = env.action_space.n                              # 행동 개수(2: 왼쪽/오른쪽)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Q-Network 정의 ============================================
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(num_state, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, num_act)

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(1, keepdim=True))

# 정책(policy) 네트워크와 타깃(target) 네트워크 두 개 생성
policy_net = QNetwork().to(device)
target_net = QNetwork().to(device)
target_net.load_state_dict(policy_net.state_dict())         # 초기 가중치 동기화
target_net.eval()                                           # 평가 모드로 전환

optimizer  = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_mem = deque(maxlen=MEMORY_CAPACITY)                  # 경험 저장소

# === 행동 선택 함수 (ε-greedy) ===================================
def select_action(state, steps_done):
    """ε-greedy 정책: 무작위 탐험 vs Q값 활용 균형"""
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    if random.random() < epsilon:                            # 탐험: 무작위 행동
        return env.action_space.sample()
    with torch.no_grad():                                    # 활용: Q값 최대 행동
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return policy_net(state_t).argmax(1).item()

# === 최적화 함수 ================================================
def optimize():
    """리플레이 버퍼에서 배치 샘플링 → Q-Network 한 스텝 학습
        반환값: 이 배치에서의 Q값 평균"""
    if len(replay_mem) < BATCH_SIZE:
        return None                                         # 학습할 데이터 부족 시

    # 1) 배치 샘플링
    batch = random.sample(replay_mem, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # 2) NumPy 배열 → Torch 텐서 (속도·경고 해결)
    states      = torch.as_tensor(np.array(states, dtype=np.float32), device=device)
    next_states = torch.as_tensor(np.array(next_states, dtype=np.float32), device=device)
    actions     = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    dones       = torch.as_tensor(dones,   dtype=torch.float32, device=device).unsqueeze(1)

    # 3) 현재 Q값 계산
    q_values = policy_net(states).gather(1, actions)        # 선택한 행동의 Q만 추출

    # 4) 타깃 Q값 계산
    with torch.no_grad():                                   # → DDQN: 선택(select)과 평가(evaluate)를 분리
        # 1) policy_net으로 다음 상태에서 취할 행동을 '선택'
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        # 2) target_net으로 그 행동의 Q값을 '평가'
        next_q = target_net(next_states).gather(1, next_actions)
        # 3) DDQN 타깃 Q값 계산
        target_q = rewards + (1 - dones) * GAMMA * next_q

    # 5) 손실 계산 및 역전파
    loss = nn.functional.mse_loss(q_values, target_q)       # MSE 손실 사용
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 이 배치에서의 Q값 평균 반환
    return q_values.mean().item()

# === 평가 함수 (탐험 없이 policy_net만 사용) =========================
def evaluate_policy(n_eval_episodes=10):
    """
    policy_net을 탐험 없이(epsilon=0) n_eval_episodes번 실행하여 평균 리워드 계산.
    반환값: (평균 리워드, 각 에피소드 리워드 리스트)
    """
    rewards = []

    for _ in range(n_eval_episodes):
        out = env.reset()
        state = out[0] if isinstance(out, tuple) else out
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_np = np.asarray(state, dtype=np.float32)
                state_t = torch.from_numpy(state_np).unsqueeze(0).to(device)
                action = policy_net(state_t).argmax(1).item()  # greedy

            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, term, trunc, _ = step_out
                done = term or trunc
            else:
                next_state, reward, done, _ = step_out

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    avg_reward = sum(rewards) / len(rewards)
    return avg_reward, rewards

# === 학습 루프 ==================================================
episode_rewards   = []   # 에피소드별 누적 리워드 기록
episode_q_means   = []   # 에피소드별 Q값 평균 기록
avg10eval_points  = []   # 10에피소드마다 평가한 ε=0 성능 기록 ← 추가
avg10r_points     = []   # 10회 단위 리워드 평균 포인트
avg10q_points     = []   # 10회 단위 Q 평균 포인트
global_step       = 0    # 전체 스텝 수 (ε 계산용)

for episode in range(1, TOTAL_EPISODES + 1):
    # 에피소드 초기화
    out = env.reset()
    state = out[0] if isinstance(out, tuple) else out
    ep_reward = 0
    done = False

    # 이 에피소드 동안 모은 배치 Q평균 저장용
    q_means_this_ep = []

    # 에피소드가 끝날 때까지 반복
    while not done:
        # 1) 행동 선택
        action = select_action(state, global_step)

        # 2) 환경 한 스텝 진행
        step_out = env.step(action)
        if len(step_out) == 5:                             # Gym ≥0.26
            next_state, reward, term, trunc, _ = step_out
            done = term or trunc
        else:                                              # Gym <0.26
            next_state, reward, done, _ = step_out

        # 3) 경험 저장
        replay_mem.append((state, action, reward, next_state, done))

        # 4) 상태·보상 업데이트 및 학습
        state = next_state
        ep_reward += reward
        global_step += 1
        q_mean = optimize()                                # 최적화 후 Q평균 받아옴
        if q_mean is not None:
            q_means_this_ep.append(q_mean)

    # 에피소드 종료 후 처리
    episode_rewards.append(ep_reward)                     # 리워드 기록
    # 이 에피소드의 Q평균 계산
    ep_avg_q = sum(q_means_this_ep) / len(q_means_this_ep) if q_means_this_ep else 0
    episode_q_means.append(ep_avg_q)

    # 타깃 네트워크 동기화
    if episode % TARGET_UPDATE_EPS == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 10개 단위로 평균 리포트·모델 저장
    if episode % SAVE_INTERVAL == 0:
        # 최근 10개 에피소드 리워드 평균
        recent_r = episode_rewards[-SAVE_INTERVAL:]
        avg10r = sum(recent_r) / SAVE_INTERVAL
        avg10r_points.append((episode, avg10r))
        # 최근 10개 에피소드 Q평균
        recent_q = episode_q_means[-SAVE_INTERVAL:]
        avg10q = sum(recent_q) / SAVE_INTERVAL
        avg10q_points.append((episode, avg10q))

        # ε=0 평가
        eval_r, detail_eval_r = evaluate_policy()
        avg10eval_points.append((episode, eval_r))

        # 콘솔 출력
        detail_r = ', '.join(f"{r:.0f}" for r in recent_r)
        detail_q = ', '.join(f"{q:.2f}" for q in recent_q)
        detail_eval_str = ', '.join(f"{r:.0f}" for r in detail_eval_r)

        print(f"Ep {episode-SAVE_INTERVAL+1:03d}-{episode:03d} | "
              f"Avg10r {avg10r:.1f} ({detail_r}) | "
              f"Avg10q {avg10q:.2f} ({detail_q}) | "
              f"Avg10Eval {eval_r:.1f} ({detail_eval_str})")

        # 체크포인트 저장 (.pth 파일)
        ckpt = os.path.join(MODEL_DIR, f"ep{episode}.pth")
        torch.save(policy_net.state_dict(), ckpt)

env.close()  # 환경 정리

# === 학습 결과 시각화 ============================================
plt.figure(figsize=(10, 6))
# 1) 전체 리워드 실선
plt.plot(episode_rewards, label="Reward")
# 2) 전체 Q평균 실선 (투명도 낮춰 표시)
plt.plot(episode_q_means, label="Q-mean", alpha=0.3)
# 3) 10개 단위 평균 포인트
if avg10r_points:
    xr, yr = zip(*avg10r_points)
    xq, yq = zip(*avg10q_points)
    plt.plot(xr, yr, "--o", label="Avg10r", markersize=5)
    plt.plot(xq, yq, "-s", label="Avg10q", markersize=5)

# 4) ε=0 평가 포인트
if avg10eval_points:
    xe, ye = zip(*avg10eval_points)
    plt.plot(xe, ye, ":^", label="Eval@0", markersize=6)

plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("CartPole DDDQN Training (Reward & Q-mean)")
plt.ylim(0, 510)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("cartpole_dddqn_rewards_qmean.png")
plt.show()
