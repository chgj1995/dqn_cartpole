# visualize.py

import os, time
import torch, gym
import torch.nn as nn

ENV_NAME   = "CartPole-v1"    # 사용할 Gym 환경 이름
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env    = gym.make(ENV_NAME)
n_s    = env.observation_space.shape[0]  # 상태(state) 공간 차원 수
n_a    = env.action_space.n             # 행동(action) 공간 개수


# 1) 학습에 쓰인 QNet 정의 (입/출력 크기만 맞추면 됩니다)
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_s, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_a)
        )
    def forward(self, x):
        return self.net(x)

# 2) 저장된 모델 불러오기
model_path = os.path.join("model_ddqn", "ep1000.pth")
assert os.path.exists(model_path), f"모델이 없습니다: {model_path}"
model = QNet()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 3) 환경 생성 (render_mode="human" 으로 시각화)
env = gym.make(ENV_NAME, render_mode="human")

for ep in range(5):                  # 5 에피소드 실행
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    done = False
    total_r = 0

    while not done:
        env.render()
        # 모델 예측 행동
        with torch.no_grad():
            action = int(model(torch.FloatTensor(state)).argmax().item())
        # 한 스텝 진행
        step = env.step(action)
        if len(step) == 4:
            state, reward, done, _ = step
        else:
            state, reward, term, trunc, _ = step
            done = term or trunc
        total_r += reward
        time.sleep(0.02)            # 프레임 속도 조절

    print(f"[Vis] Episode {ep+1}: Reward = {total_r}")
    
env.close()
