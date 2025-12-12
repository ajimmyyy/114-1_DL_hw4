# record_demo_compatible.py
import cv2
import numpy as np
import pickle
import gzip
from models.client.tetris_env import TetrisEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv
from models.client.globel_constant import MOVE_LEFT, MOVE_RIGHT, ROTATE_CW, ROTATE_CCW, DROP

def record_demonstration(output_file="tetris_demo_stacked.pkl.gz", episodes=5):
    env = make_vec_env(TetrisEnv, n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)

    buffer = []
    
    print(f"=== 開始錄製 ===")
    print("操作: A(左) D(右) W/E(旋轉) S(下落)")
    
    KEY_MAPPING = {
        ord('a'): MOVE_LEFT, ord('d'): MOVE_RIGHT, 
        ord('q'): ROTATE_CW, ord('e'): ROTATE_CCW, 
        ord('s'): DROP
    }
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        
        while not done:
            display_img = obs[0, :, :, 3]
            
            if display_img.max() <= 1.0: display_img *= 255
            display_img = display_img.astype(np.uint8)
            display_img = cv2.resize(display_img, (200, 400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Recorder", display_img)
            
            key = cv2.waitKey(500) & 0xFF
            action = [-1]
            
            if key in KEY_MAPPING:
                action = [KEY_MAPPING[key]]
            elif key == 27:
                env.close()
                return

            next_obs, reward, dones, info = env.step(action)
            buffer.append((obs, action, reward, next_obs, dones))
            
            obs = next_obs
            score += reward[0]
            
            if dones[0]:
                done = True

            print(info)
                
        print(f"Episode {ep+1} Score: {score}")

    env.close()
    cv2.destroyAllWindows()
    
    print(f"Saving {len(buffer)} transitions...")
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(buffer, f)
    print("Done!")

if __name__ == "__main__":
    record_demonstration()