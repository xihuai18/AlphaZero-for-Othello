# AlphaZero for Othello

Requirements

+ numpy==1.15.4
+ keras_resnet==0.1.0
+ Keras==2.2.4
+ numba==0.41.0
+ tqdm==4.28.1
+ torch==1.0.0
+ gym==0.10.9
+ imageio==2.4.1
+ old==0.01
+ ray==0.6.1
+ tensorboardX==1.5

Usages

+ train

  ```shell
  python train.py [-h] [--iter ITER] [--start_iter START_ITER]
                  [--log_dir LOG_DIR]
  ```

+ play

  ```shell
  python  AI_player.py [-h] [--load_dir LOAD_DIR] [--ai_side AI_SIDE]
                      [--mcts MCTS]
  ```


