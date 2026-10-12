# Kalman 滤波器轨迹预测模块
# 用于在 YOLO 检测帧率较低/不固定的情况下，高频预测目标像素坐标
# 支持异步 predict / update，适配视觉伺服控制系统
# YYZ 2025-12-10


import numpy as np
import time


class KalmanPredictor:
    """
    基于常速度模型 (Constant Velocity Model) 的卡尔曼滤波器轨迹预测器。

    状态向量 X = [u, v, u_dot, v_dot]^T
      - u, v          : 目标中心像素坐标
      - u_dot, v_dot  : 目标像素速度 (px/s)

    测量向量 Z = [u, v]^T （YOLO 检测输出的中心坐标）

    设计目标:
      - predict()  可被控制器以高频（100~200 Hz）调用，返回预测坐标
      - update()   仅在 YOLO 有效检测时调用（~10-30 Hz），修正滤波器状态
      - 支持变时间步长 dt，适应不定帧率输入
    """

    # —— 状态 & 测量维度 ——
    STATE_DIM = 4       # [u, v, u_dot, v_dot]
    MEAS_DIM = 2        # [u, v]

    def __init__(self,
                 init_state: np.ndarray = None,
                 init_cov: np.ndarray = None,
                 process_noise: float = 0.1,
                 meas_noise: float = 5.0):
        """
        初始化 Kalman 滤波器。

        参数:
            init_state    : 初始状态向量 (4,), 默认 [0,0,0,0]
            init_cov      : 初始协方差矩阵 (4,4), 默认较大值表示初始不确定度高
            process_noise : 过程噪声缩放系数 q (加速度方差, px²/s⁴)
            meas_noise    : 测量噪声标准差 (px), YOLO 定位误差典型值
        """
        # —— 状态向量 X (4x1) ——
        if init_state is not None:
            self.X = np.array(init_state, dtype=np.float64).reshape(self.STATE_DIM, 1)
        else:
            self.X = np.zeros((self.STATE_DIM, 1), dtype=np.float64)

        # —— 状态协方差矩阵 P (4x4) ——
        if init_cov is not None:
            self.P = np.array(init_cov, dtype=np.float64)
        else:
            # 初始不确定度较大：位置标准差 100px, 速度标准差 50px/s
            self.P = np.diag([100.0**2, 100.0**2, 50.0**2, 50.0**2])

        # —— 测量矩阵 H (2x4) : Z = H * X ——
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # —— 测量噪声协方差矩阵 R (2x2) ——
        # YOLO 检测的像素定位噪声，假设 u, v 独立同分布
        self.R = np.eye(self.MEAS_DIM, dtype=np.float64) * (meas_noise ** 2)

        # —— 过程噪声基础参数 q ——
        # Q 矩阵将根据 dt 动态计算 (离散化连续白噪声模型)
        self.q = process_noise

        # —— 时间戳管理 ——
        self.last_update_time: float = None   # 上一次 update() 的时间戳 (秒)
        self.last_predict_time: float = None  # 上一次 predict() 的时间戳 (秒)

        # —— 健康度 / 置信度指标 ——
        self.consecutive_predict_without_update: int = 0   # 连续纯预测次数(未更新)
        self.max_predict_without_update: int = 20           # 超过此值认为预测不可靠
        self.last_innovation: np.ndarray = None             # 最后一次更新的测量残差 (2,)
        self.last_innovation_cov: np.ndarray = None         # 对应创新协方差 S (2x2)

        # —— 统计信息 ——
        self.total_updates: int = 0
        self.total_predicts: int = 0

    # ========================== 公共接口 ==========================

    def predict(self, current_time: float = None) -> tuple:
        """
        在任意时刻预测目标像素位置 (高频调用接口)。
        - 不依赖 YOLO 检测结果，仅根据内部状态外推。
        - 返回 (u_pred, v_pred), 类型 float。

        参数:
            current_time: 当前时间戳 (秒)，若为 None 则使用 time.time()
        返回:
            (u_pred, v_pred): 预测的像素坐标
        """
        if current_time is None:
            current_time = time.time()

        # —— 计算时间步长 dt ——
        if self.last_predict_time is None:
            # 首次调用，无历史时间，无法外推，直接返回当前状态估计
            dt = 0.0
            self.last_predict_time = current_time
            self.total_predicts += 1
            return float(self.X[0, 0]), float(self.X[1, 0])

        dt = current_time - self.last_predict_time
        if dt <= 0:
            # 防止非单调时间戳
            return float(self.X[0, 0]), float(self.X[1, 0])

        dt = min(dt, 1.0)  # 限制最大步长防止过度外推

        # —— 构造状态转移矩阵 F (4x4) ——
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float64)

        # —— 构造过程噪声协方差矩阵 Q (4x4) ——
        # 连续白噪声离散化: Q = G * q_c * G^T * dt
        # 其中 G 是噪声增益, q_c = q * I_2
        # 对于 CV 模型: Q 的推导见公式 (14)
        dt2 = dt * dt
        dt3 = dt2 * dt
        q_t = self.q * self.q  # 加速度方差 (过程噪声功率谱密度)
        Q = np.array([
            [dt3/3,    0, dt2/2,    0],
            [0,    dt3/3,    0, dt2/2],
            [dt2/2,    0,    dt,    0],
            [0,    dt2/2,    0,    dt]
        ], dtype=np.float64) * q_t

        # —— Kalman 预测步骤 ——
        # (1) 状态先验估计: X_pred = F * X
        X_pred = F @ self.X
        # (2) 协方差先验估计: P_pred = F * P * F^T + Q
        P_pred = F @ self.P @ F.T + Q

        # —— 更新内部状态（仅预测，不更新协方差引用） ——
        self.X = X_pred
        self.P = P_pred
        self.last_predict_time = current_time
        self.total_predicts += 1

        # 如果连续预测次数过多（长期未更新），标记状态不可靠
        # 但依然返回预测值，由 get_confidence() 供控制器判断
        self.consecutive_predict_without_update += 1

        return float(self.X[0, 0]), float(self.X[1, 0])

    def update(self, u: float, v: float, timestamp: float = None) -> tuple:
        """
        当 YOLO 检测到目标时调用，用测量值修正滤波器状态。
        内部先执行 predict 以同步时间，再执行 measurement update。

        参数:
            u         : YOLO 检测的目标中心 u 坐标 (px)
            v         : YOLO 检测的目标中心 v 坐标 (px)
            timestamp : 当前帧时间戳 (秒)，若为 None 则使用 time.time()

        返回:
            (u_filtered, v_filtered): 滤波后的坐标估计
        """
        if timestamp is None:
            timestamp = time.time()

        # —— Step 1: 先 predict 到当前时刻，对齐时间 ——
        if self.last_update_time is not None:
            dt = timestamp - self.last_update_time
            if dt > 0:
                dt = min(dt, 1.0)
                F = np.array([
                    [1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1,  0],
                    [0, 0, 0,  1]
                ], dtype=np.float64)

                dt2 = dt * dt
                dt3 = dt2 * dt
                q_t = self.q * self.q
                Q = np.array([
                    [dt3/3,    0, dt2/2,    0],
                    [0,    dt3/3,    0, dt2/2],
                    [dt2/2,    0,    dt,    0],
                    [0,    dt2/2,    0,    dt]
                ], dtype=np.float64) * q_t

                self.X = F @ self.X
                self.P = F @ self.P @ F.T + Q

        # —— Step 2: 测量更新 (Kalman Update) ——
        Z = np.array([[u], [v]], dtype=np.float64)

        # 测量残差 (Innovation): y = Z - H * X
        y = Z - self.H @ self.X

        # 创新协方差: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # 卡尔曼增益: K = P * H^T * S^{-1}
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 数值异常时用伪逆兜底
            S_inv = np.linalg.pinv(S)

        K = self.P @ self.H.T @ S_inv

        # 状态更新: X = X + K * y
        self.X = self.X + K @ y

        # 协方差更新 (Joseph 形式，数值更稳定):
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I = np.eye(self.STATE_DIM, dtype=np.float64)
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # —— 记录创新信息，用于置信度计算 ——
        self.last_innovation = y.flatten()
        self.last_innovation_cov = S

        # —— 更新时间戳与统计 ——
        self.last_update_time = timestamp
        self.last_predict_time = timestamp
        self.total_updates += 1
        self.consecutive_predict_without_update = 0  # 重置连续预测计数

        return float(self.X[0, 0]), float(self.X[1, 0])

    def get_velocity(self) -> tuple:
        """
        返回当前滤波器估计的速度分量。

        返回:
            (u_dot, v_dot): 像素速度 (px/s)
        """
        return float(self.X[2, 0]), float(self.X[3, 0])

    def get_acceleration(self) -> tuple:
        """
        预留加速度接口 (常速度模型下恒为 0)。
        扩展为匀加速模型 (CA) 后可返回实际值。

        返回:
            (a_u, a_v): 像素加速度 (px/s²)
        """
        return 0.0, 0.0

    def get_state(self) -> np.ndarray:
        """
        返回完整状态向量 X = [u, v, u_dot, v_dot]^T (4x1)。
        """
        return self.X.copy()

    def get_covariance(self) -> np.ndarray:
        """
        返回状态协方差矩阵 P (4x4)。
        """
        return self.P.copy()

    def get_confidence(self) -> float:
        """
        返回当前预测可信度 (0~1)。

        判断依据:
          1) 连续未更新次数 —— 越大可信度越低
          2) 协方差矩阵 trace (位置不确定度) —— 越大可信度越低
          3) 最近一次更新的残差马氏距离 —— 异常时可信度低

        返回:
            confidence: 0.0 (完全不可信) ~ 1.0 (高度可信)
        """
        # —— 因子1: 连续预测惩罚 ——
        if self.consecutive_predict_without_update >= self.max_predict_without_update:
            return 0.0  # 长期未更新，完全不可信

        pred_factor = 1.0 - (self.consecutive_predict_without_update /
                             max(self.max_predict_without_update, 1))
        pred_factor = max(0.0, min(1.0, pred_factor))

        # —— 因子2: 协方差不确定度 (位置部分 trace) ——
        pos_trace = self.P[0, 0] + self.P[1, 1]  # u, v 方差之和
        # 当位置标准差 > 200px 时认为极不可信
        cov_factor = max(0.0, 1.0 - pos_trace / (200.0 ** 2))
        cov_factor = min(1.0, cov_factor)

        # —— 因子3: 创新残差马氏距离 ——
        innovation_factor = 1.0
        if (self.last_innovation is not None and
            self.last_innovation_cov is not None):
            y = self.last_innovation.reshape(2, 1)
            S = self.last_innovation_cov
            try:
                mahalanobis_sq = float(y.T @ np.linalg.inv(S) @ y)
                # 自由度为2的卡方分布: 95% 置信区间约为 5.99
                # 残差 > 阈值认为检测异常
                if mahalanobis_sq > 5.99:
                    innovation_factor = max(0.3, 1.0 - (mahalanobis_sq - 5.99) / 20.0)
            except np.linalg.LinAlgError:
                pass

        # —— 综合置信度 (加权几何平均) ——
        confidence = (pred_factor * 0.5 +
                      cov_factor * 0.3 +
                      innovation_factor * 0.2)

        return max(0.0, min(1.0, confidence))

    def is_reliable(self, threshold: float = 0.5) -> bool:
        """
        快速判断当前预测是否可靠 (封装 get_confidence 阈值判断)。

        参数:
            threshold: 可信度阈值，默认 0.5

        返回:
            True 表示预测可靠，可发送给控制系统
        """
        return self.get_confidence() >= threshold

    def reset(self, new_state: np.ndarray = None):
        """
        目标丢失 / 误检后重置滤波器状态。

        参数:
            new_state: 新的初始状态向量 (4,)，若为 None 则完全清零
        """
        if new_state is not None:
            self.X = np.array(new_state, dtype=np.float64).reshape(self.STATE_DIM, 1)
        else:
            self.X = np.zeros((self.STATE_DIM, 1), dtype=np.float64)

        # 重置协方差为较大值（高不确定度）
        self.P = np.diag([100.0**2, 100.0**2, 50.0**2, 50.0**2])

        # 清除时间戳
        self.last_update_time = None
        self.last_predict_time = None

        # 清除创新记录
        self.last_innovation = None
        self.last_innovation_cov = None

        # 重置计数器
        self.consecutive_predict_without_update = 0

    # ========================== 辅助方法 ==========================

    def get_info_dict(self) -> dict:
        """
        返回滤波器当前状态的摘要字典，便于调试与日志。
        """
        return {
            'state_u':       float(self.X[0, 0]),
            'state_v':       float(self.X[1, 0]),
            'velocity_u':    float(self.X[2, 0]),
            'velocity_v':    float(self.X[3, 0]),
            'pos_std_u':     float(np.sqrt(max(0, self.P[0, 0]))),
            'pos_std_v':     float(np.sqrt(max(0, self.P[1, 1]))),
            'vel_std_u':     float(np.sqrt(max(0, self.P[2, 2]))),
            'vel_std_v':     float(np.sqrt(max(0, self.P[3, 3]))),
            'confidence':    self.get_confidence(),
            'consecutive_pred': self.consecutive_predict_without_update,
            'total_updates': self.total_updates,
            'total_predicts': self.total_predicts,
            'last_update_ts': self.last_update_time,
        }


# ========================== 模块自测试 ==========================
if __name__ == "__main__":
    print("=" * 60)
    print("KalmanPredictor 自测试")
    print("=" * 60)

    # 创建预测器
    kf = KalmanPredictor(process_noise=0.5, meas_noise=3.0)

    # 模拟场景: YOLO 以 ~10Hz 检测, 控制器以 ~50Hz 调用 predict
    sim_start = time.time()

    # 初始检测
    kf.update(u=320.0, v=240.0, timestamp=sim_start)
    print(f"[t=0.00s] UPDATE  (320, 240) -> filtered: {kf.get_state().flatten()[:2]}")

    # 模拟 0.1s 后第二次检测 (目标向右下移动)
    t = sim_start + 0.1
    kf.update(u=325.0, v=245.0, timestamp=t)
    print(f"[t=0.10s] UPDATE  (325, 245) -> filtered: {kf.get_state().flatten()[:2]}, "
          f"vel=({kf.get_velocity()[0]:.1f}, {kf.get_velocity()[1]:.1f}) px/s")

    # 0.1s~0.2s 期间, 控制器以 ~50Hz 高频 predict
    print("\n--- 控制器高频 predict (50Hz 模拟) ---")
    for i in range(6):
        t += 0.02
        u_p, v_p = kf.predict(t)
        print(f"  t={t-sim_start:.3f}s  predict -> ({u_p:.1f}, {v_p:.1f}), "
              f"confidence={kf.get_confidence():.3f}")

    # 0.2s 又有 YOLO 检测
    t = sim_start + 0.2
    kf.update(u=332.0, v=252.0, timestamp=t)
    print(f"\n[t=0.20s] UPDATE  (332, 252) -> filtered: {kf.get_state().flatten()[:2]}, "
          f"confidence={kf.get_confidence():.3f}")

    # 模拟 1s 丢帧后继续 predict
    print("\n--- 模拟丢帧 1s (仅 predict) ---")
    for i in range(10):
        t += 0.1
        u_p, v_p = kf.predict(t)
        print(f"  t={t-sim_start:.3f}s  predict -> ({u_p:.1f}, {v_p:.1f}), "
              f"confidence={kf.get_confidence():.3f}, reliable={kf.is_reliable()}")

    # 查看最终状态摘要
    print("\n--- 滤波器状态摘要 ---")
    for k, v in kf.get_info_dict().items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.3f}")
        else:
            print(f"  {k:20s}: {v}")

    print("\n自测试完成!")
