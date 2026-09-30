"""Frozen original TOPP controller plus one yield after each completed physics tick.
The final successful tick returns directly; _rtc_ticks still counts it.
No action interpolation, arm schedule, success check, or planner is changed.
"""
from __future__ import annotations
import numpy as np

def take_action_ticks(self, action, action_type='qpos'):
    if self.take_action_cnt == self.step_lim or self.eval_success:
        return
    eval_video_freq = 1
    if self.eval_video_path is not None and self.take_action_cnt % eval_video_freq == 0:
        self.eval_video_ffmpeg.stdin.write(self.now_obs['observation']['head_camera']['rgb'].tobytes())
    self.take_action_cnt += 1
    print(f'\rstep: \x1b[92m{self.take_action_cnt} / {self.step_lim}\x1b[0m', end='', flush=True)
    self._update_render()
    if self.render_freq:
        self.viewer.render()
    actions = np.array([action])
    left_jointstate = self.robot.get_left_arm_jointState()
    right_jointstate = self.robot.get_right_arm_jointState()
    left_arm_dim = len(left_jointstate) - 1 if action_type == 'qpos' else 7
    right_arm_dim = len(right_jointstate) - 1 if action_type == 'qpos' else 7
    current_jointstate = np.array(left_jointstate + right_jointstate)
    left_arm_actions, left_gripper_actions, left_current_qpos, left_path = ([], [], [], [])
    right_arm_actions, right_gripper_actions, right_current_qpos, right_path = ([], [], [], [])
    left_arm_actions, left_gripper_actions = (actions[:, :left_arm_dim], actions[:, left_arm_dim])
    right_arm_actions, right_gripper_actions = (actions[:, left_arm_dim + 1:left_arm_dim + right_arm_dim + 1], actions[:, left_arm_dim + right_arm_dim + 1])
    left_current_gripper, right_current_gripper = (self.robot.get_left_gripper_val(), self.robot.get_right_gripper_val())
    left_gripper_path = np.hstack((left_current_gripper, left_gripper_actions))
    right_gripper_path = np.hstack((right_current_gripper, right_gripper_actions))
    if action_type == 'qpos':
        left_current_qpos, right_current_qpos = (current_jointstate[:left_arm_dim], current_jointstate[left_arm_dim + 1:left_arm_dim + right_arm_dim + 1])
        left_path = np.vstack((left_current_qpos, left_arm_actions))
        right_path = np.vstack((right_current_qpos, right_arm_actions))
        topp_left_flag, topp_right_flag = (True, True)
        try:
            times, left_pos, left_vel, acc, duration = self.robot.left_mplib_planner.TOPP(left_path, 1 / 250, verbose=True)
            left_result = dict()
            left_result['position'], left_result['velocity'] = (left_pos, left_vel)
            left_n_step = left_result['position'].shape[0]
        except Exception as e:
            topp_left_flag = False
            left_n_step = 50
        if left_n_step == 0:
            topp_left_flag = False
            left_n_step = 50
        try:
            times, right_pos, right_vel, acc, duration = self.robot.right_mplib_planner.TOPP(right_path, 1 / 250, verbose=True)
            right_result = dict()
            right_result['position'], right_result['velocity'] = (right_pos, right_vel)
            right_n_step = right_result['position'].shape[0]
        except Exception as e:
            topp_right_flag = False
            right_n_step = 50
        if right_n_step == 0:
            topp_right_flag = False
            right_n_step = 50
    elif action_type == 'ee':
        left_result = self.robot.left_plan_path(left_arm_actions[0])
        right_result = self.robot.right_plan_path(right_arm_actions[0])
        if left_result['status'] != 'Success':
            left_n_step = 50
            topp_left_flag = False
        else:
            left_n_step = left_result['position'].shape[0]
            topp_left_flag = True
        if right_result['status'] != 'Success':
            right_n_step = 50
            topp_right_flag = False
        else:
            right_n_step = right_result['position'].shape[0]
            topp_right_flag = True
    left_mod_num = left_n_step % len(left_gripper_actions)
    right_mod_num = right_n_step % len(right_gripper_actions)
    left_gripper_step = [0] + [left_n_step // len(left_gripper_actions) + (1 if i < left_mod_num else 0) for i in range(len(left_gripper_actions))]
    right_gripper_step = [0] + [right_n_step // len(right_gripper_actions) + (1 if i < right_mod_num else 0) for i in range(len(right_gripper_actions))]
    left_gripper = []
    for gripper_step in range(1, left_gripper_path.shape[0]):
        region_left_gripper = np.linspace(left_gripper_path[gripper_step - 1], left_gripper_path[gripper_step], left_gripper_step[gripper_step] + 1)[1:]
        left_gripper = left_gripper + region_left_gripper.tolist()
    left_gripper = np.array(left_gripper)
    right_gripper = []
    for gripper_step in range(1, right_gripper_path.shape[0]):
        region_right_gripper = np.linspace(right_gripper_path[gripper_step - 1], right_gripper_path[gripper_step], right_gripper_step[gripper_step] + 1)[1:]
        right_gripper = right_gripper + region_right_gripper.tolist()
    right_gripper = np.array(right_gripper)
    now_left_id, now_right_id = (0, 0)
    while now_left_id < left_n_step or now_right_id < right_n_step:
        if now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
            if topp_left_flag:
                self.robot.set_arm_joints(left_result['position'][now_left_id], left_result['velocity'][now_left_id], 'left')
            self.robot.set_gripper(left_gripper[now_left_id], 'left')
            now_left_id += 1
        if now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
            if topp_right_flag:
                self.robot.set_arm_joints(right_result['position'][now_right_id], right_result['velocity'][now_right_id], 'right')
            self.robot.set_gripper(right_gripper[now_right_id], 'right')
            now_right_id += 1
        self.scene.step()
        self._rtc_ticks += 1
        self._update_render()
        if self.check_success():
            self.eval_success = True
            self.get_obs()
            if self.eval_video_path is not None:
                self.eval_video_ffmpeg.stdin.write(self.now_obs['observation']['head_camera']['rgb'].tobytes())
            return
        yield None
    self._update_render()
    if self.render_freq:
        self.viewer.render()
