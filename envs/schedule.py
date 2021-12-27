"""
Training schedules

author: William Tong (wtong@g.harvard.edu)
"""

from stable_baselines3.common.callbacks import BaseCallback

class ScheduleCallback(BaseCallback):
    def __init__(self, schedule, eval_env = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.checkpoints = iter(schedule)
        self.eval_env = eval_env

        self.curr_ckpt = next(self.checkpoints)
        self.curr_iter = 0
        self.curr_gen = 0
    
    def _on_training_start(self) -> None:
        self.training_env.env_method('queue_map', self.curr_ckpt['map'])

        if self.eval_env:
            self.eval_env.queue_map(self.curr_ckpt['map'])
    
    def _on_step(self) -> bool:
        self.curr_iter += 1
        if self.curr_iter > self.curr_ckpt['iters']:
            try:
                self.curr_ckpt = next(self.checkpoints)
            except StopIteration:
                return False

            self.training_env.env_method('queue_map', self.curr_ckpt['map'])
            self.curr_iter = 0

            if self.eval_env:
                self.eval_env.queue_map(self.curr_ckpt['map'])

            self.model.save(f'gen{self.curr_gen}')
            self.curr_gen += 1
        
        return True


class Schedule:
    def __init__(self, trail_class):
        self.trail_class = trail_class
        self.checkpoints = []
        self.curr_ckpt = 0
    
    def add_ckpt(self, iters, **trail_args):
        self.checkpoints.append({
            'iters': iters,
            'map': self.trail_class(**trail_args)
        })

        return self
    
    def __iter__(self):
        return iter(self.checkpoints)
    

    
    