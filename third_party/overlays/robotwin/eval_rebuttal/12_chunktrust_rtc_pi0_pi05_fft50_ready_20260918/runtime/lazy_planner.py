"""Defer unused expert CuRobo allocations; retain its exact CPU gripper method.

The evaluated qpos controller exclusively calls the original MPLib TOPP. The
expert's inherited plan_grippers is pure NumPy and needs no CUDA model. If an
expert method is ever needed, first attribute access constructs the real planner.
This patch is process-local; the RoboTwin repository is not edited.
"""
def install(enabled=True):
    import envs.robot.robot as robot_module
    base=robot_module.CuroboPlanner
    if not enabled:
        if getattr(base,'_rtc_lazy_expert',False):robot_module.CuroboPlanner=base._rtc_original
        return
    if getattr(base,'_rtc_lazy_expert',False):return

    class LazyCuroboPlanner(base):
        _rtc_lazy_expert=True
        _rtc_original=base
        def __init__(self,*args,**kwargs):
            self._rtc_constructor=(args,kwargs)
            self._rtc_initializing=False
            self._rtc_initialized=False

        def __getattr__(self,name):
            if name.startswith('_rtc_') or self._rtc_initializing:
                raise AttributeError(name)
            if not self._rtc_initialized:
                self._rtc_initializing=True
                try:
                    args,kwargs=self._rtc_constructor
                    base.__init__(self,*args,**kwargs)
                    self._rtc_initialized=True
                finally:self._rtc_initializing=False
            return object.__getattribute__(self,name)

    robot_module.CuroboPlanner=LazyCuroboPlanner
