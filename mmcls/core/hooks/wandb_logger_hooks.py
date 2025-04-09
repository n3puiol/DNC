from mmcv.runner import HOOKS, LoggerHook


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):
    """Class to log metrics to wandb.
    It requires `wandb` to be installed.
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 commit=True,
                 by_epoch=True,
                 with_step=True,
                 log_artifact=True):
        super(WandbLoggerHook, self).__init__(interval, ignore_last, reset_flag,
                                              by_epoch)
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step
        self.log_artifact = log_artifact

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def before_run(self, runner):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                self.wandb.log(tags, step=self.get_iter(runner), commit=self.commit)
            else:
                self.wandb.log(tags, commit=self.commit)

    def after_run(self, runner):
        if self.log_artifact:
            self.wandb.log_artifact(
                self.wandb.Artifact(
                    'model-checkpoint',
                    type='model',
                    metadata=dict(runner.meta)
                ),
                aliases=['latest']
            )
        self.wandb.finish()
