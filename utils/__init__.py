from torch.utils.tensorboard import SummaryWriter
import os
import datetime


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def get_tensorboard():
    base_path = 'log_tensorboard'

    dir_name = '{}-{}'.format('model', get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer

def early_stop(score, best, step_count, max_step=10):
    if score[0][1] > best[0][1]:
        return score, step_count, False
    else:
        step_count += 1
        if step_count >= max_step:
            return best, step_count, True
        return best, step_count, False