import torch
import numpy as np
import time
from termcolor import colored
import inspect
import random
import sys


_print_refresh_rate = 0.25  # seconds

progress_bar_num_iters = None
progress_bar_len_str_num_iters = None
progress_bar_time_start = None
progress_bar_prev_duration = None


def progress_bar(i, len):
    bar_len = 20
    filled_len = int(round(bar_len * i / len))
    # percents = round(100.0 * i / len, 1)
    return '#' * filled_len + '-' * (bar_len - filled_len)


def progress_bar_init(message, num_iters, iter_name='Items', rejections=False):
    global progress_bar_num_iters
    global progress_bar_len_str_num_iters
    global progress_bar_time_start
    global progress_bar_prev_duration
    if num_iters < 1:
        raise ValueError('num_iters must be a positive integer')
    progress_bar_num_iters = num_iters
    progress_bar_time_start = time.time()
    progress_bar_prev_duration = 0
    progress_bar_len_str_num_iters = len(str(progress_bar_num_iters))
    print(message)
    sys.stdout.flush()
    if not rejections:
        print('Time spent  | Time remain.| Progress             | {} | {}/sec'.format(
            iter_name.ljust(progress_bar_len_str_num_iters * 2 + 1), iter_name))
    else:
        print('Time spent  | Time remain.| Progress             | {} | {}/sec '\
              '| Rejected Samples'.format(
            iter_name.ljust(progress_bar_len_str_num_iters * 2 + 1), iter_name))


def progress_bar_update(iter,rejections=None):
    global progress_bar_prev_duration
    duration = time.time() - progress_bar_time_start
    if rejections is None:
        if ((duration - progress_bar_prev_duration > _print_refresh_rate) or
            (iter >= progress_bar_num_iters - 1)):
            progress_bar_prev_duration = duration
            traces_per_second = (iter + 1) / duration
            print('{} | {} | {} | {}/{} | {:,.2f}       '.format(
                days_hours_mins_secs_str(duration),
                days_hours_mins_secs_str((progress_bar_num_iters - iter) / traces_per_second),
                progress_bar(iter, progress_bar_num_iters), str(iter).rjust(
                    progress_bar_len_str_num_iters),
                progress_bar_num_iters,
                traces_per_second), end='\r')
            sys.stdout.flush()
    else:
        if ((duration - progress_bar_prev_duration > _print_refresh_rate) or
                (iter >= progress_bar_num_iters - 1)):
            progress_bar_prev_duration = duration
            traces_per_second = (iter + 1) / duration
            print('{} | {} | {} | {}/{} | {:,.2f} |  {:,.2f}     '.format(
                days_hours_mins_secs_str(duration),
                days_hours_mins_secs_str((progress_bar_num_iters - iter) / traces_per_second),
                progress_bar(iter, progress_bar_num_iters), str(iter).rjust(
                    progress_bar_len_str_num_iters),
                progress_bar_num_iters,
                traces_per_second,rejections), end='\r')
            sys.stdout.flush()


def progress_bar_end(message=None):
    progress_bar_update(progress_bar_num_iters)
    print()
    if message is not None:
        print(message)


def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))


def has_nan_or_inf(value):
    if torch.is_tensor(value):
        value = torch.sum(value)
        isnan = int(torch.isnan(value)) > 0
        isinf = int(torch.isinf(value)) > 0
        return isnan or isinf
    else:
        value = float(value)
        return (value == float('inf')) or (value == float('-inf')) or (value == float('NaN'))


class LogProbError(Exception):
    pass


def eval_print(*expressions):
    print('\n\n' + colored(inspect.stack()[1][3], 'white', attrs=['bold']))
    frame = sys._getframe(1)
    max_str_length = 0
    for expression in expressions:
        if len(expression) > max_str_length:
            max_str_length = len(expression)
    for expression in expressions:
        val = eval(expression, frame.f_globals, frame.f_locals)
        if isinstance(val, np.ndarray):
            val = val.tolist()
        print('  {} = {}'.format(expression.ljust(max_str_length), repr(val)))

