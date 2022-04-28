import time

_start_time = time.time()

def reset_timer():
    global _start_time
    _start_time = time.time()

def print_elapsed_time(prefix='', suffix=': '):
    global _start_time
    elapsed_time = time.time() - _start_time
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_minutes = int((elapsed_time - elapsed_hours * 3600) / 60)
    elapsed_seconds = int(elapsed_time - elapsed_hours * 3600 - elapsed_minutes * 60)
    formatted_time = f'{elapsed_hours:02}h {elapsed_minutes:02}m {elapsed_seconds:02}s'
    print(f'{prefix}{formatted_time}{suffix}', end='')

def tprint(s, end='\n'):
    print_elapsed_time()
    print(s, end=end)
