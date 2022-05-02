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

def compute_document_scores(documents, model):
    y_probas = model.predict_proba(documents)
    w_booked = 5
    w_clicked = 1
    w_combined = w_booked + w_clicked
    for i in range(len(y_probas[0])):
        p_b = y_probas[0][i][1]
        p_c = y_probas[1][i][1]
        score = (p_b * w_booked + p_c * w_clicked) / w_combined
        yield score
