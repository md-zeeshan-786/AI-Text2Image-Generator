
import gc
import torch
from contextlib import contextmanager

def free_cuda():
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

def free_all():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            free_cuda()
    except Exception:
        pass

@contextmanager
def status_spinner(st, text="Working..."):
    with st.status(text, state="running") as status:
        yield status
        status.update(label="Done", state="complete")
