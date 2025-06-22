import time
from datetime import datetime

class TurnCounter:
    def __init__(self):
        self.start_time = time.time()
        self.turn_count = 0
    
    def next_turn(self) -> int:
        self.turn_count += 1
        return self.turn_count
    
    def time_since_start(self) -> float:
        return time.time() - self.start_time
    
    def get_timestamp(self) -> str:
        return datetime.now().isoformat()

def format_duration(seconds: float) -> str:
    """Format duration in human-readable form"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds // 60
    seconds %= 60
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"
    hours = minutes // 60
    minutes %= 60
    return f"{int(hours)}h {int(minutes)}m"