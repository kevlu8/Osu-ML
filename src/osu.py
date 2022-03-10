import psutil

# Check if osu is running in Win32
def is_running(): 
    if "osu!.exe" in (p.name() for p in psutil.process_iter()):
        return True
    return False