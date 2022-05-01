from psutil import process_iter
# from win32api import GetSystemMetrics

# Check if osu is running in Win32
def is_running(): 
	if "osu!.exe" in (p.name() for p in process_iter()):
		return True
	return False

# def getScreenSize():
#	return GetSystemMetrics(0), GetSystemMetrics(1) # width, height
