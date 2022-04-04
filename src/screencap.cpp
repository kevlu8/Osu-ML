// this was copied from my older project

#include <iostream>
#include <vector>
#include <Windows.h>
#include <fstream>
#include <filesystem>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
	UINT num = 0, size = 0;

	Gdiplus::ImageCodecInfo* pImageCodecInfo = NULL;

	Gdiplus::GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;

	pImageCodecInfo = (Gdiplus::ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL)
		return -1;

	Gdiplus::GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j) {
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;
		}
	}
	free(pImageCodecInfo);
	return -1;
}

void takeSS(LPCWSTR filename) {
	HDC hdc = GetDC(NULL);
	HDC hdcMem = CreateCompatibleDC(hdc);
	HBITMAP hbmp = CreateCompatibleBitmap(hdc, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
	SelectObject(hdcMem, hbmp);
	BitBlt(hdcMem, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), hdc, 0, 0, SRCCOPY);
	//BitBlt(hdc, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), hdcMem, 0, 0, SRCCOPY);

	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR token;
	Gdiplus::GdiplusStartup(&token, &gdiplusStartupInput, NULL);

	Gdiplus::Bitmap* image = new Gdiplus::Bitmap(hbmp, NULL);

	CLSID clsid;
	int retVal = GetEncoderClsid(L"image/png", &clsid);

	image->Save(filename, &clsid, NULL);
	delete image;

	Gdiplus::GdiplusShutdown(token);
	DeleteObject(hbmp);
	DeleteDC(hdcMem);
	ReleaseDC(NULL, hdc);
}

void video(const int fps = 144) {
	std::filesystem::current_path("C:\\Users\\kevie\\Desktop\\GitHub\\Osu-ML\\src\\data\\imgs\\");
	std::filesystem::file_status s{};
	int i = 0;
	wchar_t si[32];
	while (true) {
		_itow_s(i, si, 10);
		Sleep(ceil(1000 / fps));
		takeSS(si);
		i++;
	}
}

int main() {
	SetConsoleTitle(L"LWSR Screen Recorder");
	int fps; bool stop = false;
	std::string cmd;

	std::cout << "Enter a command: ";
	std::cin >> cmd;
	while (cmd != "exit" || cmd != "e") {
		if (cmd == "record" || cmd == "rec" || cmd == "r") {
			std::filesystem::path prevDir = std::filesystem::current_path();
			std::cout << "\nRecord at how many FPS: ";
			std::cin >> fps;

			HANDLE recThread = CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)video, &fps, NULL, NULL);

			std::cout << "\nRecording started. Type \"stop\" at any time to stop recording: ";
			std::cin >> cmd;

			if (cmd == "stop" || cmd == "s") {
				TerminateThread(recThread, 0);
				CloseHandle(recThread);
				std::filesystem::current_path(prevDir);
			}
		}
		else if (cmd == "ss" || cmd == "screenshot") {
			HWND cwnd = GetConsoleWindow();
			ShowWindow(cwnd, SW_MINIMIZE);
			takeSS(L"Screenshot.png");
			ShowWindow(cwnd, SW_NORMAL);
			std::cout << "Successfully took a screenshot and copied to your clipboard.\n";
		}
		else if (cmd == "help" || cmd == "h") {
			std::cout << "record/rec/r - Start a recording.\nstop/s - Stop the current recording and save it\nscreenshot/ss - Takes a screenshot of your entire screen";
		}
		std::cout << "Enter a command: ";
		std::cin >> cmd;
	}
	
	return 0;
}
