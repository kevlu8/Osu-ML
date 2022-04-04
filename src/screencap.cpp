#include <Windows.h>

void SaveBitmapToFile(BITMAP bmp, LPCWSTR filename)
{
	HANDLE hFile = CreateFile(filename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE)
		return;

	BITMAPFILEHEADER bmfHeader;
	ZeroMemory(&bmfHeader, sizeof(BITMAPFILEHEADER));
	bmfHeader.bfType = 0x4D42;
	bmfHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + bmp.bmWidth * bmp.bmHeight * 3;
	bmfHeader.bfOffBits = 0x36;

	DWORD dwBytesWritten = 0;
	WriteFile(hFile, &bmfHeader, sizeof(BITMAPFILEHEADER), &dwBytesWritten, NULL);

	BITMAPINFOHEADER  bi;
	ZeroMemory(&bi, sizeof(BITMAPINFOHEADER));
	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = bmp.bmWidth;
	bi.biHeight = bmp.bmHeight;
	bi.biPlanes = 1;
	bi.biBitCount = 24;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	WriteFile(hFile, &bi, sizeof(BITMAPINFOHEADER), &dwBytesWritten, NULL);
	WriteFile(hFile, bmp.bmBits, bmp.bmWidth * bmp.bmHeight * 3, &dwBytesWritten, NULL);
	CloseHandle(hFile);
}

int main() {
	HWND hWnd = GetDesktopWindow();
	HDC hdc = GetDC(hWnd);
	int i = 0;
	while (true) {
		HBITMAP bmp;

		BitBlt((HDC)bmp, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), hdc, 0, 0, SRCCOPY);
		// save the bmp to file.bmp
		i++;
	}
}