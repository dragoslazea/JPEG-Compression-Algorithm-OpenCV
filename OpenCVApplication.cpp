// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <queue>
#include <random>

using namespace cv;
using namespace std;

typedef struct {
	char val;
	uchar count;
}rleElement;

rleElement EOB = { (char)255, (uchar)0 };

// *************************************************************************************************
//							Auxiliary Functions
// *************************************************************************************************


int minInt(int x, int y) {
	if (x < y) {
		return x;
	}
	else {
		return y;
	}
}

int maxInt(int x, int y) {
	if (x > y) {
		return x;
	}
	else {
		return y;
	}
}

bool isInside(Mat img, int i, int j) {
	return (0 <= i && i < img.rows) && (0 <= j && j < img.cols);
}

int getNumberOfBlocksX(Mat img, int sizeOfBlock) {
	int blocksX = img.cols / sizeOfBlock;

	if (img.cols % sizeOfBlock) {
		blocksX++;
	}

	return blocksX;
}

int getNumberOfBlocksY(Mat img, int sizeOfBlock) {
	int blocksY = img.rows / sizeOfBlock;

	if (img.rows % sizeOfBlock) {
		blocksY++;
	}

	return blocksY;
}

Mat_<uchar> get8x8BlockAt(int x, int y, Mat_<uchar> img) {
	Mat_<uchar> block(8, 8);

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			if (isInside(img, 8 * y + j, 8 * x + i)) {
				block(j, i) = img(8 * y + j, 8 * x + i);
			}
			else {
				block(j, i) = 0;
			}
		}
	}

	return block;
}

Mat_<uchar> getLuminance(Mat_<Vec3b> img) {
	Mat_<Vec3b> y(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			y(i, j) = img(i, j)[0];
		}
	}

	return y;
}

Mat_<uchar> getRedChromatics(Mat_<Vec3b> img) {
	Mat_<Vec3b> cr(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			cr(i, j) = img(i, j)[1];
		}
	}

	return cr;
}

Mat_<uchar> getBlueChromatics(Mat_<Vec3b> img) {
	Mat_<Vec3b> cb(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			cb(i, j) = img(i, j)[2];
		}
	}

	return cb;
}

// *************************************************************************************************
//							Compression
// *************************************************************************************************


Mat_<uchar> chromaticDownsampling(Mat_<uchar> component) {
	int blocksX = getNumberOfBlocksX(component, 2);
	int blocksY = getNumberOfBlocksY(component, 2);

	Mat_<uchar> reducedComponent(blocksY, blocksX);

	for (int x = 0; x < blocksX; x++) {
		for (int y = 0; y < blocksY; y++) {
			int sum = 0;
			int num = 0;

			if (isInside(component, 2 * y, 2 * x)) {
				sum += component(2 * y, 2 * x);
				num++;
			}

			if (isInside(component, 2 * y + 1, 2 * x)) {
				sum += component(2 * y + 1, 2 * x);
				num++;
			}

			if (isInside(component, 2 * y, 2 * x + 1)) {
				sum += component(2 * y, 2 * x + 1);
				num++;
			}

			if (isInside(component, 2 * y + 1, 2 * x + 1)) {
				sum += component(2 * y + 1, 2 * x + 1);
				num++;
			}

			uchar avg = (uchar)round((float)sum / num);

			reducedComponent(y, x) = avg;
		}
	}

	return reducedComponent;
}

Mat_<Vec3b> colorSpaceConversion(Mat_<Vec3b> img) {
	Mat_<Vec3b> convertedImg(img.rows, img.cols);
	Mat_<Vec3b> imgOut(img.rows, img.cols);

	cvtColor(img, convertedImg, COLOR_BGR2YCrCb);

	int blocksX = getNumberOfBlocksX(img, 2);
	int blocksY = getNumberOfBlocksY(img, 2);

	Mat_<uchar> y(img.rows, img.cols);
	Mat_<uchar> cr(img.rows, img.cols);
	Mat_<uchar> cb(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			y(i, j) = convertedImg(i, j)[0];
			cr(i, j) = convertedImg(i, j)[1];
			cb(i, j) = convertedImg(i, j)[2];
		}
	}

	Mat_<uchar> smallCr(blocksY, blocksX);
	Mat_<uchar> smallCb(blocksY, blocksX);

	smallCr = chromaticDownsampling(cr);
	smallCb = chromaticDownsampling(cb);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			imgOut(i, j)[0] = convertedImg(i, j)[0];
		}
	}

	for (int x = 0; x < blocksX; x++) {
		for (int y = 0; y < blocksY; y++) {
			if (isInside(imgOut, 2 * y, 2 * x)) {
				imgOut(2 * y, 2 * x)[1] = smallCr(y, x);
				imgOut(2 * y, 2 * x)[2] = smallCb(y, x);
			}

			if (isInside(imgOut, 2 * y + 1, 2 * x)) {
				imgOut(2 * y + 1, 2 * x)[1] = smallCr(y, x);
				imgOut(2 * y + 1, 2 * x)[2] = smallCb(y, x);
			}

			if (isInside(imgOut, 2 * y, 2 * x + 1)) {
				imgOut(2 * y, 2 * x + 1)[1] = smallCr(y, x);
				imgOut(2 * y, 2 * x + 1)[2] = smallCb(y, x);
			}

			if (isInside(imgOut, 2 * y + 1, 2 * x + 1)) {
				imgOut(2 * y + 1, 2 * x + 1)[1] = smallCr(y, x);
				imgOut(2 * y + 1, 2 * x + 1)[2] = smallCb(y, x);
			}
		}
	}

	return imgOut;
}

Mat_<float> convertToSigned(Mat_<uchar> block) {
	Mat_<float> newBlock(block.rows, block.cols);

	block.convertTo(newBlock, CV_32FC1);

	newBlock = newBlock - 128.0f;

	return newBlock;
}

float ci(int i, int n) {
	if (i == 0) {
		return sqrt(1.0f / n);
	}
	else {
		return sqrt(2.0f / n);
	}
}

Mat_<float> discreteCosineTransform(Mat_<float> block) {
	Mat_<float> transformedBlock(block.rows, block.cols, 0.0f);

	for (int i = 0; i < block.rows; i++) {
		for (int j = 0; j < block.cols; j++) {

			float sX = 0.0f;

			for (int x = 0; x < 8; x++) {

				float sY = 0.0f;

				for (int y = 0; y < 8; y++) {
					sY += block(y, x) * cos((2 * y + 1) * i * PI / 16.0f) * cos((2 * x + 1) * j * PI / 16.0f);
				}

				sX += sY;
			}

			sX *= ci(i, 8);
			sX *= ci(j, 8);

			transformedBlock(i, j) = round(sX);
		}
	}

	return transformedBlock;
}


Mat_<char> quantization(Mat_<float> block) {
	uchar values[] = {
		16, 11, 10, 16,  24,  40,  51,  61,
		12, 12, 14, 19,  26,  58,  60,  55,
		14, 13, 16, 24,  40,  57,  69,  56,
		14, 17, 22, 29,  51,  87,  80,  62,
		18, 22, 37, 56,  68, 109, 103,  77,
		24, 35, 55, 64,  81, 104, 113,  92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103,  99
	};

	Mat_<uchar> q(8, 8, values);

	Mat_<char> qBlock(8, 8);

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			int value = (int)round(block(i, j) / q(i, j));
			qBlock(i, j) = (char)value;
		}
	}

	return qBlock;

}

char* zigZagTraversal(Mat_<char> mat) {
	char* result = (char*)calloc(mat.rows * mat.cols, sizeof(char));
	int count = 0;

	int row = 0;
	int col = 0;

	bool nextRow = false;

	for (int rowLimit = 1; rowLimit <= minInt(mat.rows, mat.cols); rowLimit++) {
		for (int i = 0; i < rowLimit; i++) {
			result[count] = mat(row, col);
			count++;

			if (i == rowLimit - 1) {
				break;
			}

			if (nextRow) {
				row++;
				col--;
			}
			else {
				row--;
				col++;
			}
		}

		if (rowLimit == minInt(mat.rows, mat.cols)) {
			break;
		}

		if (nextRow) {
			row++;
			nextRow = false;
		}
		else {
			col++;
			nextRow = true;
		}
	}

	if (row == 0) {
		if (col == mat.cols - 1) {
			row++;
		}
		else {
			col++;
		}

		nextRow = true;
	}
	else {
		if (row == mat.rows - 1) {
			col++;
		}
		else {
			row++;
		}

		nextRow = false;
	}

	for (int leftLimit = maxInt(mat.rows - 1, mat.cols - 1); leftLimit > 0; leftLimit--) {
		int rightLimit = minInt(leftLimit, minInt(mat.rows, mat.cols));

		for (int i = 0; i < rightLimit; i++) {
			result[count] = mat(row, col);
			count++;

			if (i == rightLimit - 1) {
				break;
			}

			if (nextRow) {
				row++;
				col--;
			}
			else {
				row--;
				col++;
			}
		}

		if (row == 0) {
			col++;
			nextRow = true;
		}
		else if (col == mat.cols - 1) {
			row++;
			nextRow = true;
		}
		else if (col == 0) {
			row++;
			nextRow = false;
		}
		else if (row == mat.rows - 1) {
			col++;
			nextRow = false;
		}
	}

	return result;
}

rleElement* rle(char* vals, int len, int* newLen) {
	rleElement* encoded = (rleElement*)calloc(len + 1, sizeof(rleElement));
	int n = 0;
	int i = 0;
	while (i < len) {
		int count = 1;

		while (i < len - 1 && vals[i] == vals[i + 1]) {
			count++;
			i++;
		}

		encoded[n].val = vals[i];
		encoded[n].count = count;
		n++;

		i++;
	}

	encoded[n] = EOB;
	n++;

	*newLen = n;

	return encoded;
}

void writeBlock(rleElement* vals, char* filename) {
	FILE* pf = fopen(filename, "ab");

	if (pf == NULL) {
		puts("Error opening the file...");
		return;
	}

	int i = 0;

	while (true) {
		fwrite(&vals[i], sizeof(rleElement), 1, pf);

		if (vals[i].val == EOB.val && vals[i].count == EOB.count) {
			break;
		}

		i++;
	}

	fclose(pf);

}

rleElement* readBlock(FILE* pf) {
	rleElement* rleArray = (rleElement*)calloc(65, sizeof(rleElement));
	int count = 0;

	rleElement e;

	while (true) {
		fread(&e, sizeof(rleElement), 1, pf);

		rleArray[count] = e;
		count++;

		if (e.val == EOB.val && e.count == EOB.count) {
			break;
		}
	}

	return rleArray;
}

void compressBlock(Mat_<uchar> block, char* compressedFileName) {
	Mat_<float> signedBlock = convertToSigned(block);

	Mat_<float> transformedBlock = discreteCosineTransform(signedBlock);

	Mat_<char> quantizedBlock = quantization(transformedBlock);

	char* vals = (char*)calloc(quantizedBlock.rows * quantizedBlock.cols, sizeof(char));

	vals = zigZagTraversal(quantizedBlock);

	int len = 0;
	rleElement* rleEl = rle(vals, 64, &len);

	writeBlock(rleEl, compressedFileName);
}

void compressImage(Mat_<Vec3b> img, char* filename) {
	Mat_<Vec3b> cvt(img.rows, img.cols);

	cvtColor(img, cvt, COLOR_BGR2YCrCb);

	FILE* pf = fopen(filename, "wb");
	fclose(pf);

	Mat_<uchar> lum(img.rows, img.cols);
	Mat_<uchar> cr(img.rows, img.cols);
	Mat_<uchar> cb(img.rows, img.cols);

	lum = getLuminance(cvt);
	cr = getRedChromatics(cvt);
	cb = getBlueChromatics(cvt);

	for (int x = 0; x < getNumberOfBlocksX(img, 8); x++) {
		for (int y = 0; y < getNumberOfBlocksY(img, 8); y++) {
			Mat_<uchar> yBlock = get8x8BlockAt(x, y, lum);
			Mat_<uchar> crBlock = get8x8BlockAt(x, y, cr);
			Mat_<uchar> cbBlock = get8x8BlockAt(x, y, cb);

			compressBlock(yBlock, filename);
			compressBlock(crBlock, filename);
			compressBlock(cbBlock, filename);
		}
	}
}

// *************************************************************************************************
//							Decompression
// *************************************************************************************************


char* rleDecode(rleElement* e) {
	char* decoded = (char*)calloc(64, sizeof(char));
	int n = 0;

	int i = 0;

	while (true) {
		if (e[i].count == EOB.count && e[i].val == EOB.val) {
			break;
		}

		int cnt = e[i].count;
		while (cnt > 0) {
			decoded[n] = e[i].val;
			n++;
			cnt--;
		}

		i++;

	}

	while (n < 64) {
		decoded[n] = 0;
		n++;
	}
	return decoded;
}

Mat_<char> zigZagReconstruction(char* vals) {
	Mat_<char> mat(8, 8);

	int count = 0;

	int row = 0;
	int col = 0;

	bool nextRow = false;

	for (int rowLimit = 1; rowLimit <= minInt(mat.rows, mat.cols); rowLimit++) {
		for (int i = 0; i < rowLimit; i++) {
			mat(row, col) = vals[count];
			count++;

			if (i == rowLimit - 1) {
				break;
			}

			if (nextRow) {
				row++;
				col--;
			}
			else {
				row--;
				col++;
			}
		}

		if (rowLimit == minInt(mat.rows, mat.cols)) {
			break;
		}

		if (nextRow) {
			row++;
			nextRow = false;
		}
		else {
			col++;
			nextRow = true;
		}
	}

	if (row == 0) {
		if (col == mat.cols - 1) {
			row++;
		}
		else {
			col++;
		}

		nextRow = true;
	}
	else {
		if (row == mat.rows - 1) {
			col++;
		}
		else {
			row++;
		}

		nextRow = false;
	}

	for (int leftLimit = maxInt(mat.rows, mat.cols) - 1; leftLimit > 0; leftLimit--) {
		int rightLimit = minInt(leftLimit, minInt(mat.rows, mat.cols));

		for (int i = 0; i < rightLimit; i++) {
			mat(row, col) = vals[count];
			count++;

			if (i == rightLimit - 1) {
				break;
			}

			if (nextRow) {
				row++;
				col--;
			}
			else {
				row--;
				col++;
			}
		}

		if (row == 0) {
			col++;
			nextRow = true;
		}
		else if (col == mat.cols - 1) {
			row++;
			nextRow = true;
		}
		else if (col == 0) {
			row++;
			nextRow = false;
		}
		else if (row == mat.rows - 1) {
			col++;
			nextRow = false;
		}
	}

	return mat;
}

Mat_<float> dequantization(Mat_<char> qBlock) {
	uchar values[] = {
		16, 11, 10, 16,  24,  40,  51,  61,
		12, 12, 14, 19,  26,  58,  60,  55,
		14, 13, 16, 24,  40,  57,  69,  56,
		14, 17, 22, 29,  51,  87,  80,  62,
		18, 22, 37, 56,  68, 109, 103,  77,
		24, 35, 55, 64,  81, 104, 113,  92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103,  99
	};

	Mat_<uchar> q(8, 8, values);

	Mat_<float> block(8, 8);

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			block(i, j) = qBlock(i, j) * q(i, j);
		}
	}

	return block;
}

Mat_<float> inverseDiscreteCosineTransform(Mat_<float> tBlock) {
	Mat_<float> block(tBlock.rows, tBlock.cols, 0.0f);

	for (int x = 0; x < tBlock.rows; x++) {
		for (int y = 0; y < tBlock.cols; y++) {

			float sI = 0.0f;

			for (int i = 0; i < 8; i++) {

				float sJ = 0.0f;

				for (int j = 0; j < 8; j++) {
					sJ += ci(i, 8) * ci(j, 8) * tBlock(j, i) * cos((2 * y + 1) * i * PI / 16.0f) * cos((2 * x + 1) * j * PI / 16.0f);
				}

				sI += sJ;
			}


			block(x, y) = round(sI);
		}
	}

	return block;
}

Mat_<uchar> convertToUnsigned(Mat_<float> block) {
	Mat_<float> newBlock(block.rows, block.cols);

	for (int i = 0; i < block.rows; i++) {
		for (int j = 0; j < block.cols; j++) {
			newBlock(i, j) = (uchar)(block(i, j) + 128.0f);
		}
	}

	return newBlock;
}

Mat_<uchar> decompressBLock(rleElement* code) {
	Mat_<uchar> decompressed(8, 8);

	int len = 0;

	char* decoded = rleDecode(code);

	Mat_<char> zigZagDecodedBlock = zigZagReconstruction(decoded);

	Mat_<float> dequantizedBlock = dequantization(zigZagDecodedBlock);

	Mat_<float> idctBlock = inverseDiscreteCosineTransform(dequantizedBlock);

	decompressed = convertToUnsigned(idctBlock);

	return decompressed;
}

Mat_<Vec3b> decompressImage(char* filename, int sizeX, int sizeY) {
	Mat_<Vec3b> decompressed(8 * sizeY, 8 * sizeX);

	FILE* pf = fopen(filename, "rb");

	if (!pf) {
		puts("Error opening the file...");
		return decompressed;
	}

	rleElement* rleY = (rleElement*)calloc(64, sizeof(rleElement));
	rleElement* rleCr = (rleElement*)calloc(64, sizeof(rleElement));
	rleElement* rleCb = (rleElement*)calloc(64, sizeof(rleElement));

	int countY = 0;
	int countCr = 0;
	int countCb = 0;

	for (int x = 0; x < sizeX; x++) {
		for (int y = 0; y < sizeY; y++) {
			countY = 0;
			countCr = 0;
			countCb = 0;

			rleElement yCode;

			while (true) {
				fread(&yCode, sizeof(rleElement), 1, pf);

				rleY[countY] = yCode;
				countY++;

				if (yCode.val == EOB.val && yCode.count == EOB.count) {
					break;
				}
			}

			rleElement crCode;

			while (true) {
				fread(&crCode, sizeof(rleElement), 1, pf);

				rleCr[countCr] = crCode;
				countCr++;

				if (crCode.val == EOB.val && crCode.count == EOB.count) {
					break;
				}
			}

			rleElement cbCode;

			while (true) {
				fread(&cbCode, sizeof(rleElement), 1, pf);

				rleCb[countCb] = cbCode;
				countCb++;

				if (cbCode.val == EOB.val && cbCode.count == EOB.count) {
					break;
				}
			}

			Mat_<uchar> decYBlock = decompressBLock(rleY);
			Mat_<uchar> decCrBlock = decompressBLock(rleCr);
			Mat_<uchar> decCbBlock = decompressBLock(rleCb);

			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					decompressed(8 * y + j, 8 * x + i)[0] = decYBlock(j, i);
					decompressed(8 * y + j, 8 * x + i)[1] = decCrBlock(j, i);
					decompressed(8 * y + j, 8 * x + i)[2] = decCbBlock(j, i);
				}
			}
		}
	}

	Mat_<Vec3b> result(8 * sizeY, 8 * sizeX);

	cvtColor(decompressed, result, COLOR_YCrCb2BGR);

	fclose(pf);

	return result;
}

// *************************************************************************************************
//							Test functions
// *************************************************************************************************

void getBlockTest() {
	int size = 210;

	uchar a[210];

	for (int i = 0; i < size; i++) {
		a[i] = i;
	}

	Mat_<uchar> m(14, 15, a);

	cout << m << endl << endl;

	for (int x = 0; x < getNumberOfBlocksX(m, 8); x++) {
		for (int y = 0; y < getNumberOfBlocksY(m, 8); y++) {
			cout << "x = " << x << ", y = " << y << endl;
			Mat_<uchar> b = get8x8BlockAt(x, y, m);
			cout << b << endl << endl;
			Mat_<float> nb = convertToSigned(b);
		}
	}
}

void chromaticDownsamplingTest() {
	uchar vals[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	Mat_<uchar> m(5, 3, vals);
	cout << "Original mat: " << endl << m << endl << endl;
	Mat_<uchar> rm(5, 3);
	rm = chromaticDownsampling(m);
	cout << "Reduced mat: " << endl << rm << endl << endl;
}

void idctTest() {
	uchar values[] = { 52, 55, 61, 66,  70,  61,  64, 73,
					   64, 59, 55, 90,  109, 85,  69, 72,
					   62, 59, 68, 113, 144, 104, 66, 73,
					   63, 58, 71, 122, 154, 106, 70, 69,
					   67, 61, 68, 104, 126, 88,  68, 70,
					   79, 65, 60, 70,  77,  68,  58, 75,
					   85, 71, 64, 59,  55,  61,  65, 83,
					   87, 79, 69, 68,  65,  76,  78, 94 };

	Mat_<uchar> b(8, 8, values);

	Mat_<float> sb = convertToSigned(b);

	Mat_<float> tb = discreteCosineTransform(sb);

	cout << "Initial signed block: " << endl << sb << endl << endl;
	cout << "Transformed block: " << endl << tb << endl << endl;

	Mat_<float> ib = inverseDiscreteCosineTransform(tb);
	Mat_<float> ub = convertToUnsigned(ib);

	cout << "Inverse dct transformed block: " << endl << ib << endl << endl;
	cout << "Inverse dct transformed unsigned block: " << endl << ub << endl << endl;
}

void compressOneBlockTest() {
	uchar values[] = { 52, 55, 61, 66,  70,  61,  64, 73,
					   64, 59, 55, 90,  109, 85,  69, 72,
					   62, 59, 68, 113, 144, 104, 66, 73,
					   63, 58, 71, 122, 154, 106, 70, 69,
					   67, 61, 68, 104, 126, 88,  68, 70,
					   79, 65, 60, 70,  77,  68,  58, 75,
					   85, 71, 64, 59,  55,  61,  65, 83,
					   87, 79, 69, 68,  65,  76,  78, 94 };

	Mat_<uchar> b(8, 8, values);

	Mat_<float> sb = convertToSigned(b);

	Mat_<float> tb = discreteCosineTransform(sb);

	Mat_<char> qb = quantization(tb);

	cout << "Initial block: " << endl << b << endl << endl;
	cout << "Signed block: " << endl << sb << endl << endl;
	cout << "Transformed block: " << endl << tb << endl << endl;
	cout << "Quantized block: " << endl << qb << endl << endl;
	cout << "Zig zag traversal: " << endl;

	char* array = zigZagTraversal(qb);

	for (int i = 0; i < 64; i++) {
		printf("%d ", array[i]);
	}
	
	cout << endl << endl << "Run-lenght encoding:" << endl;

	int len = 0;
	rleElement* rleEl = rle(array, 64, &len);

	int i = 0;
	while (true) {
		if (rleEl[i].val == EOB.val && rleEl[i].count == EOB.count) {
			printf("EOB\n\n");
			break;
		}
		printf("{ %d x %d } ", rleEl[i].val, rleEl[i].count);
		i++;
	}

	cout << endl << endl;
}

void compressAndDecompressBlockTest() {
	uchar values[] = { 52, 55, 61, 66,  70,  61,  64, 73,
					   64, 59, 55, 90,  109, 85,  69, 72,
					   62, 59, 68, 113, 144, 104, 66, 73,
					   63, 58, 71, 122, 154, 106, 70, 69,
					   67, 61, 68, 104, 126, 88,  68, 70,
					   79, 65, 60, 70,  77,  68,  58, 75,
					   85, 71, 64, 59,  55,  61,  65, 83,
					   87, 79, 69, 68,  65,  76,  78, 94 };

	Mat_<uchar> b(8, 8, values);

	cout << "Initial block: " << endl << b << endl << endl;

	cout << "Compression of the block: " << endl;
	compressBlock(b, "compressedBlock.bin");

	FILE* pf = fopen("compressedBlock.bin", "rb");

	rleElement* code = readBlock(pf);

	fclose(pf);

	int i = 0;
	while (true) {
		if (code[i].val == EOB.val && code[i].count == EOB.count) {
			printf("EOB\n\n");
			break;
		}

		printf("{ %d x %d } ", code[i].val, code[i].count);
		i++;
	}

	Mat_<uchar> db = decompressBLock(code);

	cout << "Decompressed block: " << endl << db << endl << endl;
}

void zigZagTest() {
	uchar values[] = { 52, 55, 61, 66,  70,  61,  64, 73,
					   64, 59, 55, 90,  109, 85,  69, 72,
					   62, 59, 68, 113, 144, 104, 66, 73,
					   63, 58, 71, 122, 154, 106, 70, 69,
					   67, 61, 68, 104, 126, 88,  68, 70,
					   79, 65, 60, 70,  77,  68,  58, 75,
					   85, 71, 64, 59,  55,  61,  65, 83,
					   87, 79, 69, 68,  65,  76,  78, 94 };

	Mat_<uchar> b(8, 8, values);

	cout << b << endl << endl;

	char* a = zigZagTraversal(b);

	for (int i = 0; i < 64; i++) {
		printf("%d ", a[i]);
	}

	cout << endl;

	Mat_<char> rebuilt = zigZagReconstruction(a);

	cout << rebuilt << endl << endl;
}

void compressAndDecompressImageTest(Mat_<Vec3b> img) {
	fclose(fopen("compressed.bin", "wb"));

	compressImage(img, "compressed.bin");

	Mat_<Vec3b> decompressed = decompressImage("compressed.bin", getNumberOfBlocksX(img, 8), getNumberOfBlocksY(img, 8));
	
	imshow("original img", img);
	imshow("decompressed img", decompressed);
	waitKey(0);
}

int main()
{
	int op;
	do
	{
		destroyAllWindows();
		printf("Menu:\n");
		printf("1. Get a 8x8 block by its position (example)\n");
		printf("2. Show steps in compressing a block (example)\n");
		printf("3. Chromatic Downsampling (example)\n");
		printf("4. Compress and decompress a single block (example)\n");
		printf("5. Zig zag traversal (example)\n");
		printf("6. Inverse discrete cosine transform (example)\n");
		printf("7. Compress and decompress an image\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
			case 1:
				getBlockTest();
				break;
			case 2:
				compressOneBlockTest();
				break;
			case 3:
				chromaticDownsamplingTest();
				break;
			case 4:
				compressAndDecompressBlockTest();
				break;
			case 5:
				zigZagTest();
				break;
			case 6:
				idctTest();
				break;
			case 7:
			{
				compressAndDecompressImageTest(imread("Images/Set/mexico.bmp", IMREAD_COLOR));
				break;
			}


		}
	}
	while (op!=0);
	return 0;
}