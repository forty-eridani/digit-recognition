#include <stdio.h>
#include "matrix.h"

int main() {
	double marr[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9
	};

	Matrix m = CreateMatrix(marr, 3, 3);

	PrintMatrix(m);

	FreeMatrix(&m);
}
