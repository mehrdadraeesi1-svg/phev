import csv
import numpy as np

class LoadFileData:
    def __init__(self, nameFile, typeFile, delimiter=',', startRow=1):
        self.nameFile = nameFile
        self.typeFile = typeFile
        self.delimiter = delimiter
        self.startRow = startRow
        self.x = []
        self.y = []
        self.z = []

        if typeFile == 'xy' or typeFile == 'XY':
            self.read_xy_file()
        elif typeFile == 'XYZ':
            self.read_xyz_file()
        elif typeFile == '2DTable':
            self.read_2d_table_file()
        elif typeFile == 'M1DTable':
            self.read_m1d_table_file()

    def read_xy_file(self):
        with open(self.nameFile, 'r') as file:
            data = file.read()
        self.read_xy_text(data)

    def read_xyz_file(self):
        with open(self.nameFile, 'r') as file:
            data = file.read()
        self.read_xyz_text(data)

    def read_2d_table_file(self):
        with open(self.nameFile, 'r') as file:
            data = file.read()
        self.read_2d_table_text(data)

    def read_m1d_table_file(self):
        with open(self.nameFile, 'r') as file:
            data = file.read()
        self.read_m1d_table_text(data)

    def read_xy_text(self, data):
        self.x = []
        self.y = []
        rows = data.split('\n')
        for i in range(self.startRow, len(rows)):
            values = rows[i].split(self.delimiter)
            self.y.append(float(values[1]))
            self.x.append(float(values[0]))

    def read_xyz_text(self, data):
        self.x = []
        self.y = []
        self.z = []
        rows = data.split('\n')
        for i in range(self.startRow, len(rows)):
            values = rows[i].split(self.delimiter)
            self.z.append(float(values[2]))
            self.y.append(float(values[1]))
            self.x.append(float(values[0]))

    def read_2d_table_text(self, data):
        self.row_size = 0
        self.col_size = 0
        self.x = []
        self.y = []
        self.z = []
        rows = data.split('\n')
    
        for i in range(self.startRow - 1, len(rows)):
            values = rows[i].split(self.delimiter)
            # print(values)
            if values[0][0] == '#':
                self.startRow += 1
            else:
                if i == (self.startRow - 1):
                    self.col_size = int(values[0])
                elif i == self.startRow:
                    self.row_size = int(values[0])
                elif i == (self.startRow + 1):
                    self.x = self.row_to_array(values, self.col_size)
                elif i == (self.startRow + 2):
                    self.y = self.row_to_array(values, self.row_size)
                else:
                    self.z.append(self.row_to_array(values, self.row_size))

    def read_m1d_table_text(self, data):
        self.x = []
        self.y = []
        self.z = []
        rows = data.split('\n')
        flag = 0
        j = 0
        N = 0
        x1, z1 = [], []
        for i in range(self.startRow - 1, len(rows)):
            values = rows[i].split(self.delimiter)
            if values[0][0] == '#':
                self.startRow += 1
            if flag == 0:
                self.y.append(float(values[0]))
                N = int(values[1])
                flag = 1
                j = 0
            else:
                x1.append(float(values[1]))
                z1.append(float(values[2]))
                j = j + 1
            if j == N:
                self.x.append(x1)
                self.z.append(z1)
                x1 = []
                z1 = []
                flag = 0

    def row_to_array(self, row, n):
        arr = []
        for i in range(n):
            arr.append(float(row[i]))
        return arr

    def linear_1d(self, Xnew):
        n = len(self.x)
        Ynew, yl, yu = self.y[0], self.y[-1], self.y[0]
        lb, ub = self.x[0], self.x[-1]
        for i in range(1, n - 1):
            if Xnew > self.x[i]:
                lb, yl, ub, yu = self.x[i], self.y[i], self.x[i + 1], self.y[i + 1]
        Ynew = yl * (ub - Xnew) / (ub - lb) + yu * (Xnew - lb) / (ub - lb)
        return Ynew

    def interp2d(self, arr, xarr, yarr, x, y):
        ix1 = self.ind_largest_number_index_less_than(xarr, x)
        x1 = xarr[ix1]
        ix2, x2, iy2, y2 = 0, 0, 0, 0

        if ix1 == len(xarr) - 1 or xarr[ix1] == x:
            x2 = x1
            ix2 = ix1
        else:
            x2 = xarr[ix1 + 1]
            ix2 = ix1 + 1
        iy1 = self.ind_largest_number_index_less_than(yarr, y)
        y1 = yarr[iy1]
        if iy1 == len(yarr) - 1 or yarr[iy1] == y:
            y2 = y1
            iy2 = iy1
        else:
            y2 = yarr[iy1 + 1]
            iy2 = iy1 + 1

        q11 = arr[ix1][iy1]
        q21 = arr[ix2][iy2]
        q12 = arr[ix1][iy2]
        q22 = arr[ix1][iy2]

        x2x1 = x2 - x1
        y2y1 = y2 - y1
        x2x = x2 - x
        y2y = y2 - y
        xx1 = x - x1
        yy1 = y - y1

        denom = x2x1 * y2y1

        f11 = q11 * x2x * y2y
        f21 = q21 * xx1 * y2y
        f12 = q12 * x2x * yy1
        f22 = q22 * xx1 * yy1

        value = (f11 + f21 + f12 + f22) / denom
        return value

    def ind_largest_number_index_less_than(self, arr, x):
        largest_index = -1
        largest_number = float('-inf')
        for i in range(len(arr)):
            if arr[i] <= x and arr[i] > largest_number:
                largest_number = arr[i]
                largest_index = i
        return largest_index
if __name__=='__main__':
    # Example Usage:
    data_loader = LoadFileData('OCV.data', '2DTable','    ')
    print(data_loader.x)
    print(data_loader.y)
    print(data_loader.z)
