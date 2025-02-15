package com.ai.networks.convolutional.train.backpropagation;

import com.ai.networks.Tokens;
import com.ai.networks.convolutional.Filter;
import com.ai.networks.convolutional.Matrix;

import java.util.Random;

final class TrainFilter extends Filter {
    private Matrix[] lastMode;

    TrainFilter(int size, int count){
        matrices = new Matrix[count];
        lastMode = new Matrix[count];
        Random random = new Random(System.currentTimeMillis());
        for (int k = 0; k < count;k++) {
            matrices[k] = new Matrix(size, size);
            lastMode[k] = new Matrix(size, size);
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    matrices[k].addTo(i, j, random.nextDouble());
                }
            }
        }
        biasValue = random.nextDouble();
    }

    TrainFilter(Filter f){
        matrices = f.getMatrices();
        biasValue = f.getBiasValue();
    }

    public void correcting(Matrix[] in, Matrix deltasLayer, double speed, double alpha, double maxWeight){
        boolean f = false;
        for (int k = 0; k < in.length; k++) {
            for (int i = 0; i < matrices[k].getH(); i++) {
                for (int j = 0; j < matrices[k].getW(); j++) {
                    double con = 0;
                    for (int ii = 0; ii < deltasLayer.getH(); ii++) {
                        for (int jj = 0; jj < deltasLayer.getW(); jj++) {
                            con += in[k].getCell(ii + i, jj + j) * deltasLayer.getCell(ii, jj);
                        }
                    }
                    double wd = speed * con + alpha * lastMode[k].getCell(i, j);
                    matrices[k].addTo(i, j, wd);
                    f = f || matrices[k].getCell(i, j) > maxWeight;
                    lastMode[k].addTo(i, j, wd - lastMode[k].getCell(i, j));
                }
            }
        }
        if(f){
            for (Matrix matrix : matrices) {
                for (int i = 0; i < matrix.getH(); i++) {
                    for (int j = 0; j < matrix.getW(); j++) {
                        matrix.addTo(i, j, -(matrix.getCell(i, j) / 2));
                    }
                }
            }
        }
    }

    public double overlayWithRotatedFilter(Matrix a, int i, int j, int numFilter){
        double res = 0;
        int h = matrices[numFilter].getH();
        int w = matrices[numFilter].getW();
        for (int ii = 0; ii < h; ii++) {
            for (int jj = 0; jj < w; jj++) {
                res += (i + ii < 0 || j + jj < 0 || i + ii == a.getH() || j + jj == a.getW() ? 0 : a.getCell(i + ii, j + jj)) * matrices[numFilter].getCell(h - ii - 1, w - jj - 1);
            }
        }
        return res;
    }

    public String getContent(){
        StringBuilder content = new StringBuilder(matrices.length);
        content.append(Tokens.SEP_OF_ELEMENTS);
        int h = matrices[0].getH();
        int w = matrices[0].getW();
        content.append(h);
        content.append(Tokens.SEP_OF_ELEMENTS);
        content.append(w);
        content.append(Tokens.SEP_OF_ELEMENTS);
        for (Matrix matrix : matrices) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    content.append(matrix.getCell(i, j));
                    content.append(Tokens.SEP_OF_ELEMENTS);
                }
            }
        }
        content.append(biasValue);
        return content.toString();
    }
}
