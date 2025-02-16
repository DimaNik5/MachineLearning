package com.ai.networks.convolutional.train.backpropagation;

import com.ai.networks.Tokens;
import com.ai.networks.convolutional.Filter;
import com.ai.networks.convolutional.Matrix;

import java.util.Random;

/**
 * Класс TrainFilter дополняет сущность {@code Filter} возможностью обучения.
 * @author Никифоров Дмитрий
 * @since 1.1
 */
final class TrainFilter extends Filter {
    private Matrix[] lastMode;

    /**
     * Конструктор, создающий и заполняющий матрицы случайными значениями.
     * @param size размер матриц
     * @param count количество матриц
     */
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
        lastMode = new Matrix[matrices.length];
        for (int k = 0; k < matrices.length;k++) {
            lastMode[k] = new Matrix(matrices[0].getH(), matrices[0].getW());
        }
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
                    double wd =derSigmoid(matrices[k].getCell(i, j)) * speed * con + alpha * lastMode[k].getCell(i, j); //derSigmoid(matrices[k].getCell(i, j)) *
                    matrices[k].addTo(i, j, wd);
                    f = f || Math.abs(matrices[k].getCell(i, j)) > maxWeight;
                    lastMode[k].addTo(i, j, wd - lastMode[k].getCell(i, j));
                }
            }
        }
        double sum = 0;
        for (int i = 0; i < deltasLayer.getH(); i++) {
            for (int j = 0; j < deltasLayer.getW(); j++) {
                sum += deltasLayer.getCell(i, j);
            }
        }
        biasValue += sum;
        if(Math.abs(biasValue) > maxWeight) biasValue /= 10;
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

    /**
     * Метод получения значения производной функции активации (сигмоиды) в конкретной точке
     * @param res - нормализованое значение
     * @return значение производной
     */
    private double derSigmoid(double res){
        return (1 - res) * res;
    }

    /**
     * Метод для подсчета "перекрытия" определенного фильтра с данной матрицей по координатам.
     * Данный метод используется для подсчета дельт входной матрицы слоя.
     * В нем матрица дельт выходного слоя, дополненая нулями, сворачивается с
     * перевернутым на 180 фильтром.
     * @param a матрица дельт выходного слоя
     * @param i индекс по высоте
     * @param j индекс по ширине
     * @param numFilter индекс фильтра
     * @return сумма произведений "перекрытия" по координатам
     */
    public double overlayWithRotatedFilter(Matrix a, int i, int j, int numFilter){
        double res = 0;
        int h = matrices[numFilter].getH();
        int w = matrices[numFilter].getW();
        for (int ii = 0; ii < h; ii++) {
            for (int jj = 0; jj < w; jj++) {
                res += (i + ii < 0 || j + jj < 0 || i + ii >= a.getH() || j + jj >= a.getW() ? 0 : a.getCell(i + ii, j + jj)) * matrices[numFilter].getCell(h - ii - 1, w - jj - 1);
            }
        }
        return res;
    }

    /**
     * Используется для сохранения фильтра в файл
     * @return строка, содержащая информацию по фильтру
     */
    public String getContent(){
        StringBuilder content = new StringBuilder(String.valueOf(matrices.length));
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
