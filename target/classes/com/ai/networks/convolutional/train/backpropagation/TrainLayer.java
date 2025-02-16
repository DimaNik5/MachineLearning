package com.ai.networks.convolutional.train.backpropagation;

import com.ai.networks.Tokens;
import com.ai.networks.convolutional.Layer;
import com.ai.networks.convolutional.Matrix;

/**
 * Класс TrainLayer дополняет сущность {@code Layer} возможностью обучения.
 * @author Никифоров Дмитрий
 * @since 1.1
 */
final class TrainLayer extends Layer {
    private Matrix[] deltasOfInput;
    private Matrix[] deltasOFTemp;
    private TrainFilter[] trainFilters;

    /**
     * Метод инициализирует поля, предназначенные для обучения.
     * Используется при загрузке нейроной сети.
     */
    private void initTrain() {
        trainFilters = new TrainFilter[filters.length];
        for (int i = 0; i < filters.length; i++) {
            trainFilters[i] = new TrainFilter(filters[i]);
        }
        filters = null;
        int l = inputs.length;
        deltasOfInput = new Matrix[l];
        deltasOFTemp = new Matrix[trainFilters.length];
        int ih = inputs[0].getH();
        int iw = inputs[0].getW();
        int oh = temp[0].getH();
        int ow = temp[0].getW();
        for (int i = 0; i < l; i++) {
            deltasOfInput[i] = new Matrix(ih, iw);
        }
        for (int i = 0; i < trainFilters.length; i++) {
            deltasOFTemp[i] = new Matrix(oh, ow);
        }
    }

    @Override
    public void loadFirstFromString(String content) throws NumberFormatException {
        super.loadFirstFromString(content);
        initTrain();
    }

    @Override
    public void loadFromString(String content, Matrix[] input) throws NumberFormatException {
        super.loadFromString(content, input);
        initTrain();
    }

    @Override
    public void counting() {
        for (int i = 0; i < trainFilters.length; i++) {
            trainFilters[i].convolution(inputs, temp[i]);
            temp[i].normalize();
            temp[i].pooling(outputs[i]);
        }
    }

    public Matrix[] getDeltasOfInput() {
        return deltasOfInput;
    }

    /**
     * Метод используется для установки дельт выходного слоя.
     * Поскольку выходной слоя получается через max pooling,
     * дельты устанавливаются только в максимальные значения.
     * @param deltas массив дельт
     */
    public void setDeltas(Matrix[] deltas) {
        for (int k = 0; k < deltas.length; k++) {
            deltasOFTemp[k].setZero();
            for (int i = 0; i < deltas[k].getH(); i++) {
                for (int j = 0; j < deltas[k].getW(); j++) {
                    double max = outputs[k].getCell(i, j);
                    for (int l = 0; l < 4; l++) {
                        if (l == 1 && i + 1 == temp[k].getW()) continue;
                        if (temp[k].getCell(j + l / 2, i + l % 2) == max) {
                            deltasOFTemp[k].addTo(i + l % 2, j / 2, deltas[k].getCell(i, j));
                            break;
                        }
                    }
                }
            }
        }
    }

    public void correcting(double speed, double alpha, double maxWeight) {
        for (int i = 0; i < trainFilters.length; i++) {
            trainFilters[i].correcting(inputs, deltasOFTemp[i], speed, alpha, maxWeight);
        }

        int hf = trainFilters[0].getMatrices()[0].getH();
        int wf = trainFilters[0].getMatrices()[0].getW();
        for (Matrix matrix : deltasOfInput) {
            matrix.setZero();
        }
        for (int a = 0; a < deltasOFTemp.length; a++) {
            for (int i = -1; i <= deltasOFTemp[a].getH() - hf + 1; i++) {
                for (int j = -1; j <= deltasOFTemp[a].getW() - wf + 1; j++) {
                    for (int z = 0; z < deltasOfInput.length; z++) {
                        deltasOfInput[z].addTo(i + 1, j + 1, trainFilters[a].overlayWithRotatedFilter(deltasOFTemp[a], i, j, z));
                    }
                }
            }
        }
    }

    /**
     * Используется для сохранения первого слоя в файл
     * @return {@code StringBuilder}, содержащая информацию по слою
     * @see #getContent(StringBuilder)
     */
    public StringBuilder getContent() {
        StringBuilder res = new StringBuilder(String.valueOf(inputs.length));
        res.append(Tokens.SEP_OF_ELEMENTS);
        res.append(inputs[0].getH());
        res.append(Tokens.SEP_OF_ELEMENTS);
        res.append(inputs[0].getW());
        res.append(Tokens.SEP_OF_ELEMENTS);
        getContent(res);
        return res;
    }

    /**
     * Дополняет строчку информацией по слою.
     * @param content {@code StringBuilder}, содержащая информацию по предыдущим слоям
     * @see #getContent()
     */
    public void getContent(StringBuilder content) {
        content.append(trainFilters[0].getMatrices()[0].getH());
        content.append(Tokens.SEP_OF_ELEMENTS);
        content.append(trainFilters[0].getMatrices()[0].getW());
        for (TrainFilter trainFilter : trainFilters) {
            content.append(Tokens.SEP_OF_OBJECTS);
            content.append(trainFilter.getContent());
        }
    }

    /**
     * Метод для инициализации первого нового слоя случайными значениями
     * @param h высота входной матрицы
     * @param w ширина входной матрицы
     * @param countIn количесво входных матриц
     * @param countFilters количество фильтров
     * @param sizeFilters размер фильтров (ширина и высота)
     * @see #createNewLayer(Matrix[], int, int)
     */
    public void createNewLayer(int h, int w, int countIn, int countFilters, int sizeFilters) {
        inputs = new Matrix[countIn];
        trainFilters = new TrainFilter[countFilters];
        temp = new Matrix[countFilters];
        outputs = new Matrix[countFilters];
        for (int i = 0; i < countIn; i++) {
            inputs[i] = new Matrix(h, w);
        }
        int th = h - sizeFilters + 1;
        int tw = w - sizeFilters + 1;
        for (int i = 0; i < countFilters; i++) {
            temp[i] = new Matrix(th, tw);
            outputs[i] = new Matrix(th / 2, tw / 2);
            trainFilters[i] = new TrainFilter(sizeFilters, countIn);
        }
        deltasOfInput = new Matrix[countIn];
        deltasOFTemp = new Matrix[countFilters];
        int ih = inputs[0].getH();
        int iw = inputs[0].getW();
        int oh = temp[0].getH();
        int ow = temp[0].getW();
        for (int i = 0; i < countIn; i++) {
            deltasOfInput[i] = new Matrix(ih, iw);
        }
        for (int i = 0; i < countFilters; i++) {
            deltasOFTemp[i] = new Matrix(oh, ow);
        }
    }

    /**
     * Метод для инициализации нового слоя случайными значениями
     * @param in массив входных матриц
     * @param countFilters количество фильтров
     * @param sizeFilters размер фильтров (ширина и высота)
     * @see #createNewLayer(int, int, int, int, int)
     */
    public void createNewLayer(Matrix[] in, int countFilters, int sizeFilters) {
        inputs = in;
        trainFilters = new TrainFilter[countFilters];
        temp = new Matrix[countFilters];
        outputs = new Matrix[countFilters];

        int th = in[0].getH() - sizeFilters + 1;
        int tw = in[0].getW() - sizeFilters + 1;
        for (int i = 0; i < countFilters; i++) {
            temp[i] = new Matrix(th, tw);
            outputs[i] = new Matrix(th / 2 + th % 2, tw / 2 + tw % 2);
            trainFilters[i] = new TrainFilter(sizeFilters, in.length);
        }
        int l = inputs.length;
        deltasOfInput = new Matrix[l];
        deltasOFTemp = new Matrix[countFilters];
        int ih = inputs[0].getH();
        int iw = inputs[0].getW();
        int oh = temp[0].getH();
        int ow = temp[0].getW();
        for (int i = 0; i < l; i++) {
            deltasOfInput[i] = new Matrix(ih, iw);
        }
        for (int i = 0; i < countFilters; i++) {
            deltasOFTemp[i] = new Matrix(oh, ow);
        }
    }
}
