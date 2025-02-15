package com.ai.networks.convolutional.train.backpropagation;

import com.ai.networks.Tokens;
import com.ai.networks.convolutional.Layer;
import com.ai.networks.convolutional.Matrix;

final class TrainLayer extends Layer {
    private Matrix[] deltasOfInput;
    private Matrix[] deltasOFTemp;
    private TrainFilter[] trainFilters;

    private void initTrain() {
        trainFilters = new TrainFilter[filters.length];
        for (int i = 0; i < filters.length; i++) {
            trainFilters[i] = new TrainFilter(filters[i]);
        }
        filters = null;
        int l = inputs.length;
        deltasOfInput = new Matrix[l];
        deltasOFTemp = new Matrix[l];
        int ih = inputs[0].getH();
        int iw = inputs[0].getW();
        int oh = temp[0].getH();
        int ow = temp[0].getW();
        for (int i = 0; i < l; i++) {
            deltasOfInput[i] = new Matrix(ih, iw);
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
        for (int a = 0; a < temp.length; a++) {
            for (int i = -1; i <= temp[a].getH() - hf + 1; i++) {
                for (int j = -1; j <= temp[a].getW() - wf + 1; j++) {
                    for (int z = 0; z < deltasOfInput.length; z++) {
                        deltasOfInput[z].addTo(i + 1, j + 1, trainFilters[z].overlayWithRotatedFilter(deltasOFTemp[a], i, j, z));
                    }
                }
            }
        }
    }

    public StringBuilder getContent() {
        StringBuilder res = new StringBuilder(inputs.length);
        res.append(Tokens.SEP_OF_ELEMENTS);
        res.append(inputs[0].getH());
        res.append(Tokens.SEP_OF_ELEMENTS);
        res.append(inputs[0].getW());
        res.append(Tokens.SEP_OF_ELEMENTS);
        getContent(res);
        return res;
    }

    public void getContent(StringBuilder content) {
        content.append(filters[0].getMatrices()[0].getH());
        content.append(Tokens.SEP_OF_ELEMENTS);
        content.append(filters[0].getMatrices()[0].getW());
        for (TrainFilter trainFilter : trainFilters) {
            content.append(Tokens.SEP_OF_OBJECTS);
            content.append(trainFilter.getContent());
        }
    }

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
            outputs[i] = new Matrix(th / 2 + th % 2, tw / 2 + tw % 2);
            trainFilters[i] = new TrainFilter(sizeFilters, countIn);
        }
    }

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
    }
}
