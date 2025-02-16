package com.ai.networks.convolutional;

import com.ai.networks.Tokens;

import java.util.Arrays;

/**
 * Класс Layer хранит фильтры определенного слоя и входные, выходные матрицы.
 * Слои связаны между собой матрицами:<br>
 * M -> L -> M -> L -> M ...<br>
 * "M" - экземпляр матрицы.<br>
 * "L" - слой.<br>
 * "->" - поток данных.
 * @author Никифоров Дмитрий
 * @since 1.1
 */
public class Layer {

    protected Matrix[] inputs;
    protected Matrix[] temp;
    protected Matrix[] outputs;
    protected Filter[] filters;

    /**
     * Метод для загрузки первого слоя из строки.
     * @param content {@code String}, содержащая информацию о слое.
     * @throws NumberFormatException если информация не является числом
     * @see #loadFromString(String, Matrix[])
     */
    public void loadFirstFromString(String content) throws NumberFormatException{
        String[] layer = content.split(Tokens.SEP_OF_OBJECTS);
        int[] cont = Arrays.stream(layer[0].split(Tokens.SEP_OF_ELEMENTS)).mapToInt(Integer::parseInt).toArray();
        inputs = new Matrix[cont[0]];
        for(int i = 0; i < cont[0]; i++){
            inputs[i] = new Matrix(cont[1], cont[2]);
        }
        int count = layer.length - 1;
        int h = cont[1] - cont[3] + 1;
        int w = cont[2] - cont[4] + 1;
        filters = new Filter[count];
        temp = new Matrix[count];
        outputs = new Matrix[count];
        for (int i = 0; i < count; i++){
            temp[i] = new Matrix(h, w);
            outputs[i] = new Matrix(h / 2 + h % 2, w / 2 + w % 2);
            filters[i] = new Filter();
            filters[i].loadFromString(layer[i + 1]);
        }
    }

    /**
     * Метод для загрузки слоя из строки.
     * @param content {@code String}, содержащая информацию о слое.
     * @param input массив выходных {@code Matrix} предыдущего слоя
     * @throws NumberFormatException если информация не является числом
     * @see #loadFirstFromString(String)
     */
    public void loadFromString(String content, Matrix[] input) throws NumberFormatException{
        String[] layer = content.split(Tokens.SEP_OF_OBJECTS);
        double[] cont = Arrays.stream(layer[0].split(Tokens.SEP_OF_ELEMENTS)).mapToDouble(Double::parseDouble).toArray();
        inputs = input;
        int count = layer.length - 1;
        int h = (int)(inputs[0].getH() - cont[0]) + 1;
        int w = (int)(inputs[0].getW() - cont[1]) + 1;
        filters = new Filter[count];
        temp = new Matrix[count];
        outputs = new Matrix[count];
        for (int i = 0; i < count; i++){
            temp[i] = new Matrix(h, w);
            outputs[i] = new Matrix(h / 2 + h %2, w / 2 + w % 2);
            filters[i] = new Filter();
            filters[i].loadFromString(layer[i + 1]);
        }
    }

    public void counting() {
        for(int i = 0; i < filters.length; i++){
            filters[i].convolution(inputs, temp[i]);
            temp[i].normalize();
            temp[i].pooling(outputs[i]);
        }
    }

    public Matrix[] getOut() {
        return outputs;
    }

    public void setInputs(Matrix[] inputs){
        this.inputs = inputs;
    }
}
