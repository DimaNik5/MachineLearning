package com.ai.networks.convolutional;

import com.ai.networks.Network;
import com.ai.networks.NetworkModels;

public class Convolution implements Network<Matrix, Matrix> {
    /**
     * MODEL - модель используемой нейронной сети.
     * Инициализируется при создании нового экземпляра
     */
    protected final NetworkModels MODEL;

    {
        MODEL = NetworkModels.CONVOLUTION;
    }

    protected Layer[] layers;

    @Override
    public int loadFromFile(String fileName) {
        return 0;
    }

    @Override
    public int loadFromString(String content) {
        return 0;
    }

    @Override
    public void setInput(Matrix[] input) {

    }

    @Override
    public void counting() {
        for (Layer layer : layers) {
            layer.counting();
        }
    }

    @Override
    public Matrix[] getOutput() {
        return layers[layers.length - 1].getOut();
    }

    @Override
    public NetworkModels getModel() {
        return MODEL;
    }
}
